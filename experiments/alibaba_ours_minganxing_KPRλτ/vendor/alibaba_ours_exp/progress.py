"""Crash-tolerant progress state, event and failure logging.

``status.json`` is atomically replaced after each mutation.  ``events.jsonl``
and ``failures.jsonl`` are append-only and guarded by a small cross-platform
lock file so multiple workers do not interleave JSON records.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback as traceback_module
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "skipped"}


def _format_duration(seconds: Any) -> str:
    if seconds is None:
        return "--:--:--"
    try:
        value = max(int(float(seconds)), 0)
    except (TypeError, ValueError, OverflowError):
        return "--:--:--"
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_console_progress(status: Mapping[str, Any], *, bar_width: int = 24) -> str:
    """Build one ASCII-only progress line that renders correctly in Windows CMD."""

    stage = str(status.get("stage") or "working")
    completed = int(status.get("completed") or 0)
    total_value = status.get("total")
    total = None if total_value is None else int(total_value)
    fraction_value = status.get("progress_fraction")
    fraction = 0.0 if fraction_value is None else min(max(float(fraction_value), 0.0), 1.0)
    filled = min(max(int(round(fraction * bar_width)), 0), bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = 100.0 * fraction
    parts = [f"[{stage}]", f"[{bar}]", f"{percent:6.2f}%"]

    details = status.get("details") if isinstance(status.get("details"), Mapping) else {}
    is_one_pass = stage == "prepare-one-pass"
    if (stage.startswith("prepare-pass") or is_one_pass) and total and total > 1:
        source_bytes = int(details.get("source_bytes") or (total if is_one_pass else total // 2))
        byte_offset = int(details.get("byte_offset") or 0)
        pass_number = 2 if stage.endswith("pass2") else 1
        pass_total = 1 if is_one_pass else 2
        pass_fraction = min(max(byte_offset / max(source_bytes, 1), 0.0), 1.0)
        parts.extend(
            [
                f"pass={pass_number}/{pass_total}:{100.0 * pass_fraction:5.1f}%",
                f"data={byte_offset / (1024 ** 3):.2f}/{source_bytes / (1024 ** 3):.2f}GiB",
            ]
        )
        rows = details.get("rows_scanned")
        if rows is not None:
            parts.append(f"rows={int(rows):,}")
        rate = status.get("rate_per_second")
        if rate:
            parts.append(f"speed={float(rate) / (1024 ** 2):.2f}MiB/s")
    elif total is not None:
        parts.append(f"units={completed}/{total}")
        rate = status.get("rate_per_second")
        if rate:
            parts.append(f"rate={float(rate):.3f}/s")

    for key, label in (
        ("history_length", "L"),
        ("split", "split"),
        ("origin_id", "origin"),
        ("epoch", "epoch"),
    ):
        if details.get(key) is not None:
            parts.append(f"{label}={details[key]}")
    if status.get("forecast_horizon") is not None:
        parts.append(f"h={int(status['forecast_horizon'])}")
    parts.append(f"elapsed={_format_duration(status.get('elapsed_seconds'))}")
    parts.append(f"ETA={_format_duration(status.get('eta_seconds'))}")
    message = status.get("message")
    if (
        message
        and str(message).isascii()
        and not stage.startswith("prepare-pass")
        and not is_one_pass
    ):
        parts.append(str(message))
    return " | ".join(parts)


class ProgressError(RuntimeError):
    """Base class for progress tracking failures."""


class ProgressStateError(ProgressError):
    """Raised for an invalid lifecycle transition."""


class ProgressCorruptError(ProgressError):
    """Raised when an existing status file is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _jsonable(value),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


class _FileLock(AbstractContextManager["_FileLock"]):
    def __init__(self, path: Path, timeout: float = 30.0, poll_interval: float = 0.05) -> None:
        self.path = path
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self._handle: Any | None = None

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write(b"\0")
            self._handle.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._lock_once()
                return self
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise TimeoutError(f"timed out acquiring progress lock {self.path}")
                time.sleep(self.poll_interval)

    def _lock_once(self) -> None:
        assert self._handle is not None
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _append_jsonl_unlocked(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            _jsonable(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_jsonl(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    lock_path: str | Path | None = None,
    lock_timeout: float = 30.0,
) -> Path:
    destination = Path(path)
    lock = Path(lock_path) if lock_path is not None else destination.with_suffix(destination.suffix + ".lock")
    with _FileLock(lock, timeout=lock_timeout):
        _append_jsonl_unlocked(destination, record)
    return destination


def read_jsonl(path: str | Path, *, strict: bool = True) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not a JSON object")
                records.append(value)
            except (json.JSONDecodeError, ValueError) as exc:
                if strict:
                    raise ProgressCorruptError(
                        f"invalid JSONL record at {source}:{line_number}: {exc}"
                    ) from exc
    return records


def read_status(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProgressCorruptError(f"cannot read status {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProgressCorruptError(f"status {source} is not a JSON object")
    return value


class ProgressTracker:
    """Manage one run's status and append-only audit streams."""

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        lock_timeout: float = 30.0,
    ) -> None:
        if not run_id or not str(run_id).strip():
            raise ValueError("run_id must be non-empty")
        self.directory = Path(directory)
        self.run_id = str(run_id)
        self.lock_timeout = float(lock_timeout)
        self.status_path = self.directory / "status.json"
        self.events_path = self.directory / "events.jsonl"
        self.failures_path = self.directory / "failures.jsonl"
        self.lock_path = self.directory / ".progress.lock"
        self._console_last_width = 0
        self._console_last_stage: str | None = None

    @property
    def status(self) -> dict[str, Any] | None:
        value = read_status(self.status_path)
        if value is not None and value.get("run_id") != self.run_id:
            raise ProgressStateError(
                f"status run_id {value.get('run_id')!r} != tracker run_id {self.run_id!r}"
            )
        return value

    def _new_status(
        self,
        *,
        total: int | None,
        stage: str | None,
        message: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        now = _utc_now()
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "running",
            "stage": stage,
            "completed": 0,
            "total": total,
            "progress_fraction": 0.0 if total else None,
            "progress_percent": 0.0 if total else None,
            "started_at": now,
            "updated_at": now,
            "ended_at": None,
            "elapsed_seconds": 0.0,
            "active_elapsed_before_session": 0.0,
            "session_started_at": now,
            "session_completed_baseline": 0,
            "rate_per_second": None,
            "eta_seconds": None,
            "message": message,
            "metrics": {},
            "details": {},
            "metadata": dict(metadata or {}),
            "summary": {},
            "failure_count": 0,
            "last_event_seq": 0,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        }

    def _event(self, status: Mapping[str, Any], event_type: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "seq": int(status.get("last_event_seq", 0)),
            "timestamp": status["updated_at"],
            "run_id": self.run_id,
            "event": event_type,
            "status": status["status"],
            "stage": status.get("stage"),
            "completed": status.get("completed"),
            "total": status.get("total"),
            "progress_fraction": status.get("progress_fraction"),
            "forecast_horizon": status.get("forecast_horizon"),
            "pid": os.getpid(),
            "payload": dict(payload or {}),
        }

    def _refresh_derived(self, status: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        status["updated_at"] = now.isoformat().replace("+00:00", "Z")
        session_started_at = status.get("session_started_at") or status.get("started_at")
        session_elapsed = (
            max((now - _parse_utc(session_started_at)).total_seconds(), 0.0)
            if session_started_at
            else 0.0
        )
        elapsed = float(status.get("active_elapsed_before_session") or 0.0) + session_elapsed
        status["elapsed_seconds"] = elapsed
        completed = int(status.get("completed") or 0)
        total = status.get("total")
        if total is not None:
            total = int(total)
            fraction = min(max(completed / total, 0.0), 1.0) if total > 0 else 1.0
            status["progress_fraction"] = fraction
            status["progress_percent"] = 100.0 * fraction
        else:
            status["progress_fraction"] = None
            status["progress_percent"] = None
        baseline = int(status.get("session_completed_baseline") or 0)
        session_completed = max(completed - baseline, 0)
        rate = session_completed / session_elapsed if session_completed > 0 and session_elapsed > 0.0 else None
        status["rate_per_second"] = rate
        if rate and total is not None and total >= completed:
            status["eta_seconds"] = (total - completed) / rate
        else:
            status["eta_seconds"] = None
        status["pid"] = os.getpid()

    def _render_console(self, status: Mapping[str, Any], event_type: str) -> None:
        stream = sys.stdout
        is_terminal = bool(getattr(stream, "isatty", lambda: False)())
        if not is_terminal and event_type not in {
            "started",
            "resumed",
            "completed",
            "failed",
            "cancelled",
            "skipped",
        }:
            return
        stage = str(status.get("stage") or "working")
        if is_terminal and self._console_last_stage not in {None, stage}:
            stream.write("\n")
            self._console_last_width = 0
        line = format_console_progress(status)
        width = shutil.get_terminal_size(fallback=(140, 24)).columns
        if width > 20 and len(line) >= width:
            line = line[: width - 4] + "..."
        terminal_event = status.get("status") in TERMINAL_STATUSES or event_type in TERMINAL_STATUSES
        if is_terminal:
            padded = line.ljust(max(self._console_last_width, len(line)))
            stream.write("\r" + padded + ("\n" if terminal_event else ""))
            self._console_last_width = 0 if terminal_event else len(line)
        else:
            stream.write(line + "\n")
        stream.flush()
        self._console_last_stage = stage

    def _publish(
        self,
        status: dict[str, Any],
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        status["last_event_seq"] = int(status.get("last_event_seq", 0)) + 1
        self._refresh_derived(status)
        _append_jsonl_unlocked(self.events_path, self._event(status, event_type, payload))
        atomic_write_json(self.status_path, status)
        self._render_console(status, event_type)
        return dict(status)

    def start(
        self,
        *,
        total: int | None = None,
        stage: str | None = None,
        message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        resume: bool = False,
        reset: bool = False,
    ) -> dict[str, Any]:
        if total is not None and int(total) < 0:
            raise ValueError("total must be non-negative")
        with _FileLock(self.lock_path, timeout=self.lock_timeout):
            existing = read_status(self.status_path)
            if existing is not None and existing.get("run_id") != self.run_id:
                raise ProgressStateError("existing status belongs to another run_id")
            if existing is not None and not reset:
                if not resume:
                    raise ProgressStateError("status already exists; pass resume=True or reset=True")
                if existing.get("status") in TERMINAL_STATUSES:
                    raise ProgressStateError(
                        f"cannot resume terminal status {existing.get('status')!r}"
                    )
                status = dict(existing)
                status["status"] = "running"
                status["active_elapsed_before_session"] = float(
                    status.get("elapsed_seconds") or 0.0
                )
                status["session_started_at"] = _utc_now()
                status["session_completed_baseline"] = int(status.get("completed") or 0)
                if total is not None:
                    status["total"] = int(total)
                if stage is not None:
                    status["stage"] = str(stage)
                if message is not None:
                    status["message"] = str(message)
                if metadata:
                    status.setdefault("metadata", {}).update(dict(metadata))
                return self._publish(status, "resumed", {"message": message})
            status = self._new_status(
                total=None if total is None else int(total),
                stage=None if stage is None else str(stage),
                message=None if message is None else str(message),
                metadata=metadata,
            )
            return self._publish(status, "started", {"reset": bool(reset)})

    def update(
        self,
        *,
        completed: int | None = None,
        increment: int | None = None,
        total: int | None = None,
        stage: str | None = None,
        message: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        forecast_horizon: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if completed is not None and increment is not None:
            raise ValueError("supply completed or increment, not both")
        with _FileLock(self.lock_path, timeout=self.lock_timeout):
            status = read_status(self.status_path)
            if status is None:
                raise ProgressStateError("start must be called before update")
            if status.get("run_id") != self.run_id:
                raise ProgressStateError("existing status belongs to another run_id")
            if status.get("status") != "running":
                raise ProgressStateError(f"cannot update status {status.get('status')!r}")
            if completed is not None:
                status["completed"] = int(completed)
            elif increment is not None:
                status["completed"] = int(status.get("completed") or 0) + int(increment)
            if int(status.get("completed") or 0) < 0:
                raise ValueError("completed work must be non-negative")
            if total is not None:
                if int(total) < 0:
                    raise ValueError("total must be non-negative")
                status["total"] = int(total)
            if stage is not None:
                status["stage"] = str(stage)
            if message is not None:
                status["message"] = str(message)
            if metrics:
                status.setdefault("metrics", {}).update(dict(metrics))
            if extra:
                status.setdefault("details", {}).update(dict(extra))
                if extra.get("rebase_rate"):
                    status["active_elapsed_before_session"] = float(
                        status.get("elapsed_seconds") or 0.0
                    )
                    status["session_started_at"] = _utc_now()
                    status["session_completed_baseline"] = int(
                        status.get("completed") or 0
                    )
            if forecast_horizon is not None:
                if int(forecast_horizon) < 0:
                    raise ValueError("forecast_horizon must be non-negative")
                status["forecast_horizon"] = int(forecast_horizon)
            payload = {"message": message, "metrics": dict(metrics or {}), **dict(extra or {})}
            return self._publish(status, "progress", payload)

    def heartbeat(self, *, message: str | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with _FileLock(self.lock_path, timeout=self.lock_timeout):
            status = read_status(self.status_path)
            if status is None or status.get("status") != "running":
                raise ProgressStateError("heartbeat requires a running status")
            if message is not None:
                status["message"] = str(message)
            return self._publish(status, "heartbeat", {"message": message, **dict(extra or {})})

    def complete(
        self,
        *,
        summary: Mapping[str, Any] | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        with _FileLock(self.lock_path, timeout=self.lock_timeout):
            status = read_status(self.status_path)
            if status is None:
                raise ProgressStateError("start must be called before complete")
            if status.get("status") != "running":
                raise ProgressStateError(f"cannot complete status {status.get('status')!r}")
            status["status"] = "completed"
            status["summary"] = dict(summary or {})
            if message is not None:
                status["message"] = str(message)
            if status.get("total") is not None:
                status["completed"] = max(
                    int(status.get("completed") or 0), int(status["total"])
                )
            status["ended_at"] = _utc_now()
            return self._publish(status, "completed", {"summary": dict(summary or {})})

    def fail(
        self,
        error: BaseException | str,
        *,
        context: Mapping[str, Any] | None = None,
        fatal: bool = True,
    ) -> dict[str, Any]:
        if isinstance(error, BaseException):
            error_type = type(error).__name__
            message = str(error)
            trace = "".join(traceback_module.format_exception(type(error), error, error.__traceback__))
        else:
            error_type = "Error"
            message = str(error)
            trace = None
        with _FileLock(self.lock_path, timeout=self.lock_timeout):
            status = read_status(self.status_path)
            if status is None:
                status = self._new_status(total=None, stage=None, message=None, metadata=None)
            if status.get("run_id") != self.run_id:
                raise ProgressStateError("existing status belongs to another run_id")
            status["failure_count"] = int(status.get("failure_count", 0)) + 1
            status["last_event_seq"] = int(status.get("last_event_seq", 0)) + 1
            self._refresh_derived(status)
            failure = {
                "schema_version": SCHEMA_VERSION,
                "seq": status["last_event_seq"],
                "timestamp": status["updated_at"],
                "run_id": self.run_id,
                "fatal": bool(fatal),
                "error_type": error_type,
                "message": message,
                "traceback": trace,
                "context": dict(context or {}),
                "stage": status.get("stage"),
                "forecast_horizon": status.get("forecast_horizon"),
                "pid": os.getpid(),
            }
            _append_jsonl_unlocked(self.failures_path, failure)
            if fatal:
                status["status"] = "failed"
                status["ended_at"] = _utc_now()
                status["message"] = message
            _append_jsonl_unlocked(
                self.events_path,
                self._event(status, "failed" if fatal else "failure_recorded", failure),
            )
            atomic_write_json(self.status_path, status)
            self._render_console(status, "failed" if fatal else "failure_recorded")
            return dict(status)

    def cancel(self, *, message: str | None = None) -> dict[str, Any]:
        return self._terminate("cancelled", message)

    def skip(self, *, message: str | None = None) -> dict[str, Any]:
        return self._terminate("skipped", message)

    def _terminate(self, terminal_status: str, message: str | None) -> dict[str, Any]:
        with _FileLock(self.lock_path, timeout=self.lock_timeout):
            status = read_status(self.status_path)
            if status is None:
                status = self._new_status(total=None, stage=None, message=message, metadata=None)
            if status.get("status") in TERMINAL_STATUSES:
                raise ProgressStateError(f"status is already terminal: {status.get('status')!r}")
            status["status"] = terminal_status
            status["ended_at"] = _utc_now()
            if message is not None:
                status["message"] = str(message)
            return self._publish(status, terminal_status, {"message": message})

    def track(
        self,
        iterable: Iterable[Any],
        *,
        total: int | None = None,
        stage: str | None = None,
        update_every: int = 1,
    ) -> Iterator[Any]:
        """Yield an iterable and increment the running status periodically."""

        if update_every < 1:
            raise ValueError("update_every must be positive")
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)  # type: ignore[arg-type]
        current = self.status
        if current is None:
            self.start(total=total, stage=stage)
        elif current.get("status") != "running":
            raise ProgressStateError("track requires a running status")
        for index, item in enumerate(iterable, start=1):
            yield item
            if index % update_every == 0:
                self.update(increment=update_every, total=total, stage=stage)
        remainder = index % update_every if "index" in locals() else 0
        if remainder:
            self.update(increment=remainder, total=total, stage=stage)


__all__ = [
    "ProgressCorruptError",
    "ProgressError",
    "ProgressStateError",
    "ProgressTracker",
    "append_jsonl",
    "atomic_write_json",
    "format_console_progress",
    "read_jsonl",
    "read_status",
]
