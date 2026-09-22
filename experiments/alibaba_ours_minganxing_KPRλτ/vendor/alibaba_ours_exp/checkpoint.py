"""Atomic, hash-validated checkpoints for training and recursive forecasting.

Checkpoints are single compressed NPZ files.  Arbitrarily nested mappings,
lists and tuples are represented by a JSON manifest while NumPy arrays remain
native NPZ members; loading never enables pickle.  The published file embeds
configuration, data and state hashes and is written through ``os.replace``.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
RUN_IDENTITY_SCHEMA_VERSION = 1
ARTIFACT_GUARD_SCHEMA_VERSION = 1
_METADATA_KEY = "__checkpoint_metadata_json__"
_STATE_MANIFEST_KEY = "__checkpoint_state_manifest_json__"


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class CheckpointMismatchError(CheckpointError):
    """Raised when expected configuration, data or checkpoint kind differs."""


class CheckpointCorruptError(CheckpointError):
    """Raised when a checkpoint is incomplete or fails integrity validation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise TypeError("object-dtype arrays are not canonical JSON values")
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: canonical_json(item))
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise TypeError("NaN and infinity are not valid canonical values")
        return value
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return a stable UTF-8 JSON representation suitable for hashing."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_hash(config: Any) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def hash_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: str | Path | Sequence[str | Path]) -> str:
    """Hash file contents and relative names for one file or directory set."""

    supplied = [paths] if isinstance(paths, (str, Path)) else list(paths)
    files: list[tuple[str, Path]] = []
    for supplied_path in supplied:
        path = Path(supplied_path)
        if path.is_file():
            files.append((path.name, path))
        elif path.is_dir():
            files.extend(
                (child.relative_to(path).as_posix(), child)
                for child in path.rglob("*")
                if child.is_file()
            )
        else:
            raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for relative, path in sorted(files, key=lambda item: (item[0], str(item[1]))):
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(hash_file(path)))
    return digest.hexdigest()


def stable_file_fingerprint(
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
    include_content_hash: bool = False,
) -> dict[str, Any]:
    """Return a cheap, stable fingerprint for an input file.

    Large raw Alibaba files are intentionally identified by logical path,
    byte size and nanosecond mtime.  ``include_content_hash`` is reserved for
    the much smaller prepared caches and produced artifacts.
    """

    source = Path(path).resolve()
    logical_path: str
    if relative_to is not None:
        try:
            logical_path = source.relative_to(Path(relative_to).resolve()).as_posix()
        except ValueError:
            logical_path = source.as_posix()
    else:
        logical_path = source.as_posix()
    result: dict[str, Any] = {
        "path": logical_path,
        "exists": source.is_file(),
    }
    if source.is_file():
        stat = source.stat()
        result.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        if include_content_hash:
            result["sha256"] = hash_file(source)
    return result


def code_fingerprint(experiment_root: str | Path) -> dict[str, Any]:
    """Hash only stable source files that define this experiment.

    Runtime outputs, tests, bytecode and cache directories are excluded by
    construction.  The core ``workload_fmm`` package is included because the
    experiment imports its DTW implementation.
    """

    root = Path(experiment_root).resolve()
    candidates: list[tuple[str, Path]] = []
    entry = root / "run_experiment.py"
    if entry.is_file():
        candidates.append(("run_experiment.py", entry))
    experiment_src = root / "src"
    if experiment_src.is_dir():
        candidates.extend(
            (f"src/{path.relative_to(experiment_src).as_posix()}", path)
            for path in experiment_src.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    our_root = root.parent.parent
    core_src = our_root / "src" / "workload_fmm"
    if core_src.is_dir():
        candidates.extend(
            (f"core/workload_fmm/{path.relative_to(core_src).as_posix()}", path)
            for path in core_src.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for logical, path in sorted(candidates, key=lambda item: item[0]):
        content_digest = hash_file(path)
        encoded = logical.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(content_digest))
        files.append({"path": logical, "sha256": content_digest})
    if not files:
        raise FileNotFoundError(f"no Python sources found below {root}")
    return {"sha256": digest.hexdigest(), "files": files}


def make_run_identity(
    *,
    raw_config: Any,
    experiment_root: str | Path,
    project_root: str | Path,
    input_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Build the immutable identity shared by every resume layer."""

    code = code_fingerprint(experiment_root)
    inputs = {
        str(name): stable_file_fingerprint(path, relative_to=project_root)
        for name, path in sorted(input_paths.items())
    }
    payload: dict[str, Any] = {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "config_hash": config_hash(raw_config),
        "code_hash": code["sha256"],
        "code_files": code["files"],
        "input_fingerprints": inputs,
    }
    payload["identity_hash"] = config_hash(payload)
    return payload


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def ensure_run_identity(
    run_dir: str | Path,
    expected: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    """Atomically bind a run directory to exactly one run identity.

    A custom run id therefore cannot bypass configuration, data or code
    compatibility.  Legacy output directories without an identity are never
    accepted for ``--resume``.
    """

    directory = Path(run_dir)
    path = directory / "run_identity.json"
    expected_value = dict(_jsonable(dict(expected)))
    expected_digest = str(expected_value.get("identity_hash", ""))
    calculated = config_hash({k: v for k, v in expected_value.items() if k != "identity_hash"})
    if expected_digest != calculated:
        raise CheckpointMismatchError("expected run identity has an invalid identity_hash")
    if path.exists():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptError(f"cannot read run identity {path}: {exc}") from exc
        observed_digest = str(observed.get("identity_hash", ""))
        observed_calculated = config_hash(
            {k: v for k, v in observed.items() if k != "identity_hash"}
        )
        if observed_digest != observed_calculated:
            raise CheckpointCorruptError(f"run identity is corrupt: {path}")
        if observed_digest != expected_digest:
            raise CheckpointMismatchError(
                "run identity mismatch; configuration, relevant input data, or source code changed "
                f"(stored={observed_digest}, expected={expected_digest})"
            )
        return observed
    if directory.exists() and any(directory.iterdir()):
        raise CheckpointMismatchError(
            f"legacy run directory has no run_identity.json and cannot be reused: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, expected_value)
    return expected_value


def artifact_bundle_fingerprint(
    artifacts: Mapping[str, str | Path],
) -> tuple[dict[str, Any], str]:
    details: dict[str, Any] = {}
    for name, path in sorted(artifacts.items()):
        logical_name = str(name)
        fingerprint = stable_file_fingerprint(path, include_content_hash=True)
        # The mapping key is the portable logical path.  Never persist the
        # machine-specific absolute artifact path in a resume guard.
        fingerprint["path"] = logical_name
        fingerprint.pop("mtime_ns", None)
        details[logical_name] = fingerprint
    missing = [name for name, value in details.items() if not value["exists"]]
    if missing:
        raise FileNotFoundError(f"artifact bundle is incomplete: {missing}")
    return details, config_hash(details)


def write_artifact_guard(
    path: str | Path,
    *,
    stage: str,
    run_identity_digest: str,
    relevant_data_digest: str,
    artifacts: Mapping[str, str | Path],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    details, artifact_digest = artifact_bundle_fingerprint(artifacts)
    payload = {
        "schema_version": ARTIFACT_GUARD_SCHEMA_VERSION,
        "stage": str(stage),
        "run_identity_hash": str(run_identity_digest),
        "relevant_data_hash": str(relevant_data_digest),
        "artifact_hash": artifact_digest,
        "artifacts": details,
        "extra": dict(_jsonable(dict(extra or {}))),
    }
    payload["guard_hash"] = config_hash(payload)
    _atomic_write_json(Path(path), payload)
    return payload


def validate_artifact_guard(
    path: str | Path,
    *,
    stage: str,
    run_identity_digest: str,
    relevant_data_digest: str,
    artifacts: Mapping[str, str | Path],
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise CheckpointMismatchError(f"artifact guard is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptError(f"cannot read artifact guard {source}: {exc}") from exc
    stored_guard = str(payload.get("guard_hash", ""))
    calculated_guard = config_hash({k: v for k, v in payload.items() if k != "guard_hash"})
    if stored_guard != calculated_guard:
        raise CheckpointCorruptError(f"artifact guard is corrupt: {source}")
    expectations = {
        "schema_version": ARTIFACT_GUARD_SCHEMA_VERSION,
        "stage": str(stage),
        "run_identity_hash": str(run_identity_digest),
        "relevant_data_hash": str(relevant_data_digest),
    }
    for key, expected_value in expectations.items():
        if payload.get(key) != expected_value:
            raise CheckpointMismatchError(
                f"artifact guard {key} mismatch: stored={payload.get(key)!r}, "
                f"expected={expected_value!r}"
            )
    _, current_digest = artifact_bundle_fingerprint(artifacts)
    if payload.get("artifact_hash") != current_digest:
        raise CheckpointMismatchError(
            f"artifact contents changed since completion: {source}"
        )
    return payload


def _update_data_digest(digest: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise TypeError("object-dtype arrays cannot be hashed safely")
        contiguous = np.ascontiguousarray(value)
        digest.update(b"array\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(canonical_json(list(contiguous.shape)).encode("ascii"))
        digest.update(memoryview(contiguous).cast("B"))
        return
    if isinstance(value, np.generic):
        _update_data_digest(digest, value.item())
        return
    if isinstance(value, Path):
        digest.update(b"path\0")
        digest.update(bytes.fromhex(hash_paths(value)))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            _update_data_digest(digest, str(key))
            _update_data_digest(digest, item)
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_data_digest(digest, item)
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
        return
    digest.update(b"scalar\0")
    digest.update(canonical_json(value).encode("utf-8"))


def data_hash(data: Any) -> str:
    """Hash arrays/nested data; pass ``Path`` objects to hash file contents."""

    digest = hashlib.sha256()
    _update_data_digest(digest, data)
    return digest.hexdigest()


def _pack_state(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise TypeError("checkpoint arrays may not use object dtype")
        key = f"array_{len(arrays):06d}"
        arrays[key] = np.ascontiguousarray(value)
        return {"__kind__": "ndarray", "key": key}
    if isinstance(value, np.generic):
        return _pack_state(value.item(), arrays)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__kind__": "dataclass",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _pack_state(dataclasses.asdict(value), arrays),
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("checkpoint mapping keys must be strings")
        return {
            "__kind__": "mapping",
            "items": {key: _pack_state(item, arrays) for key, item in sorted(value.items())},
        }
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_pack_state(item, arrays) for item in value]}
    if isinstance(value, list):
        return {"__kind__": "list", "items": [_pack_state(item, arrays) for item in value]}
    if isinstance(value, Path):
        return {"__kind__": "path", "value": value.as_posix()}
    if isinstance(value, bytes):
        return {"__kind__": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise TypeError("checkpoint JSON scalars may not be NaN or infinity")
        return value
    raise TypeError(f"unsupported checkpoint state type: {type(value).__name__}")


def _unpack_state(manifest: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if not isinstance(manifest, Mapping) or "__kind__" not in manifest:
        return manifest
    kind = manifest["__kind__"]
    if kind == "ndarray":
        key = str(manifest["key"])
        if key not in arrays:
            raise CheckpointCorruptError(f"state array {key!r} is missing")
        return np.array(arrays[key], copy=True)
    if kind == "mapping":
        items = manifest.get("items")
        if not isinstance(items, Mapping):
            raise CheckpointCorruptError("mapping state manifest has invalid items")
        return {str(key): _unpack_state(item, arrays) for key, item in items.items()}
    if kind == "dataclass":
        # Dataclasses are intentionally returned as mappings: reconstruction of
        # arbitrary importable classes would weaken portability and safety.
        return _unpack_state(manifest.get("value"), arrays)
    if kind == "tuple":
        return tuple(_unpack_state(item, arrays) for item in manifest["items"])
    if kind == "list":
        return [_unpack_state(item, arrays) for item in manifest["items"]]
    if kind == "path":
        return Path(str(manifest["value"]))
    if kind == "bytes":
        return base64.b64decode(str(manifest["value"]).encode("ascii"))
    raise CheckpointCorruptError(f"unknown state manifest kind: {kind!r}")


def _state_hash(manifest: Any, arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(manifest).encode("utf-8"))
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(canonical_json(list(array.shape)).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointMetadata:
    kind: str
    created_at: str
    config_hash: str | None
    data_hash: str | None
    state_hash: str
    schema_version: int = SCHEMA_VERSION
    run_id: str | None = None
    epoch: int | None = None
    global_step: int | None = None
    forecast_horizon: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointMetadata":
        try:
            return cls(
                schema_version=int(value["schema_version"]),
                kind=str(value["kind"]),
                created_at=str(value["created_at"]),
                config_hash=value.get("config_hash"),
                data_hash=value.get("data_hash"),
                state_hash=str(value["state_hash"]),
                run_id=value.get("run_id"),
                epoch=None if value.get("epoch") is None else int(value["epoch"]),
                global_step=None if value.get("global_step") is None else int(value["global_step"]),
                forecast_horizon=(
                    None
                    if value.get("forecast_horizon") is None
                    else int(value["forecast_horizon"])
                ),
                extra=dict(value.get("extra") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruptError(f"invalid checkpoint metadata: {exc}") from exc


@dataclass(frozen=True)
class LoadedCheckpoint:
    metadata: CheckpointMetadata
    state: Any
    path: Path


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


def _resolve_hash(
    *,
    supplied_hash: str | None,
    supplied_value: Any | None,
    hash_function: Any,
    name: str,
) -> str | None:
    if supplied_hash is not None and supplied_value is not None:
        calculated = hash_function(supplied_value)
        if calculated != supplied_hash:
            raise CheckpointMismatchError(f"supplied {name} hash does not match supplied {name}")
        return supplied_hash
    if supplied_hash is not None:
        normalized = str(supplied_hash).lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError(f"{name} hash must be a SHA-256 hex digest")
        return normalized
    return None if supplied_value is None else hash_function(supplied_value)


def save_checkpoint(
    path: str | Path,
    state: Any,
    *,
    kind: str,
    config: Any | None = None,
    config_digest: str | None = None,
    data: Any | None = None,
    data_digest: str | None = None,
    run_id: str | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    forecast_horizon: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> CheckpointMetadata:
    """Serialize and atomically publish a validated checkpoint."""

    if not kind or not str(kind).strip():
        raise ValueError("kind must be non-empty")
    arrays: dict[str, np.ndarray] = {}
    manifest = _pack_state(state, arrays)
    state_digest = _state_hash(manifest, arrays)
    metadata = CheckpointMetadata(
        kind=str(kind),
        created_at=_utc_now(),
        config_hash=_resolve_hash(
            supplied_hash=config_digest,
            supplied_value=config,
            hash_function=config_hash,
            name="config",
        ),
        data_hash=_resolve_hash(
            supplied_hash=data_digest,
            supplied_value=data,
            hash_function=data_hash,
            name="data",
        ),
        state_hash=state_digest,
        run_id=run_id,
        epoch=None if epoch is None else int(epoch),
        global_step=None if global_step is None else int(global_step),
        forecast_horizon=None if forecast_horizon is None else int(forecast_horizon),
        extra=dict(_jsonable(dict(extra or {}))),
    )
    payload = dict(arrays)
    payload[_METADATA_KEY] = np.asarray(canonical_json(metadata.to_dict()))
    payload[_STATE_MANIFEST_KEY] = np.asarray(canonical_json(manifest))

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
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
    return metadata


def _read_json_scalar(archive: Any, key: str) -> Any:
    if key not in archive.files:
        raise CheckpointCorruptError(f"checkpoint member {key!r} is missing")
    value = archive[key]
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise CheckpointCorruptError(f"checkpoint member {key!r} is not a JSON scalar")
    try:
        raw = value.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptError(f"invalid JSON in checkpoint member {key!r}") from exc


def _expected_digest(value: Any | None, digest: str | None, function: Any, name: str) -> str | None:
    return _resolve_hash(
        supplied_hash=digest,
        supplied_value=value,
        hash_function=function,
        name=name,
    )


def load_checkpoint(
    path: str | Path,
    *,
    expected_kind: str | None = None,
    expected_config: Any | None = None,
    expected_config_digest: str | None = None,
    expected_data: Any | None = None,
    expected_data_digest: str | None = None,
    verify_state: bool = True,
) -> LoadedCheckpoint:
    """Load a checkpoint and enforce all supplied compatibility expectations."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with np.load(source, allow_pickle=False) as archive:
            metadata = CheckpointMetadata.from_dict(_read_json_scalar(archive, _METADATA_KEY))
            manifest = _read_json_scalar(archive, _STATE_MANIFEST_KEY)
            if metadata.schema_version != SCHEMA_VERSION:
                raise CheckpointMismatchError(
                    f"checkpoint schema {metadata.schema_version} != supported {SCHEMA_VERSION}"
                )
            arrays = {
                key: np.array(archive[key], copy=True)
                for key in archive.files
                if key not in {_METADATA_KEY, _STATE_MANIFEST_KEY}
            }
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointCorruptError(f"cannot read checkpoint {source}: {exc}") from exc

    if verify_state:
        calculated_state_hash = _state_hash(manifest, arrays)
        if calculated_state_hash != metadata.state_hash:
            raise CheckpointCorruptError("checkpoint state hash mismatch")
    if expected_kind is not None and metadata.kind != expected_kind:
        raise CheckpointMismatchError(
            f"checkpoint kind {metadata.kind!r} != expected {expected_kind!r}"
        )
    expected_config_hash = _expected_digest(
        expected_config, expected_config_digest, config_hash, "config"
    )
    expected_data_hash = _expected_digest(expected_data, expected_data_digest, data_hash, "data")
    if expected_config_hash is not None and metadata.config_hash != expected_config_hash:
        raise CheckpointMismatchError(
            f"config hash mismatch: checkpoint={metadata.config_hash}, expected={expected_config_hash}"
        )
    if expected_data_hash is not None and metadata.data_hash != expected_data_hash:
        raise CheckpointMismatchError(
            f"data hash mismatch: checkpoint={metadata.data_hash}, expected={expected_data_hash}"
        )
    state = _unpack_state(manifest, arrays)
    return LoadedCheckpoint(metadata=metadata, state=state, path=source)


def save_training_checkpoint(
    path: str | Path,
    *,
    model_state: Any,
    config: Any | None = None,
    config_digest: str | None = None,
    data: Any | None = None,
    data_digest: str | None = None,
    optimizer_state: Any | None = None,
    scheduler_state: Any | None = None,
    rng_state: Any | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    run_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> CheckpointMetadata:
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "rng": rng_state,
    }
    return save_checkpoint(
        path,
        state,
        kind="training",
        config=config,
        config_digest=config_digest,
        data=data,
        data_digest=data_digest,
        run_id=run_id,
        epoch=epoch,
        global_step=global_step,
        extra=extra,
    )


def load_training_checkpoint(path: str | Path, **expectations: Any) -> LoadedCheckpoint:
    return load_checkpoint(path, expected_kind="training", **expectations)


def save_recursive_checkpoint(
    path: str | Path,
    *,
    recursive_state: Any,
    forecast_horizon: int,
    config: Any | None = None,
    config_digest: str | None = None,
    data: Any | None = None,
    data_digest: str | None = None,
    run_id: str | None = None,
    rng_state: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> CheckpointMetadata:
    if int(forecast_horizon) < 0:
        raise ValueError("forecast_horizon must be non-negative")
    state = {"recursive": recursive_state, "rng": rng_state}
    return save_checkpoint(
        path,
        state,
        kind="recursive",
        config=config,
        config_digest=config_digest,
        data=data,
        data_digest=data_digest,
        run_id=run_id,
        forecast_horizon=int(forecast_horizon),
        extra=extra,
    )


def load_recursive_checkpoint(path: str | Path, **expectations: Any) -> LoadedCheckpoint:
    return load_checkpoint(path, expected_kind="recursive", **expectations)


def checkpoint_is_compatible(
    path: str | Path,
    *,
    kind: str | None = None,
    config: Any | None = None,
    config_digest: str | None = None,
    data: Any | None = None,
    data_digest: str | None = None,
) -> bool:
    try:
        load_checkpoint(
            path,
            expected_kind=kind,
            expected_config=config,
            expected_config_digest=config_digest,
            expected_data=data,
            expected_data_digest=data_digest,
        )
        return True
    except (CheckpointError, FileNotFoundError):
        return False


def find_latest_checkpoint(
    directory: str | Path,
    *,
    pattern: str = "*.npz",
    kind: str | None = None,
) -> Path | None:
    """Return the newest readable compatible checkpoint in a directory."""

    candidates = sorted(
        Path(directory).glob(pattern), key=lambda path: path.stat().st_mtime_ns, reverse=True
    )
    for candidate in candidates:
        try:
            loaded = load_checkpoint(candidate, expected_kind=kind)
            return loaded.path
        except CheckpointError:
            continue
    return None


__all__ = [
    "ARTIFACT_GUARD_SCHEMA_VERSION",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointMetadata",
    "CheckpointMismatchError",
    "LoadedCheckpoint",
    "RUN_IDENTITY_SCHEMA_VERSION",
    "artifact_bundle_fingerprint",
    "canonical_json",
    "checkpoint_is_compatible",
    "config_hash",
    "data_hash",
    "ensure_run_identity",
    "find_latest_checkpoint",
    "hash_file",
    "hash_paths",
    "code_fingerprint",
    "load_checkpoint",
    "load_recursive_checkpoint",
    "load_training_checkpoint",
    "make_run_identity",
    "save_checkpoint",
    "save_recursive_checkpoint",
    "save_training_checkpoint",
    "stable_file_fingerprint",
    "validate_artifact_guard",
    "write_artifact_guard",
]
