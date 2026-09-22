from __future__ import annotations

"""Progress bars specialized for the sensitivity training lifecycle."""

import threading
from pathlib import Path
from typing import Any, Mapping

from alibaba_ours_exp.progress import ProgressTracker


class SensitivityProgressTracker(ProgressTracker):
    """Show epochs as train units and emit heartbeats during long setup work."""

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        training_epochs: int,
        heartbeat_interval_seconds: float = 30.0,
    ) -> None:
        super().__init__(directory, run_id=run_id)
        self.training_epochs = max(int(training_epochs), 1)
        self.heartbeat_interval_seconds = max(float(heartbeat_interval_seconds), 1.0)
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()

        def run() -> None:
            while not self._heartbeat_stop.wait(self.heartbeat_interval_seconds):
                try:
                    current = self.status
                    if current is None or current.get("status") != "running":
                        return
                    if current.get("stage") != "train":
                        continue
                    super(SensitivityProgressTracker, self).heartbeat()
                except Exception:
                    # Progress reporting must never interrupt model training.
                    return

        self._heartbeat_thread = threading.Thread(
            target=run,
            name=f"progress-heartbeat-{self.run_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._heartbeat_thread = None

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
        if stage == "train":
            total = self.training_epochs
            message = message or "initializing memberships and BackOff"
        status = super().start(
            total=total,
            stage=stage,
            message=message,
            metadata=metadata,
            resume=resume,
            reset=reset,
        )
        if stage == "train":
            self._start_heartbeat()
        return status

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
        details = dict(extra or {})
        current = self.status
        active_stage = stage or (None if current is None else current.get("stage"))
        if active_stage == "train":
            total = self.training_epochs
            epoch = details.get("epoch")
            if epoch is not None:
                completed = min(max(int(epoch), 0), self.training_epochs)
                increment = None
                history_length = details.get("history_length")
                prefix = "" if history_length is None else f"L={history_length}, "
                message = f"{prefix}epoch={completed}/{self.training_epochs}"
            elif completed is not None and int(completed) <= 1:
                # The inherited trainer reports one completed model after early
                # stopping. Map that terminal model unit to 100% of this bar.
                completed = self.training_epochs
        return super().update(
            completed=completed,
            increment=increment,
            total=total,
            stage=stage,
            message=message,
            metrics=metrics,
            forecast_horizon=forecast_horizon,
            extra=details,
        )

    def complete(
        self,
        *,
        summary: Mapping[str, Any] | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        self._stop_heartbeat()
        return super().complete(summary=summary, message=message)

    def fail(
        self,
        error: BaseException | str,
        *,
        context: Mapping[str, Any] | None = None,
        fatal: bool = True,
    ) -> dict[str, Any]:
        if fatal:
            self._stop_heartbeat()
        return super().fail(error, context=context, fatal=fatal)

    def cancel(self, *, message: str | None = None) -> dict[str, Any]:
        self._stop_heartbeat()
        return super().cancel(message=message)

    def skip(self, *, message: str | None = None) -> dict[str, Any]:
        self._stop_heartbeat()
        return super().skip(message=message)


__all__ = ["SensitivityProgressTracker"]
