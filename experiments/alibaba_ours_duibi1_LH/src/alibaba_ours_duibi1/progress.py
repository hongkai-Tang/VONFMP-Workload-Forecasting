from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field

from .utils import format_duration


@dataclass
class ProgressBar:
    name: str
    total: int
    interval_seconds: float = 2.0
    width: int = 24
    started_at: float = field(default_factory=time.perf_counter)
    last_print_at: float = 0.0
    completed: int = 0
    suffix: str = ""
    _line_open: bool = False

    def update(self, completed: int, *, suffix: str = "", force: bool = False) -> None:
        self.completed = max(0, min(int(completed), max(self.total, 0)))
        self.suffix = suffix
        now = time.perf_counter()
        if not force and self.completed < self.total and now - self.last_print_at < self.interval_seconds:
            return
        self.last_print_at = now
        elapsed = max(now - self.started_at, 1e-9)
        rate = self.completed / elapsed if self.completed else 0.0
        remaining = self.total - self.completed
        eta = remaining / rate if rate > 0 else None
        ratio = self.completed / self.total if self.total else 1.0
        filled = min(self.width, int(round(self.width * ratio)))
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"[{self.name}] | [{bar}] | {ratio * 100:6.2f}% | "
            f"units={self.completed}/{self.total} | rate={rate:.3f}/s | "
            f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}"
        )
        if suffix:
            line += f" | {suffix}"
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        self._line_open = True
        if self.completed >= self.total:
            self.finish()

    def advance(self, amount: int = 1, *, suffix: str = "") -> None:
        self.update(self.completed + int(amount), suffix=suffix)

    def finish(self, *, suffix: str | None = None) -> None:
        if suffix is not None:
            self.suffix = suffix
        if self._line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._line_open = False


def print_stage(message: str) -> None:
    print(message, flush=True)


__all__ = ["ProgressBar", "print_stage"]
