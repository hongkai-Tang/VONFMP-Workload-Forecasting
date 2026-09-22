from __future__ import annotations

"""Portable entry point for the Alibaba K/P/R/lambda/tau sensitivity suite."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
LOCAL_SRC = ROOT / "src"

for required in (VENDOR / "alibaba_ours_exp", VENDOR / "workload_fmm", LOCAL_SRC):
    if not required.is_dir():
        raise RuntimeError(f"portable experiment dependency is missing: {required}")

sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(LOCAL_SRC))

try:
    import torch
except Exception as exc:
    raise RuntimeError(
        "PyTorch cannot be imported. Activate the documented CUDA environment first."
    ) from exc

if not hasattr(torch, "Tensor") or not hasattr(torch, "nn"):
    raise RuntimeError(
        "The imported torch package is incomplete or shadowed "
        f"(torch.__file__={getattr(torch, '__file__', None)!r})."
    )

from alibaba_ours_minganxing.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
