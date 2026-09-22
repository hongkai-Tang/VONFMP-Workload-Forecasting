from __future__ import annotations

import os
import sys
from pathlib import Path


# The Windows Anaconda NumPy and PyTorch wheels may carry different Intel
# OpenMP runtimes. Sequential MKL avoids the duplicate-runtime crash without
# enabling the unsafe KMP_DUPLICATE_LIB_OK workaround. GPU training is not
# affected by this NumPy setting.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alibaba_ours_duibi1.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
