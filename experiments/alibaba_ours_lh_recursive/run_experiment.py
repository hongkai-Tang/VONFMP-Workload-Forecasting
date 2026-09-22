from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENT_SRC = ROOT / "src"
CORE_SRC = ROOT.parent.parent / "src"

for path in (EXPERIMENT_SRC, CORE_SRC):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from alibaba_ours_exp.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
