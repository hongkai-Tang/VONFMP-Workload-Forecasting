from __future__ import annotations

"""Entry point for the five Alibaba Ours ablation variants."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_SRC = ROOT.parent / "alibaba_ours_lh_recursive" / "src"
LOCAL_SRC = ROOT / "src"
CORE_SRC = ROOT.parent.parent / "src"

if not BASE_SRC.is_dir():
    raise RuntimeError(
        "Missing sibling experiment alibaba_ours_lh_recursive. "
        "Copy the complete code/our tree to every machine."
    )

sys.path.insert(0, str(BASE_SRC))
sys.path.insert(0, str(LOCAL_SRC))
sys.path.insert(0, str(CORE_SRC))

from alibaba_ours_ablation.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
