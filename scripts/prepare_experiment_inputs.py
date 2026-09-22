"""Share the locally prepared Alibaba cohort between experiment folders."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "experiments"
    / "alibaba_ours_lh_recursive"
    / "inputs"
    / "alibaba_selected_200"
)
TARGETS = (
    ROOT / "experiments" / "alibaba_ours_ablation_recursive" / "inputs" / "alibaba_selected_200",
    ROOT / "experiments" / "alibaba_ours_minganxing_KPRλτ" / "inputs" / "alibaba_selected_200",
)


def main() -> None:
    required = ("dataset.npz", "selected_workloads.csv")
    missing = [name for name in required if not (SOURCE / name).is_file()]
    if missing:
        raise SystemExit(
            f"Missing prepared input in {SOURCE}: {', '.join(missing)}. "
            "Run the main experiment's prepare command first."
        )

    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        for name in required:
            shutil.copy2(SOURCE / name, target / name)
        if target.parent.parent.name == "alibaba_ours_minganxing_KPRλτ":
            (target / "source_schema.csv").write_text(
                "workload_id,node_id,time_seconds,cpu_observed,mem_observed,"
                "disk_i_observed,disk_o_observed\n",
                encoding="utf-8",
            )
        print(f"Prepared {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
