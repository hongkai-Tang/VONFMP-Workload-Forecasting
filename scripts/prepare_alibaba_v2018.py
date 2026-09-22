from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workload_fmm.datasets.alibaba_v2018 import AlibabaPrepareConfig, prepare_container_usage


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Alibaba v2018 container trace for workload_fmm.")
    parser.add_argument("--raw-dir", default="data/alibaba_v2018/raw")
    parser.add_argument("--processed-dir", default="data/alibaba_v2018/processed")
    parser.add_argument("--max-workloads", type=int, default=2000)
    parser.add_argument("--min-points", type=int, default=None)
    parser.add_argument("--resample-seconds", type=int, default=60)
    parser.add_argument("--progress-every", type=int, default=1000000)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    parser.add_argument("--require-full-span", action="store_true")
    parser.add_argument("--full-span-days", type=float, default=8.0)
    parser.add_argument("--full-span-start-bucket", type=int, default=None)
    parser.add_argument("--full-span-start-seconds", type=int, default=86400)
    parser.add_argument("--allowed-missing-fraction", type=float, default=0.0)
    args = parser.parse_args()

    min_points = args.min_points
    if min_points is None:
        if args.require_full_span:
            min_points = int(math.ceil(args.full_span_days * 86400.0 / args.resample_seconds))
        else:
            min_points = 30

    manifest = prepare_container_usage(
        AlibabaPrepareConfig(
            raw_dir=Path(args.raw_dir),
            processed_dir=Path(args.processed_dir),
            max_workloads=args.max_workloads,
            min_points=min_points,
            resample_seconds=args.resample_seconds,
            progress_every=args.progress_every,
            stop_when_enough=not args.no_early_stop,
            candidate_multiplier=args.candidate_multiplier,
            require_full_span=args.require_full_span,
            full_span_days=args.full_span_days,
            full_span_start_bucket=args.full_span_start_bucket,
            full_span_start_seconds=args.full_span_start_seconds,
            allowed_missing_fraction=args.allowed_missing_fraction,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
