from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path


def extract_member(tar_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        member = next((m for m in tar if m.isfile() and m.name.endswith(".csv")), None)
        if member is None:
            raise FileNotFoundError(f"no csv member found in {tar_path}")
        target = output_dir / Path(member.name).name
        if target.exists() and not overwrite:
            print(f"[skip] {target} already exists")
            return target
        tmp = target.with_suffix(target.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        print(f"[extract] {tar_path}::{member.name} -> {target}")
        src = tar.extractfile(member)
        if src is None:
            raise FileNotFoundError(f"could not extract {member.name}")
        written = 0
        with src, tmp.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024 * 16)
                if not chunk:
                    break
                dst.write(chunk)
                written += len(chunk)
                if written % (1024 * 1024 * 1024) < len(chunk):
                    print(f"[extract] written_gb={written / (1024 ** 3):.2f}", flush=True)
        tmp.replace(target)
        print(f"[done] {target} bytes={target.stat().st_size}")
        return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Alibaba v2018 CSV files from tar.gz archives.")
    parser.add_argument("--raw-dir", default="data/alibaba_v2018/raw")
    parser.add_argument("--name", default="container_usage")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    tar_path = raw_dir / f"{args.name}.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"missing {tar_path}")
    extract_member(tar_path, raw_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise
