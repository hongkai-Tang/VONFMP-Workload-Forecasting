from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workload_fmm.datasets.alibaba_v2018 import ALIBABA_V2018_FILES, ALIBABA_V2018_URL_BASE

SUBSETS = {
    "meta": ("container_meta",),
    "required": ("container_meta", "container_usage"),
    "container": ("container_meta", "container_usage"),
    "machine": ("machine_meta", "machine_usage"),
    "batch": ("batch_task", "batch_instance"),
    "all": tuple(ALIBABA_V2018_FILES),
}


def _remote_size(url: str, timeout: int = 30) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None


def _download(url: str, target: Path, timeout: int = 60) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    headers = {}
    existing = tmp.stat().st_size if tmp.exists() else 0
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing > 0 else "wb"
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open(mode) as f:
        while True:
            chunk = resp.read(1024 * 1024 * 4)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Alibaba Cluster Trace v2018 tar.gz files.")
    parser.add_argument("--subset", choices=sorted(SUBSETS), default="required")
    parser.add_argument("--output-dir", default="data/alibaba_v2018/raw")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    selected = SUBSETS[args.subset]
    manifest = {"base_url": ALIBABA_V2018_URL_BASE, "subset": args.subset, "files": []}
    for key in selected:
        filename = ALIBABA_V2018_FILES[key]
        url = f"{ALIBABA_V2018_URL_BASE}/{filename}"
        target = out_dir / filename
        size = None
        try:
            size = _remote_size(url, timeout=args.timeout)
        except Exception as exc:
            print(f"[warn] could not read remote size for {filename}: {exc}")
        manifest["files"].append({"name": key, "filename": filename, "url": url, "bytes": size})
        if args.dry_run:
            print(f"{filename}\t{size if size is not None else 'unknown'}\t{url}")
            continue
        if target.exists():
            print(f"[skip] {target} already exists")
            continue
        start = time.time()
        print(f"[download] {url} -> {target}")
        _download(url, target, timeout=args.timeout)
        print(f"[done] {target} in {time.time() - start:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
