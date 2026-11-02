"""Chạy ma trận run v2 tuần tự, có resume và log lỗi (không dừng cả matrix).

Ví dụ::

    python scripts/run_matrix.py --block F_factorial --seeds 42,123,2024
    python scripts/run_matrix.py --configs f00,f07,f15 --seeds 42
    python scripts/run_matrix.py --configs x1,x2 --seeds 42,123,2024 --folds 0,1,2

Run đã có ``meta.json`` bị bỏ qua. Run lỗi được ghi vào
``outputs/results/v2/matrix_log.csv`` và matrix tiếp tục.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RUNS_DIR  # noqa: E402
from src.train_v2 import run_training_v2  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "v2"
LOG_PATH = ROOT / "outputs" / "results" / "v2" / "matrix_log.csv"


def discover(block: str | None, configs: str | None) -> list[Path]:
    """Chọn file config theo block hoặc danh sách id."""
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    if configs:
        wanted = {c.strip().lower() for c in configs.split(",") if c.strip()}
        files = [p for p in files if p.stem.lower() in wanted]
        missing = wanted - {p.stem.lower() for p in files}
        if missing:
            raise SystemExit(f"Không thấy config: {sorted(missing)}")
    elif block:
        selected = []
        for path in files:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            if cfg["experiment"].get("block") == block:
                selected.append(path)
        files = selected
    if not files:
        raise SystemExit("Không có config nào khớp bộ lọc")
    return files


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Chạy ma trận run v2")
    parser.add_argument("--block", default=None)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--folds", default="")
    parser.add_argument("--limit", type=int, default=0, help="chỉ chạy N job đầu (smoke)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    files = discover(args.block, args.configs)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    folds = [int(s) for s in args.folds.split(",") if s.strip()] or [None]
    jobs = [(path, seed, fold) for path in files for seed in seeds for fold in folds]
    if args.limit:
        jobs = jobs[: args.limit]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.is_file()
    total_started = time.perf_counter()
    done = skipped = failed = 0
    with LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["timestamp", "config", "seed", "fold", "status", "seconds", "error"])
            fh.flush()

        for number, (path, seed, fold) in enumerate(jobs, start=1):
            name = path.stem
            run_name = f"v2_{name}_s{seed}" + ("" if fold is None else f"_f{fold}")
            if (RUNS_DIR / run_name / "meta.json").is_file():
                skipped += 1
                print(f"[{number}/{len(jobs)}] {run_name} — SKIP (đã có)", flush=True)
                continue
            started = time.perf_counter()
            try:
                meta = run_training_v2(path, seed=seed, fold=fold)
                seconds = time.perf_counter() - started
                done += 1
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, seed, fold or "",
                                 "ok", round(seconds, 1), ""])
                fh.flush()
                print(
                    f"[{number}/{len(jobs)}] {run_name} — OK — f1={meta['val_macro_f1']:.4f} "
                    f"({seconds:.0f}s)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - matrix phải tiếp tục
                seconds = time.perf_counter() - started
                failed += 1
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, seed, fold or "",
                                 "fail", round(seconds, 1), f"{type(exc).__name__}: {exc}"])
                fh.flush()
                print(f"[{number}/{len(jobs)}] {run_name} — FAIL — {type(exc).__name__}: {exc}", flush=True)
                logging.error("Lỗi run %s:\n%s", run_name, traceback.format_exc())

    elapsed = time.perf_counter() - total_started
    print(f"Tổng: {done} ok, {skipped} skip, {failed} fail / {len(jobs)} job — {elapsed/60:.1f} phút")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
