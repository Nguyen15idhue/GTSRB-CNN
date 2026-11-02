"""Tổng hợp mọi run v2 thành ``outputs/results/v2/summary.csv`` (đúng schema 11.3).

Cách dùng::

    python -m src.summarize_v2                # có CI cluster bootstrap (B=2000)
    python -m src.summarize_v2 --no-ci        # nhanh, không tính CI
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import RESULTS_DIR, RUNS_DIR
from .data_v2 import track_map
from .stats_v2 import cluster_bootstrap_f1, read_predictions

LOG = logging.getLogger(__name__)

V2_RESULTS = RESULTS_DIR / "v2"
SUMMARY_PATH = V2_RESULTS / "summary.csv"
COLUMNS = (
    "protocol", "block", "config_id", "run_id", "seed", "fold", "split", "model", "params",
    "aug", "aug_profile", "epochs_run", "best_epoch", "val_acc", "val_macro_f1",
    "val_macro_f1_ci_low", "val_macro_f1_ci_high", "test_acc", "test_macro_f1",
    "train_time_s", "infer_ms_per_img", "git_commit", "config_sha256", "split_sha256",
    "valid_for_conclusion", "notes",
)


def collect_runs() -> list[dict[str, Any]]:
    """Quét ``outputs/runs/v2_*`` có meta.json, trả danh sách meta."""
    runs = []
    for folder in sorted(RUNS_DIR.glob("v2_*")):
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["_dir"] = str(folder)
        runs.append(meta)
    return runs


def summarize(compute_ci: bool = True, B: int = 2000) -> Path:
    """Sinh summary.csv từ artifact; CI lấy từ predictions_val.csv nếu có."""
    V2_RESULTS.mkdir(parents=True, exist_ok=True)
    mapping = track_map()
    tracks_all = np.array([m["track_key"] for m in mapping], dtype=object)
    rows = []
    runs = collect_runs()
    LOG.info("Tổng hợp %d run", len(runs))
    for meta in runs:
        folder = Path(meta["_dir"])
        row = {
            "protocol": meta.get("protocol", "v2_clean"),
            "block": meta.get("block", ""),
            "config_id": meta.get("config_id", ""),
            "run_id": meta["run_id"],
            "seed": meta["seed"],
            "fold": "" if meta.get("fold") is None else meta["fold"],
            "split": meta.get("split_name", ""),
            "model": meta.get("model", ""),
            "params": meta.get("params", ""),
            "aug": int(bool(meta.get("augmentation_profile") not in (None, "", "none"))),
            "aug_profile": meta.get("augmentation_profile", ""),
            "epochs_run": meta.get("epochs_run", ""),
            "best_epoch": meta.get("best_epoch", ""),
            "val_acc": meta.get("val_acc", ""),
            "val_macro_f1": meta.get("val_macro_f1", ""),
            "val_macro_f1_ci_low": "",
            "val_macro_f1_ci_high": "",
            "test_acc": "",
            "test_macro_f1": "",
            "train_time_s": meta.get("train_time_s", ""),
            "infer_ms_per_img": meta.get("infer_ms_per_img", ""),
            "git_commit": meta.get("git_commit", ""),
            "config_sha256": meta.get("config_sha256", ""),
            "split_sha256": meta.get("split_sha256", ""),
            "valid_for_conclusion": 1 if meta.get("epochs_run", 0) > 0 else 0,
            "notes": meta.get("notes", ""),
        }
        pred_path = folder / "predictions_val.csv"
        if compute_ci and pred_path.is_file():
            index, truth, pred = read_predictions(pred_path)
            if len(truth) > 0:
                ci = cluster_bootstrap_f1(truth, pred, tracks_all[index].tolist(), B=B)
                row["val_macro_f1_ci_low"] = round(ci["ci_low"], 6)
                row["val_macro_f1_ci_high"] = round(ci["ci_high"], 6)
                point = ci["macro_f1"]
                if row["val_macro_f1"] not in ("", None) and abs(point - float(row["val_macro_f1"])) > 1e-6:
                    LOG.warning(
                        "%s: macro-F1 trong meta %.6f khác tính lại %.6f",
                        meta["run_id"], float(row["val_macro_f1"]), point,
                    )
        rows.append(row)

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("Đã ghi %d dòng → %s", len(rows), SUMMARY_PATH)
    return SUMMARY_PATH


def load_summary() -> list[dict[str, str]]:
    """Đọc summary.csv (nếu có)."""
    if not SUMMARY_PATH.is_file():
        return []
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    parser = argparse.ArgumentParser(description="Tổng hợp summary.csv v2")
    parser.add_argument("--no-ci", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    summarize(compute_ci=not args.no_ci, B=args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
