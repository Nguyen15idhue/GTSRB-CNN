"""Kiểm tra số liệu 4.1: macro-F1/43 khớp sklearn, summary khớp artifact, hash đủ.

Cách dùng::

    python scripts/verify_41.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stats_v2 import macro_f1_43  # noqa: E402

OUT = ROOT / "outputs" / "results" / "v2"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # 1) OOF x2 seed 42 vs oof_summary.json.
    rows = [r for r in csv.DictReader((OUT / "oof_predictions.csv").open(encoding="utf-8"))
            if r["config_id"] == "x2" and r["seed"] == "42"]
    truth = [int(r["true_label"]) for r in rows]
    pred = [int(r["pred_label"]) for r in rows]
    recomputed = macro_f1_43(truth, pred)
    oof = json.loads((OUT / "oof_summary.json").read_text(encoding="utf-8"))
    stored = next(r["macro_f1"] for r in oof["configs"]["x2"]["per_seed"] if r["seed"] == 42)

    # 2) summary khớp meta + predictions cho một run screening.
    summary = list(csv.DictReader((OUT / "summary.csv").open(encoding="utf-8")))
    sample = next(r for r in summary if r["run_id"] == "v2_f15_s42")
    run_dir = ROOT / "outputs" / "runs" / "v2_f15_s42"
    pr = list(csv.DictReader((run_dir / "predictions_val.csv").open(encoding="utf-8")))
    val_f1 = macro_f1_43([int(r["true_label"]) for r in pr], [int(r["pred_label"]) for r in pr])

    # 3) hash đủ cho run kết luận.
    missing_hash = [r["run_id"] for r in summary
                    if r["valid_for_conclusion"] == "1" and (not r["config_sha256"] or not r["split_sha256"])]

    payload = {
        "oof_x2_seed42": {
            "recomputed": recomputed,
            "stored": stored,
            "abs_diff": abs(recomputed - stored),
            "passed": abs(recomputed - stored) < 1e-12,
        },
        "summary_f15_s42": {
            "recomputed": val_f1,
            "stored": float(sample["val_macro_f1"]),
            "abs_diff": abs(val_f1 - float(sample["val_macro_f1"])),
            "passed": abs(val_f1 - float(sample["val_macro_f1"])) < 1e-6,
        },
        "hash_missing_runs": missing_hash,
        "passed": abs(recomputed - stored) < 1e-12
                  and abs(val_f1 - float(sample["val_macro_f1"])) < 1e-6
                  and not missing_hash,
    }
    (OUT / "verify_41.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
