"""Dựng OOF: ghép 3 fold × 3 seed, xuất predictions, CI, paired Δ, variance.

Cách dùng::

    python scripts/run_oof_analysis.py --configs x1,x2,x3,x4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis_v2 import oof_predictions, oof_table  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.data_v2 import get_indices, track_map  # noqa: E402

OUT = RESULTS_DIR / "v2"


def export_predictions(config_ids: list[str]) -> Path:
    mapping = track_map()
    dev = sorted(get_indices("dev_all")["train"])
    path = OUT / "oof_predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config_id", "seed", "image_index", "track_key", "true_label",
                         "pred_label", "prob_true", "prob_pred", "fold"])
        for config_id in config_ids:
            merged = oof_predictions(config_id)
            for seed in (42, 123, 2024):
                for image_index in dev:
                    row = merged[seed][image_index]
                    writer.writerow([
                        config_id, seed, image_index, mapping[image_index]["track_key"],
                        row["true"], row["pred"], f"{row['prob_true']:.6f}",
                        f"{row['prob_pred']:.6f}", row["fold"],
                    ])
    return path


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Phân tích OOF v2")
    parser.add_argument("--configs", required=True)
    args = parser.parse_args()
    config_ids = [c.strip() for c in args.configs.split(",") if c.strip()]
    path = export_predictions(config_ids)
    print(f"Đã ghi {path}")
    result = oof_table(config_ids, B=2000)
    (OUT / "oof_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "oof_comparisons.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["comparison", "delta", "ci_low", "ci_high"])
        for name, row in result["comparisons"].items():
            writer.writerow([name, round(row["delta"], 6), round(row["ci_low"], 6), round(row["ci_high"], 6)])
    (OUT / "variance.json").write_text(
        json.dumps(result["variance"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for config_id, row in result["configs"].items():
        print(f"{config_id}: OOF macro-F1 = {row['macro_f1_mean']:.6f} ± {row['macro_f1_sd']:.6f} "
              f"[{row['ci_low_mean']:.6f}, {row['ci_high_mean']:.6f}]")
    for name, row in result["comparisons"].items():
        print(f"{name}: Δ = {row['delta']*100:+.3f} điểm [{row['ci_low']*100:+.3f}, {row['ci_high']*100:+.3f}]")
    pooled = result["variance"].get("_pooled", {})
    print(f"Variance pooled: seed={pooled.get('var_seed', 0):.2e} fold={pooled.get('var_fold', 0):.2e} "
          f"resid={pooled.get('var_resid', 0):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
