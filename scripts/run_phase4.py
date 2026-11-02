"""Chạy toàn bộ Phase 4 (bảng, hình, decision cards) từ artifact v2.

Cách dùng::

    python scripts/run_phase4.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis_v2 import aggregate, factorial_effects, factorial_table, load_summary  # noqa: E402
from src.decision_cards_v2 import build_cards, write_cards  # noqa: E402
from src.explain_v2 import (  # noqa: E402
    confusion_from_predictions,
    corruption_curves,
    error_taxonomy,
    gradcam_grid,
    interaction_plot,
    learning_curves,
    pareto_plot,
    per_class_heatmap,
    reliability_plot,
)
from src.summarize_v2 import summarize  # noqa: E402

OUT = ROOT / "outputs" / "results" / "v2"


def step(name, function, *args, **kwargs):
    try:
        result = function(*args, **kwargs)
        print(f"OK   {name}: {result}")
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    step("summary", summarize, compute_ci=True, B=2000)

    factorial = step("factorial_effects", factorial_effects, B=1000)
    table = step("factorial_table", factorial_table)
    if factorial:
        (OUT / "factorial_effects.json").write_text(json.dumps(factorial, ensure_ascii=False, indent=2), encoding="utf-8")
    if factorial:
        step("interaction_plot", interaction_plot, factorial)

    step("learning_curves", learning_curves, {"CNN-A (F00)": "v2_f00_s42", "CNN-B (F07)": "v2_f07_s42",
                                              "CNN-B+aug (F15)": "v2_f15_s42", "4 khối (D2)": "v2_d2_s42"})
    summary = load_summary()
    config_ids = sorted({r["config_id"] for r in summary if r["split"] == "screening"})
    pareto_rows = aggregate(config_ids)
    step("pareto", pareto_plot, pareto_rows)
    step("per_class_heatmap", per_class_heatmap, ["f00", "f07", "f15", "d2", "g2"], "f15", 42)
    step("reliability", reliability_plot, ["f00", "f07", "f15", "g2", "d2"], 42)
    step("corruption_curves", corruption_curves)
    step("confusion_f15", confusion_from_predictions, "f15", 42, "Confusion matrix chuẩn hóa hàng — F15")
    step("confusion_d2", confusion_from_predictions, "d2", 42, "Confusion matrix chuẩn hóa hàng — D2")
    step("gradcam", gradcam_grid, "f15", "d3", 42)
    step("error_taxonomy", error_taxonomy, "f15", 42)
    step("decision_cards", lambda: write_cards(build_cards(B=1000)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
