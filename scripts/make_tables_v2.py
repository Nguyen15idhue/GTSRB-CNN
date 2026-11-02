"""Sinh bảng Markdown cho báo cáo v2 từ artifact (không nhập tay số nào).

Cách dùng::

    python scripts/make_tables_v2.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis_v2 import aggregate, factorial_effects, factorial_table  # noqa: E402

RESULTS = ROOT / "outputs" / "results" / "v2"
OUT = RESULTS / "tables"


def _write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _percent(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "–"
    return f"{value * 100:.{digits}f}"


def factorial_tables() -> None:
    rows = factorial_table()
    effects = factorial_effects(B=500)
    lines = [
        "| ID | BN | DO | WD | Aug | Acc (%) | Macro-F1 (%) | Δ vs F00 (điểm %) [CI] | Gap | Best epoch | Params | Time (s) |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config_id']} | {r['bn']} | {r['dropout']} | {r['wd']} | {r['aug']} | "
            f"{_percent(r['val_acc_mean'])} ± {_percent(r['val_acc_sd'])} | "
            f"{_percent(r['macro_f1_mean'])} ± {_percent(r['macro_f1_sd'])} | "
            f"{r['delta_vs_f00']*100:+.3f} [{r['delta_ci_low']*100:+.3f}, {r['delta_ci_high']*100:+.3f}] | "
            f"{r['train_val_gap_mean']*100:.3f} | {r['best_epoch_mean']:.1f} | {r['params']} | {r['train_time_s_mean']:.1f} |"
        )
    _write("factorial.md", lines)

    lines = [
        "| Yếu tố | Mức bật | Mức tắt | Δ (điểm %) | 95% CI | p (bootstrap) | Kết luận |",
        "|---|---|---|---:|---|---:|---|",
    ]
    names = {"bn": "BN", "do": "Dropout", "wd": "Weight decay", "aug": "Augmentation",
             "bn_x_aug": "BN × Aug", "do_x_wd": "Dropout × WD"}
    for key, label in names.items():
        e = effects[key]
        lines.append(
            f"| {label} | – | – | {e['effect_mean']*100:+.3f} | "
            f"[{e['ci_low']*100:+.3f}, {e['ci_high']*100:+.3f}] | "
            f"{min(r['p_boot'] for r in e['per_seed']):.4f} | – |"
        )
    _write("main_effects.md", lines)
    (RESULTS / "factorial_effects.json").write_text(
        json.dumps(effects, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def block_table(name: str, config_ids: list[str], reference: str | None, out: str) -> None:
    rows = aggregate(config_ids, reference=reference, B=500)
    head = "| ID | Acc (%) | Macro-F1 (%) | Δ vs ref (điểm %) [CI] | Params | Latency (ms) | Best epoch |"
    lines = [head, "|---|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        delta = (
            f"{r['delta_vs_reference']*100:+.3f} [{r['delta_ci_low']*100:+.3f}, {r['delta_ci_high']*100:+.3f}]"
            if r.get("delta_vs_reference") is not None else "–"
        )
        lines.append(
            f"| {r['config_id']} | {_percent(r['val_acc_mean'])} ± {_percent(r['val_acc_sd'])} | "
            f"{_percent(r['macro_f1_mean'])} ± {_percent(r['macro_f1_sd'])} | {delta} | "
            f"{r['params']} | {r['infer_ms_per_img']:.3f} | {r['best_epoch_mean']:.1f} |"
        )
    _write(out, lines)


def corruption_table() -> None:
    path = RESULTS / "corruption.csv"
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    configs = sorted({r["config_id"] for r in rows})
    kinds = ["blur", "noise", "brightness", "contrast", "jpeg", "occlusion"]
    lines = ["| Config | " + " | ".join(kinds) + " | TB |", "|---|" + "---:|" * 7]
    for config_id in configs:
        values = []
        for kind in kinds:
            subset = [float(r["accuracy"]) for r in rows if r["config_id"] == config_id and r["corruption"] == kind]
            values.append(sum(subset) / len(subset) if subset else float("nan"))
        lines.append(f"| {config_id} | " + " | ".join(f"{v*100:.2f}" for v in values) + f" | {sum(values)/len(values)*100:.2f} |")
    _write("corruption.md", lines)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    factorial_tables()
    block_table("D", ["d1", "d2", "d3", "w1", "w2"], "f15", "architecture.md")
    block_table("H", ["l1", "cw1", "i2"], "f15", "hyperparam.md")
    block_table("G", ["g1", "g2", "g3"], "f07", "augmentation.md")
    corruption_table()
    print(f"Đã ghi bảng vào {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
