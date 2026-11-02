"""Decision cards cho 10 yếu tố + hiệu chỉnh Holm cho 8 so sánh chính.

Nguồn số: summary.csv (mean ± sd qua seed), factorial effects (BN/DO/WD/Aug),
paired bootstrap theo track cho các cặp kiến trúc/hyperparam.

Artifact: ``outputs/results/v2/decision_cards.csv`` và ``main_comparisons.csv``.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis_v2 import (
    _aligned_predictions,
    _tracks_for,
    aggregate,
    factorial_effects,
    load_summary,
)
from .config import RESULTS_DIR
from .stats_v2 import decision, holm, paired_bootstrap_delta

LOG = logging.getLogger(__name__)

V2_RESULTS = RESULTS_DIR / "v2"
CARD_COLUMNS = (
    "factor", "levels", "reference", "metric", "mean", "ci_low", "ci_high", "delta",
    "delta_ci_low", "delta_ci_high", "p_holm", "mechanism", "evidence", "decision", "tradeoff",
)


def _pair_delta(config_id: str, reference: str, B: int = 1000) -> dict[str, Any]:
    """Δ + CI + p paired bootstrap qua 3 seed."""
    rows = [r for r in load_summary() if r["config_id"] == config_id]
    deltas, lows, highs, ps = [], [], [], []
    for row in rows:
        seed = int(row["seed"])
        try:
            indices, truth, preds = _aligned_predictions([config_id, reference], seed)
        except FileNotFoundError:
            continue
        result = paired_bootstrap_delta(truth, preds[config_id], preds[reference], _tracks_for(indices), B=B, seed=seed)
        deltas.append(result["delta"]); lows.append(result["ci_low"]); highs.append(result["ci_high"]); ps.append(result["p_boot"])
    if not deltas:
        return {"delta": None, "ci_low": None, "ci_high": None, "p_boot": None}
    return {
        "delta": float(np.mean(deltas)),
        "ci_low": float(np.mean(lows)),
        "ci_high": float(np.mean(highs)),
        "p_boot": float(np.mean(ps)),
    }


def _mean_sd(config_id: str) -> tuple[float, float, int]:
    rows = [r for r in load_summary() if r["config_id"] == config_id]
    values = np.array([float(r["val_macro_f1"]) for r in rows])
    return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else 0.0), len(values)


def build_cards(B: int = 1000) -> list[dict[str, Any]]:
    """Sinh 10 decision card với số thật từ artifact."""
    factorial = factorial_effects(B=B)
    cards: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}

    # 1–4: factorial.
    factor_info = {
        "bn": ("BN", "off vs on", "CNN-A (BN off)", "Ổn định phân tích activation; nếu off, gradient bất ổn và dễ overfit.",
               "Factorial 2^4 + learning curve F00 vs F07"),
        "do": ("Dropout", "0 vs 0,3", "dropout 0", "Giảm đồng thích nghi; quá cao gây underfit.",
               "Factorial 2^4 + train/val gap"),
        "wd": ("Weight decay", "0 vs 1e-4", "WD 0", "Co weight, giảm overfit; quá cao gây underfit.",
               "Factorial 2^4"),
        "aug": ("Augmentation", "off vs all", "không aug", "Tăng bất biến hình học/ánh sáng; có thể không tăng clean accuracy.",
                "Factorial 2^4 + corruption"),
    }
    for key, (name, levels, reference, mechanism, evidence) in factor_info.items():
        effect = factorial[key]
        mean_on = None  # main effect là contrast, không phải mean tuyệt đối
        p_values[name] = float(np.mean([r["p_boot"] for r in effect["per_seed"]]))
        cards.append({
            "factor": name, "levels": levels, "reference": reference, "metric": "macro_f1_43",
            "mean": mean_on, "ci_low": effect["ci_low"], "ci_high": effect["ci_high"],
            "delta": effect["effect_mean"], "delta_ci_low": effect["ci_low"], "delta_ci_high": effect["ci_high"],
            "p_holm": None, "mechanism": mechanism, "evidence": evidence,
            "decision": decision(effect["effect_mean"], effect["ci_low"], effect["ci_high"]),
            "tradeoff": "không thêm chi phí suy luận" if key != "aug" else "tăng thời gian train/epoch",
        })

    # 5–7: kiến trúc (so với D4 = F15).
    arch = [
        ("GAP", "d1", "f15", "Giảm params ~5×, mất chi tiết không gian.", "D1 vs D4 (=F15), cùng 3 khối, khác GAP"),
        ("Depth", "d2", "f15", "Receptive field tăng; quá sâu dễ overfit.", "D2 vs D4 (=F15), 4 khối vs 3 khối, cùng flatten"),
        ("Width", "w2", "f15", "Dung lượng tăng; lợi ích giảm dần.", "W2 vs D4 (=F15), 48/96/192 vs 32/64/128"),
    ]
    for name, config_id, reference, mechanism, evidence in arch:
        delta = _pair_delta(config_id, reference, B=B)
        mean, sd, n = _mean_sd(config_id)
        p_values[name] = delta["p_boot"] if delta["p_boot"] is not None else 1.0
        cards.append({
            "factor": name, "levels": f"{config_id} vs {reference}", "reference": reference,
            "metric": "macro_f1_43", "mean": mean, "ci_low": None, "ci_high": None,
            "delta": delta["delta"], "delta_ci_low": delta["ci_low"], "delta_ci_high": delta["ci_high"],
            "p_holm": None, "mechanism": mechanism, "evidence": evidence,
            "decision": decision(delta["delta"], delta["ci_low"], delta["ci_high"]) if delta["delta"] is not None else "chua_ket_luan",
            "tradeoff": "ít params hơn nếu GAP tương đương" if name == "GAP" else "chi phí train tăng theo params",
        })

    # 8: LR.
    lr_delta = _pair_delta("l1", "f15", B=B)
    mean, _, _ = _mean_sd("l1")
    p_values["LR"] = lr_delta["p_boot"] if lr_delta["p_boot"] is not None else 1.0
    cards.append({
        "factor": "LR", "levels": "1e-4 vs 1e-3", "reference": "f15 (1e-3)", "metric": "macro_f1_43",
        "mean": mean, "ci_low": None, "ci_high": None, "delta": lr_delta["delta"],
        "delta_ci_low": lr_delta["ci_low"], "delta_ci_high": lr_delta["ci_high"], "p_holm": None,
        "mechanism": "LR nhỏ ổn định hơn nhưng hội tụ chậm; LR lớn nhanh nhưng có thể bỏ qua cực tiểu.",
        "evidence": "L1 (40 epoch) vs F15 + learning curve",
        "decision": decision(lr_delta["delta"], lr_delta["ci_low"], lr_delta["ci_high"]) if lr_delta["delta"] is not None else "chua_ket_luan",
        "tradeoff": "thời gian train",
    })

    # 9–10: exploratory (chưa hiệu chỉnh).
    for name, config_id, levels, mechanism in (
        ("Class weight", "cw1", "none vs inv_sqrt", "Tăng recall lớp hiếm, có thể giảm precision lớp lớn."),
        ("Input size", "i2", "64 vs 96", "Chi tiết tăng nhưng chi phí và latency tăng."),
    ):
        delta = _pair_delta(config_id, "f15", B=B)
        mean, _, _ = _mean_sd(config_id)
        cards.append({
            "factor": name, "levels": levels, "reference": "f15", "metric": "macro_f1_43",
            "mean": mean, "ci_low": None, "ci_high": None, "delta": delta["delta"],
            "delta_ci_low": delta["ci_low"], "delta_ci_high": delta["ci_high"], "p_holm": None,
            "mechanism": mechanism, "evidence": f"{config_id} vs f15 (exploratory, chưa hiệu chỉnh)",
            "decision": decision(delta["delta"], delta["ci_low"], delta["ci_high"]) if delta["delta"] is not None else "chua_ket_luan",
            "tradeoff": "exploratory — không dùng để tuyên bố thắng/thua tuyệt đối",
        })

    # Holm cho đúng 8 so sánh chính.
    main = {name: p for name, p in p_values.items() if name in ("BN", "Dropout", "Weight decay", "Augmentation", "GAP", "Depth", "Width", "LR")}
    adjusted = holm(main)
    for card in cards:
        if card["factor"] in adjusted:
            card["p_holm"] = float(adjusted[card["factor"]])
    return cards


def write_cards(cards: Sequence[Mapping[str, Any]]) -> Path:
    V2_RESULTS.mkdir(parents=True, exist_ok=True)
    path = V2_RESULTS / "decision_cards.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CARD_COLUMNS)
        writer.writeheader()
        writer.writerows(cards)
    main = [c for c in cards if c["p_holm"] is not None]
    with (V2_RESULTS / "main_comparisons.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["factor", "delta", "ci_low", "ci_high", "p_holm", "decision"])
        for card in main:
            writer.writerow([card["factor"], card["delta"], card["delta_ci_low"], card["delta_ci_high"],
                             card["p_holm"], card["decision"]])
    LOG.info("Đã ghi %s (%d card)", path, len(cards))
    return path
