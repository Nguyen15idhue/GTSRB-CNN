"""Thống kê cho v2: macro-F1/43, cluster bootstrap theo track, paired bootstrap,
Holm, main effect/interaction factorial, phân rã phương sai.

Mọi hàm nhận ``tracks`` dạng mảng chuỗi cùng độ dài với ``y_true``; đơn vị
resample là track (frame trong cùng track tương quan mạnh).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

NUM_CLASSES = 43
LABELS = list(range(NUM_CLASSES))


def macro_f1_43(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> float:
    """Macro-F1 trên đủ 43 lớp (lớp vắng = 0)."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))


def _track_groups(tracks: Sequence[str]) -> list[np.ndarray]:
    """Danh sách mảng chỉ số ảnh cho từng track (thứ tự ổn định)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, track in enumerate(tracks):
        groups[track].append(i)
    return [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]


def cluster_bootstrap_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tracks: Sequence[str],
    B: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """CI 95% percentile cho macro-F1/43, resample theo track."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = _track_groups(tracks)
    rng = np.random.default_rng(seed)
    n = len(groups)
    point = macro_f1_43(y_true, y_pred)
    values = np.empty(B, dtype=np.float64)
    for b in range(B):
        chosen = rng.integers(0, n, size=n)
        idx = np.concatenate([groups[i] for i in chosen])
        values[b] = macro_f1_43(y_true[idx], y_pred[idx])
    return {
        "macro_f1": point,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "B": B,
        "n_tracks": n,
    }


def paired_bootstrap_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    tracks: Sequence[str],
    B: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Δ = F1(A) − F1(B) với cùng mẫu track cho cả hai model."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    groups = _track_groups(tracks)
    rng = np.random.default_rng(seed)
    n = len(groups)
    delta_point = macro_f1_43(y_true, pred_a) - macro_f1_43(y_true, pred_b)
    deltas = np.empty(B, dtype=np.float64)
    for b in range(B):
        chosen = rng.integers(0, n, size=n)
        idx = np.concatenate([groups[i] for i in chosen])
        deltas[b] = macro_f1_43(y_true[idx], pred_a[idx]) - macro_f1_43(y_true[idx], pred_b[idx])
    p_boot = 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return {
        "delta": float(delta_point),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_boot": min(p_boot, 1.0),
        "B": B,
    }


def bootstrap_contrast(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    tracks: Sequence[str],
    B: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """CI cho tổ hợp tuyến tính Σ w_c · F1_c (main effect / interaction)."""
    y_true = np.asarray(y_true)
    groups = _track_groups(tracks)
    rng = np.random.default_rng(seed)
    n = len(groups)
    configs = [c for c, w in weights.items() if abs(w) > 0]
    point = sum(weights[c] * macro_f1_43(y_true, predictions[c]) for c in configs)
    values = np.empty(B, dtype=np.float64)
    for b in range(B):
        chosen = rng.integers(0, n, size=n)
        idx = np.concatenate([groups[i] for i in chosen])
        values[b] = sum(weights[c] * macro_f1_43(y_true[idx], predictions[c][idx]) for c in configs)
    p = 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0)))
    return {
        "effect": float(point),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "p_boot": min(p, 1.0),
        "B": B,
    }


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm–Bonferroni: trả p hiệu chỉnh theo tên so sánh."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        value = min(1.0, (m - rank) * float(p))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def decision(delta: float, ci_low: float, ci_high: float, threshold: float = 0.003) -> str:
    """Áp luật 4.5: tốt hơn / kém hơn / tương đương / chưa kết luận."""
    if delta >= threshold and ci_low > 0:
        return "tot_hon"
    if delta <= -threshold and ci_high < 0:
        return "kem_hon"
    if ci_low > -threshold and ci_high < threshold:
        return "tuong_duong"
    return "chua_ket_luan"


def variance_decomposition(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Phân rã phương sai 2 chiều seed × fold (balanced) cho từng config.

    ``records``: mỗi phần tử có ``config_id``, ``seed``, ``fold``, ``value``.
    Công thức ANOVA: SS_seed, SS_fold, SS_resid; báo var components và ICC.
    """
    by_config: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_config[str(row["config_id"])].append(row)
    output: dict[str, Any] = {}
    pooled = []
    for config_id, rows in sorted(by_config.items()):
        seeds = sorted({int(r["seed"]) for r in rows})
        folds = sorted({int(r["fold"]) for r in rows})
        if len(seeds) < 2 or len(folds) < 2 or len(rows) < len(seeds) * len(folds):
            output[config_id] = {"valid": False, "reason": "thiếu seed/fold để phân rã"}
            continue
        table = {(int(r["seed"]), int(r["fold"])): float(r["value"]) for r in rows}
        values = np.array([[table[(s, f)] for f in folds] for s in seeds], dtype=np.float64)
        grand = values.mean()
        ss_seed = len(folds) * float(((values.mean(axis=1) - grand) ** 2).sum())
        ss_fold = len(seeds) * float(((values.mean(axis=0) - grand) ** 2).sum())
        ss_resid = float(((values - grand) ** 2).sum()) - ss_seed - ss_fold
        df_seed, df_fold = len(seeds) - 1, len(folds) - 1
        df_resid = df_seed * df_fold
        ms_seed = ss_seed / df_seed
        ms_fold = ss_fold / df_fold
        ms_resid = max(ss_resid / max(df_resid, 1), 1e-12)
        var_seed = max((ms_seed - ms_resid) / len(folds), 0.0)
        var_fold = max((ms_fold - ms_resid) / len(seeds), 0.0)
        var_resid = ms_resid
        total = var_seed + var_fold + var_resid
        output[config_id] = {
            "valid": True,
            "mean": float(grand),
            "var_seed": float(var_seed),
            "var_fold": float(var_fold),
            "var_resid": float(var_resid),
            "icc_seed": float(var_seed / total) if total > 0 else 0.0,
            "icc_fold": float(var_fold / total) if total > 0 else 0.0,
            "n_seeds": len(seeds),
            "n_folds": len(folds),
        }
        pooled.append(output[config_id])
    if pooled:
        output["_pooled"] = {
            "var_seed": float(np.mean([p["var_seed"] for p in pooled])),
            "var_fold": float(np.mean([p["var_fold"] for p in pooled])),
            "var_resid": float(np.mean([p["var_resid"] for p in pooled])),
            "icc_seed": float(np.mean([p["icc_seed"] for p in pooled])),
            "icc_fold": float(np.mean([p["icc_fold"] for p in pooled])),
        }
    return output


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đọc ``predictions_val.csv`` → (image_index, y_true, y_pred)."""
    import csv

    index, truth, pred = [], [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            index.append(int(row["image_index"]))
            truth.append(int(row["true_label"]))
            pred.append(int(row["pred_label"]))
    return np.asarray(index), np.asarray(truth), np.asarray(pred)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
