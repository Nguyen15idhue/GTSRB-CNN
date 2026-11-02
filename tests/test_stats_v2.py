"""Test thống kê v2: macro-F1, Holm, luật quyết định, bootstrap, phân rã phương sai."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.stats_v2 import (
    cluster_bootstrap_f1,
    decision,
    holm,
    macro_f1_43,
    variance_decomposition,
)


def test_macro_f1_matches_sklearn_with_missing_classes():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    y_pred = rng.integers(0, 5, size=200)
    expected = f1_score(y_true, y_pred, average="macro", labels=list(range(43)), zero_division=0)
    assert abs(macro_f1_43(y_true, y_pred) - expected) < 1e-12


def test_holm_is_monotone_and_bounded():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert all(0 <= v <= 1 for v in adjusted.values())
    assert adjusted["a"] <= adjusted["c"] <= adjusted["b"]
    assert adjusted["a"] == pytest.approx(0.03)


def test_decision_rule():
    assert decision(0.01, 0.002, 0.02) == "tot_hon"
    assert decision(-0.01, -0.02, -0.002) == "kem_hon"
    assert decision(0.001, -0.002, 0.002) == "tuong_duong"
    assert decision(0.004, -0.001, 0.009) == "chua_ket_luan"


def test_cluster_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 43, size=500)
    y_pred = np.where(rng.random(500) < 0.8, y_true, rng.integers(0, 43, size=500))
    tracks = [f"t{i // 30}" for i in range(500)]
    result = cluster_bootstrap_f1(y_true, y_pred, tracks, B=200, seed=0)
    assert result["ci_low"] <= result["macro_f1"] <= result["ci_high"]
    assert result["n_tracks"] == len(set(tracks))


def test_variance_decomposition_recovers_seed_component():
    records = []
    seed_effect = {42: 0.0, 123: 0.02, 2024: -0.02}
    for config in ("a", "b"):
        for seed, offset in seed_effect.items():
            for fold in range(3):
                records.append({
                    "config_id": config, "seed": seed, "fold": fold,
                    "value": 0.9 + offset + 0.001 * fold,
                })
    result = variance_decomposition(records)
    assert result["a"]["valid"]
    assert result["a"]["var_seed"] > 0
    assert 0 <= result["a"]["icc_seed"] <= 1
    assert abs(result["_pooled"]["var_seed"] - result["a"]["var_seed"]) < 1e-9
