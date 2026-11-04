"""Phân tích bảng thí nghiệm v2: factorial 2⁴, kiến trúc, hyperparam, augmentation,
OOF 3 fold × 3 seed, corruption benchmark.

Mọi số đọc từ ``outputs/runs/v2_*`` và ``outputs/results/v2/summary.csv``.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .config import NUM_CLASSES, RESULTS_DIR, RUNS_DIR
from .data_v2 import get_indices, load_images, track_map
from .stats_v2 import (
    bootstrap_contrast,
    cluster_bootstrap_f1,
    macro_f1_43,
    paired_bootstrap_delta,
    read_predictions,
    variance_decomposition,
)

LOG = logging.getLogger(__name__)

V2_RESULTS = RESULTS_DIR / "v2"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "v2"
SEEDS = (42, 123, 2024)


def load_summary() -> list[dict[str, str]]:
    path = V2_RESULTS / "summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Chạy: python -m src.summarize_v2")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def config_meta(config_id: str) -> dict[str, Any]:
    """Đọc YAML cấu hình để lấy bn/dropout/wd/aug/gap/widths/lr/..."""
    path = CONFIG_DIR / f"{config_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_dir(config_id: str, seed: int, fold: int | None = None) -> Path:
    name = f"v2_{config_id}_s{seed}" + ("" if fold is None else f"_f{fold}")
    return RUNS_DIR / name


def _prediction_map(config_id: str, seed: int, fold: int | None = None) -> dict[int, tuple[int, int]]:
    """``{image_index: (true, pred)}`` từ predictions_val.csv."""
    path = run_dir(config_id, seed, fold) / "predictions_val.csv"
    index, truth, pred = read_predictions(path)
    return {int(i): (int(t), int(p)) for i, t, p in zip(index, truth, pred)}


def _aligned_predictions(config_ids: Sequence[str], seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Căn predictions của nhiều config theo image_index; trả (indices, y_true, preds)."""
    maps = {c: _prediction_map(c, seed) for c in config_ids}
    common = set(maps[config_ids[0]])
    for c in config_ids[1:]:
        common &= set(maps[c])
    indices = np.array(sorted(common), dtype=np.int64)
    truth = np.array([maps[config_ids[0]][int(i)][0] for i in indices], dtype=np.int64)
    preds = {c: np.array([maps[c][int(i)][1] for i in indices], dtype=np.int64) for c in config_ids}
    return indices, truth, preds


def _tracks_for(indices: np.ndarray) -> list[str]:
    mapping = track_map()
    return [mapping[int(i)]["track_key"] for i in indices]


# ---------------------------------------------------------------------------
# Factorial 2^4
# ---------------------------------------------------------------------------


def _levels(config_id: str) -> dict[str, int]:
    cfg = config_meta(config_id)
    return {
        "bn": int(bool(cfg["model"]["use_bn"])),
        "do": int(float(cfg["model"]["dropout"]) > 0),
        "wd": int(float(cfg["train"]["weight_decay"]) > 0),
        "aug": int(bool(cfg["augmentation"]["enabled"])),
    }


def _factorial_weights(factor: str, configs: Sequence[str], second: str | None = None) -> dict[str, float]:
    """Trọng số contrast ±1/k cho main effect hoặc interaction."""
    weights: dict[str, float] = {}
    for config_id in configs:
        level = _levels(config_id)
        sign = 1.0 if level[factor] == 1 else -1.0
        if second is not None:
            sign *= 1.0 if level[second] == 1 else -1.0
        weights[config_id] = sign
    total = sum(abs(w) for w in weights.values()) / 2
    return {c: w / total for c, w in weights.items()}


def factorial_effects(B: int = 1000) -> dict[str, Any]:
    """Main effect + interaction của BN/DO/WD/Aug, CI cluster bootstrap."""
    configs = [f"f{i:02d}" for i in range(16)]
    effects = {name: [] for name in ("bn", "do", "wd", "aug", "bn_x_aug", "do_x_wd")}
    per_seed_points: dict[str, list[float]] = defaultdict(list)
    draws: dict[str, list[np.ndarray]] = defaultdict(list)
    for seed in SEEDS:
        indices, truth, preds = _aligned_predictions(configs, seed)
        tracks = _tracks_for(indices)
        for name in effects:
            if name == "bn_x_aug":
                weights = _factorial_weights("bn", configs, second="aug")
            elif name == "do_x_wd":
                weights = _factorial_weights("do", configs, second="wd")
            else:
                weights = _factorial_weights(name, configs)
            result = bootstrap_contrast(truth, preds, weights, tracks, B=B, seed=seed)
            per_seed_points[name].append(result["effect"])
            draws[name].append(np.asarray([result["ci_low"], result["ci_high"]]))
            effects[name].append({"seed": seed, **result})

    summary: dict[str, Any] = {}
    for name, rows in effects.items():
        points = np.array([r["effect"] for r in rows])
        # CI tổng hợp: lấy trung bình 3 phân phối bootstrap (mỗi seed một phân phối).
        # bootstrap_contrast chỉ trả percentile; dùng khoảng rộng nhất làm CI bảo thủ
        # và báo thêm sd qua seed.
        lows = np.array([r["ci_low"] for r in rows])
        highs = np.array([r["ci_high"] for r in rows])
        summary[name] = {
            "effect_mean": float(points.mean()),
            "effect_sd_seed": float(points.std(ddof=1)) if len(points) > 1 else 0.0,
            "ci_low": float(lows.mean()),
            "ci_high": float(highs.mean()),
            "ci_low_min": float(lows.min()),
            "ci_high_max": float(highs.max()),
            "per_seed": rows,
        }
    return summary


def factorial_table() -> list[dict[str, Any]]:
    """Bảng 16 dòng: mean±sd, Δ vs F00 [CI], gap, best epoch, params, time."""
    rows = load_summary()
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["block"] == "F_factorial":
            by_config[row["config_id"]].append(row)
    table = []
    f00 = {r["seed"]: r for r in by_config.get("f00", [])}
    for config_id in [f"f{i:02d}" for i in range(16)]:
        runs = by_config.get(config_id, [])
        if not runs:
            continue
        f1 = np.array([float(r["val_macro_f1"]) for r in runs])
        acc = np.array([float(r["val_acc"]) for r in runs])
        gaps = []
        for run in runs:
            meta_path = RUNS_DIR / run["run_id"] / "meta.json"
            if meta_path.is_file():
                gaps.append(json.loads(meta_path.read_text(encoding="utf-8")).get("train_val_acc_gap", 0.0))
        # Δ vs F00: paired bootstrap theo từng seed rồi lấy trung bình.
        deltas, lows, highs = [], [], []
        for run in runs:
            seed = int(run["seed"])
            ref = f00.get(str(seed))
            if ref is None:
                continue
            indices, truth, preds = _aligned_predictions([config_id, "f00"], seed)
            tracks = _tracks_for(indices)
            result = paired_bootstrap_delta(truth, preds[config_id], preds["f00"], tracks, B=1000, seed=seed)
            deltas.append(result["delta"]); lows.append(result["ci_low"]); highs.append(result["ci_high"])
        level = _levels(config_id)
        table.append({
            "config_id": config_id,
            "bn": level["bn"], "dropout": level["do"], "wd": level["wd"], "aug": level["aug"],
            "n_seeds": len(runs),
            "val_acc_mean": float(acc.mean()), "val_acc_sd": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "macro_f1_mean": float(f1.mean()), "macro_f1_sd": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
            "delta_vs_f00": float(np.mean(deltas)) if deltas else 0.0,
            "delta_ci_low": float(np.mean(lows)) if lows else 0.0,
            "delta_ci_high": float(np.mean(highs)) if highs else 0.0,
            "train_val_gap_mean": float(np.mean(gaps)) if gaps else 0.0,
            "best_epoch_mean": float(np.mean([int(r["best_epoch"]) for r in runs])),
            "params": int(runs[0]["params"]),
            "train_time_s_mean": float(np.mean([float(r["train_time_s"]) for r in runs])),
        })
    return table


# ---------------------------------------------------------------------------
# Bảng kiến trúc / hyperparam / augmentation
# ---------------------------------------------------------------------------


def aggregate(config_ids: Sequence[str], reference: str | None = None, B: int = 1000) -> list[dict[str, Any]]:
    """Mean ± sd theo seed + Δ [CI] so với reference (cùng seed, paired bootstrap)."""
    rows = load_summary()
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_config[row["config_id"]].append(row)
    table = []
    for config_id in config_ids:
        runs = by_config.get(config_id, [])
        if not runs:
            continue
        f1 = np.array([float(r["val_macro_f1"]) for r in runs])
        acc = np.array([float(r["val_acc"]) for r in runs])
        entry: dict[str, Any] = {
            "config_id": config_id,
            "n_seeds": len(runs),
            "val_acc_mean": float(acc.mean()),
            "val_acc_sd": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "macro_f1_mean": float(f1.mean()),
            "macro_f1_sd": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
            "best_epoch_mean": float(np.mean([int(r["best_epoch"]) for r in runs])),
            "epochs_run_mean": float(np.mean([int(r["epochs_run"]) for r in runs])),
            "params": int(runs[0]["params"]),
            "train_time_s_mean": float(np.mean([float(r["train_time_s"]) for r in runs])),
            "infer_ms_per_img": float(np.mean([float(r["infer_ms_per_img"]) for r in runs])),
        }
        if reference:
            deltas, lows, highs = [], [], []
            for run in runs:
                seed = int(run["seed"])
                if not run_dir(reference, seed).joinpath("predictions_val.csv").is_file():
                    continue
                indices, truth, preds = _aligned_predictions([config_id, reference], seed)
                tracks = _tracks_for(indices)
                result = paired_bootstrap_delta(truth, preds[config_id], preds[reference], tracks, B=B, seed=seed)
                deltas.append(result["delta"]); lows.append(result["ci_low"]); highs.append(result["ci_high"])
            entry.update({
                "delta_vs_reference": float(np.mean(deltas)) if deltas else None,
                "delta_ci_low": float(np.mean(lows)) if lows else None,
                "delta_ci_high": float(np.mean(highs)) if highs else None,
            })
        table.append(entry)
    return table


# ---------------------------------------------------------------------------
# OOF 3 fold × 3 seed
# ---------------------------------------------------------------------------


def oof_predictions(config_id: str) -> dict[int, dict[str, Any]]:
    """Ghép 3 fold của một seed → mỗi ảnh dev được dự đoán đúng một lần."""
    per_seed: dict[int, dict[int, dict[str, Any]]] = {}
    for seed in SEEDS:
        merged: dict[int, dict[str, Any]] = {}
        for fold in range(3):
            path = run_dir(config_id, seed, fold) / "predictions_val.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            index, truth, pred = read_predictions(path)
            with path.open(newline="", encoding="utf-8") as fh:
                prob_rows = {int(row["image_index"]): row for row in csv.DictReader(fh)}
            for i, t, p in zip(index, truth, pred):
                if int(i) in merged:
                    raise RuntimeError(f"OOF trùng ảnh {i} ở {config_id} seed {seed}")
                merged[int(i)] = {
                    "true": int(t), "pred": int(p), "fold": fold,
                    "prob_true": float(prob_rows[int(i)]["prob_true"]),
                    "prob_pred": float(prob_rows[int(i)]["prob_pred"]),
                }
        expected = set(get_indices("dev_all")["train"])
        if set(merged) != expected:
            raise RuntimeError(
                f"OOF {config_id} seed {seed}: {len(merged)} ảnh, thiếu {len(expected - set(merged))}, "
                f"thừa {len(set(merged) - expected)}"
            )
        per_seed[seed] = merged
    return per_seed


def oof_table(config_ids: Sequence[str], B: int = 2000) -> dict[str, Any]:
    """Macro-F1 OOF theo seed, CI, paired Δ, variance decomposition."""
    mapping = track_map()
    dev = np.array(sorted(get_indices("dev_all")["train"]), dtype=np.int64)
    truth = np.array([mapping[int(i)]["label"] for i in dev], dtype=np.int64)
    tracks = [mapping[int(i)]["track_key"] for i in dev]
    per_config: dict[str, Any] = {}
    predictions: dict[str, dict[int, np.ndarray]] = {}
    for config_id in config_ids:
        merged = oof_predictions(config_id)
        predictions[config_id] = {}
        seed_rows = []
        for seed in SEEDS:
            pred = np.array([merged[seed][int(i)]["pred"] for i in dev], dtype=np.int64)
            predictions[config_id][seed] = pred
            ci = cluster_bootstrap_f1(truth, pred, tracks, B=B, seed=seed)
            seed_rows.append({"seed": seed, **ci})
        f1s = np.array([r["macro_f1"] for r in seed_rows])
        per_config[config_id] = {
            "per_seed": seed_rows,
            "macro_f1_mean": float(f1s.mean()),
            "macro_f1_sd": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            "ci_low_mean": float(np.mean([r["ci_low"] for r in seed_rows])),
            "ci_high_mean": float(np.mean([r["ci_high"] for r in seed_rows])),
        }
    # Paired Δ giữa các config (cùng seed, cùng track resample).
    comparisons = {}
    if len(config_ids) > 1:
        for i, config_a in enumerate(config_ids):
            for config_b in config_ids[i + 1:]:
                deltas, lows, highs = [], [], []
                for seed in SEEDS:
                    result = paired_bootstrap_delta(
                        truth, predictions[config_a][seed], predictions[config_b][seed], tracks, B=B, seed=seed
                    )
                    deltas.append(result["delta"]); lows.append(result["ci_low"]); highs.append(result["ci_high"])
                comparisons[f"{config_a}_vs_{config_b}"] = {
                    "delta": float(np.mean(deltas)),
                    "ci_low": float(np.mean(lows)),
                    "ci_high": float(np.mean(highs)),
                    "per_seed": [{"seed": s, "delta": d, "ci_low": l, "ci_high": h}
                                 for s, d, l, h in zip(SEEDS, deltas, lows, highs)],
                }
    # Variance decomposition: mỗi fold là một giá trị F1.
    records = []
    for config_id in config_ids:
        merged = oof_predictions(config_id)
        for seed in SEEDS:
            for fold in range(3):
                idx = [i for i in dev if merged[seed][int(i)]["fold"] == fold]
                pred = np.array([merged[seed][int(i)]["pred"] for i in idx])
                records.append({
                    "config_id": config_id, "seed": seed, "fold": fold,
                    "value": macro_f1_43(truth[np.isin(dev, idx)], pred),
                })
    variance = variance_decomposition(records)
    return {"configs": per_config, "comparisons": comparisons, "variance": variance}


# ---------------------------------------------------------------------------
# Corruption benchmark
# ---------------------------------------------------------------------------


def corrupt_images(images: np.ndarray, kind: str, severity: int) -> np.ndarray:
    """Sinh ảnh corruption (uint8) cho 6 loại × 3 mức."""
    from PIL import Image, ImageEnhance, ImageFilter
    from io import BytesIO

    x = images.astype(np.float32)
    if kind == "blur":
        sigma = {1: 0.5, 2: 1.0, 3: 2.0}[severity]
        out = np.empty_like(images)
        for i, image in enumerate(images):
            out[i] = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(sigma)), dtype=np.uint8)
        return out
    if kind == "noise":
        sigma = {1: 5, 2: 10, 3: 20}[severity]
        rng = np.random.default_rng(1000 + severity)
        return np.clip(x + rng.normal(0, sigma, x.shape), 0, 255).astype(np.uint8)
    if kind == "brightness":
        factor = {1: 0.8, 2: 1.2, 3: 1.4}[severity]
        return np.clip(x * factor, 0, 255).astype(np.uint8)
    if kind == "contrast":
        factor = {1: 0.8, 2: 0.5, 3: 0.3}[severity]
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        return np.clip((x - mean) * factor + mean, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        quality = {1: 50, 2: 30, 3: 15}[severity]
        out = np.empty_like(images)
        for i, image in enumerate(images):
            buffer = BytesIO()
            Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            out[i] = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
        return out
    if kind == "occlusion":
        ratio = {1: 0.10, 2: 0.20, 3: 0.30}[severity]
        out = images.copy()
        side = int(round(np.sqrt(ratio) * images.shape[1]))
        y0 = (images.shape[1] - side) // 2
        x0 = (images.shape[2] - side) // 2
        out[:, y0:y0 + side, x0:x0 + side, :] = 0
        return out
    raise ValueError(f"corruption không hợp lệ: {kind}")


def corruption_table(config_ids: Sequence[str], seed: int = 42, max_images: int = 0) -> list[dict[str, Any]]:
    """Đánh giá checkpoint screening trên 6 corruption × 3 mức; ghi corruption.csv."""
    import torch

    from .config import get_device
    from .data_v2 import get_mean_std, make_loaders_v2
    from .models import build_model

    device = get_device()
    payload = load_images(64)
    indices = get_indices("screening")["val"]
    if max_images:
        indices = indices[:max_images]
    images = payload["x"][indices]
    labels = payload["y"][indices]
    mean, std = get_mean_std("screening")
    rows = []
    models = {}
    for config_id in config_ids:
        checkpoint_path = run_dir(config_id, seed) / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model, _ = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        models[config_id] = model.to(device).eval()

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    for kind in ("blur", "noise", "brightness", "contrast", "jpeg", "occlusion"):
        for severity in (1, 2, 3):
            corrupted = corrupt_images(images, kind, severity)
            tensor = torch.from_numpy(corrupted.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            tensor = (tensor - mean_t) / std_t
            for config_id in config_ids:
                preds = []
                with torch.no_grad():
                    for start in range(0, len(tensor), 256):
                        logits = models[config_id](tensor[start:start + 256].to(device))
                        preds.append(logits.argmax(1).cpu().numpy())
                pred = np.concatenate(preds)
                rows.append({
                    "config_id": config_id, "corruption": kind, "severity": severity,
                    "accuracy": float((pred == labels).mean()),
                    "macro_f1": macro_f1_43(labels, pred), "n": int(len(labels)),
                })
                LOG.info("%s | %s-%d | acc %.4f", config_id, kind, severity, rows[-1]["accuracy"])
    path = V2_RESULTS / "corruption.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["config_id", "corruption", "severity", "accuracy", "macro_f1", "n"])
        writer.writeheader(); writer.writerows(rows)
    return rows
