"""Baseline B1 v2: HOG 64×64 + LinearSVC trên screening train/val.

Không dùng split_random, không chạm lockbox/official test. Chọn C trên screening
val theo macro-F1/43. Chạy thêm B0 majority (miễn phí) làm mốc dưới.

Artifact:
    outputs/runs/v2_b1/metrics.json
    outputs/results/v2/baseline.json

Cách dùng::

    python -m src.baseline_v2
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import NUM_CLASSES, RESULTS_DIR, RUNS_DIR, set_seed
from .data_v2 import get_indices, get_mean_std, load_images

LOG = logging.getLogger(__name__)

V2_RESULTS = RESULTS_DIR / "v2"
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def hog_features(images: np.ndarray) -> np.ndarray:
    """HOG trên ảnh xám [0,1]; trả ma trận (N, 1764)."""
    from skimage.color import rgb2gray
    from skimage.feature import hog

    rows: list[np.ndarray] = []
    for image in images:
        gray = rgb2gray(image.astype(np.float32) / 255.0)
        rows.append(hog(gray, **HOG_PARAMS))
    return np.vstack(rows)


def run_b1(seed: int = 42) -> dict[str, Any]:
    """Huấn luyện B1 và ghi metric/artifact."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    set_seed(seed)
    payload = load_images(64)
    x, y = payload["x"], payload["y"]
    indices = get_indices("screening")
    out_dir = RUNS_DIR / "v2_b1"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    x_train = hog_features(x[indices["train"]])
    x_val = hog_features(x[indices["val"]])
    feature_time = time.perf_counter() - started
    y_train, y_val = y[indices["train"]], y[indices["val"]]
    LOG.info("HOG xong %s/%s trong %.1f s", x_train.shape, x_val.shape, feature_time)

    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)

    grid: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    for c_value in (0.1, 1.0, 10.0):
        fit_started = time.perf_counter()
        model = LinearSVC(C=c_value, max_iter=5000, dual=True, random_state=seed).fit(x_train_s, y_train)
        fit_time = time.perf_counter() - fit_started
        pred = model.predict(x_val_s)
        entry = {
            "C": c_value,
            "val_acc": round(float(accuracy_score(y_val, pred)), 6),
            "val_macro_f1": round(float(f1_score(y_val, pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)), 6),
            "fit_time_s": round(fit_time, 2),
        }
        grid.append(entry)
        LOG.info("C=%g | val acc %.4f | macro-F1 %.4f | %.1f s", c_value, entry["val_acc"], entry["val_macro_f1"], fit_time)
        if not best or entry["val_macro_f1"] > best["val_macro_f1"]:
            best = {**entry, "model": model}

    model = best.pop("model")
    infer_started = time.perf_counter()
    pred = model.predict(x_val_s)
    infer_s = time.perf_counter() - infer_started
    params = int(model.coef_.size + model.intercept_.size)

    # B0 majority trên cùng val (mốc dưới).
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    majority = int(counts.argmax())
    pred_majority = np.full(len(y_val), majority, dtype=np.int64)
    b0 = {
        "majority_class": majority,
        "val_acc": round(float(accuracy_score(y_val, pred_majority)), 6),
        "val_macro_f1": round(float(f1_score(y_val, pred_majority, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)), 6),
    }

    result = {
        "protocol": "v2_clean",
        "config_id": "B1",
        "model": "hog_linear_svc",
        "seed": seed,
        "split": "screening",
        "n_train": int(len(indices["train"])),
        "n_val": int(len(indices["val"])),
        "hog": {k: list(v) if isinstance(v, tuple) else v for k, v in HOG_PARAMS.items()},
        "feature_dim": int(x_train.shape[1]),
        "params": params,
        "c_grid": grid,
        "best_C": best["C"],
        "val_acc": best["val_acc"],
        "val_macro_f1": best["val_macro_f1"],
        "feature_time_s": round(feature_time, 2),
        "fit_time_s": best["fit_time_s"],
        "infer_ms_per_img": round(infer_s / max(len(y_val), 1) * 1000, 4),
        "b0_majority": b0,
        "mean": get_mean_std("screening")[0],
        "std": get_mean_std("screening")[1],
        "notes": "HOG 64x64 + StandardScaler + LinearSVC; C chọn trên screening val; không refit trên dev; không chạm lockbox.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    V2_RESULTS.mkdir(parents=True, exist_ok=True)
    (V2_RESULTS / "baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    import csv

    with (out_dir / "predictions_val.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_index", "true_label", "pred_label"])
        for image_index, true_id, pred_id in zip(indices["val"], y_val.tolist(), pred.tolist()):
            writer.writerow([int(image_index), int(true_id), int(pred_id)])
    LOG.info("B1: C=%g | val acc %.4f | macro-F1 %.4f | params %d", best["C"], best["val_acc"], best["val_macro_f1"], params)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    run_b1(seed=42)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
