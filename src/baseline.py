"""Baseline: majority class (E0a) và HOG + LinearSVC (E0b).

Cả hai chạy trên **cùng** `split_random` với CNN, chỉ đánh giá trên val
(không chạm test — test chỉ dùng ở Giai đoạn 6 sau khi đóng băng config).

Artifact:
    outputs/runs/e0a_majority/metrics.json
    outputs/runs/e0b_hog_svc/metrics.json + model.joblib

Cách dùng::

    python -m src.baseline --config configs/default.yaml
    python -m src.baseline --config configs/default.yaml --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import (
    NUM_CLASSES,
    RESULTS_DIR,
    RUNS_DIR,
    get_device,
    load_config,
    set_seed,
)
from .data import load_cache, load_split

LOG = logging.getLogger(__name__)

MAJORITY_DIR = RUNS_DIR / "e0a_majority"
HOG_DIR = RUNS_DIR / "e0b_hog_svc"
PHASE4_CHECK_PATH = RESULTS_DIR / "phase4_check.json"


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------


def load_split_arrays(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Nạp cache uint8 và cắt theo split đã lưu (không sinh lại split)."""
    cache = load_cache(int(cfg["data"]["image_size"]))
    split = load_split(cfg["data"]["split"])
    x, y = cache["train"]["x"], cache["train"]["y"]
    return {
        "x_train": x[split["train"]],
        "y_train": y[split["train"]],
        "x_val": x[split["val"]],
        "y_val": y[split["val"]],
        "split": cfg["data"]["split"],
        "split_file": f"split_{cfg['data']['split']}.json",
        "n_train": len(split["train"]),
        "n_val": len(split["val"]),
    }


def _hog_params(cfg: Mapping[str, Any]) -> dict[str, Any]:
    hog_cfg = cfg["baseline"]["hog"]
    return {
        "orientations": int(hog_cfg["orientations"]),
        "pixels_per_cell": tuple(int(v) for v in hog_cfg["pixels_per_cell"]),
        "cells_per_block": tuple(int(v) for v in hog_cfg["cells_per_block"]),
        "block_norm": str(hog_cfg["block_norm"]),
    }


def hog_features(images: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Trích HOG trên ảnh xám (ảnh RGB uint8 → gray [0,1])."""
    from skimage.color import rgb2gray
    from skimage.feature import hog

    params = _hog_params(cfg)
    features = np.empty((len(images), 0), dtype=np.float64)
    rows: list[np.ndarray] = []
    for image in images:
        gray = rgb2gray(image.astype(np.float32) / 255.0)
        rows.append(hog(gray, **params))
    if rows:
        features = np.vstack(rows)
    return features


# ---------------------------------------------------------------------------
# E0a — majority class
# ---------------------------------------------------------------------------


def run_majority(cfg: Mapping[str, Any], force: bool = False) -> dict[str, Any]:
    """Dự đoán mọi ảnh val là lớp nhiều nhất trong train."""
    from sklearn.metrics import accuracy_score, f1_score

    out_dir = MAJORITY_DIR
    if (out_dir / "metrics.json").is_file() and not force:
        raise FileExistsError(f"Đã có {out_dir / 'metrics.json'} (dùng --force để ghi đè)")
    data = load_split_arrays(cfg)
    counts = np.bincount(data["y_train"], minlength=NUM_CLASSES)
    majority = int(counts.argmax())
    predictions = np.full(len(data["y_val"]), majority, dtype=np.int64)
    started = time.perf_counter()
    accuracy = float(accuracy_score(data["y_val"], predictions))
    macro_f1 = float(f1_score(data["y_val"], predictions, average="macro", zero_division=0))
    infer_s = time.perf_counter() - started

    payload = {
        "run_id": out_dir.name,
        "model": "majority",
        "kind": "baseline",
        "params": 0,
        "epochs_run": 0,
        "best_epoch": 0,
        "seed": int(cfg["experiment"]["seed"]),
        "split": data["split"],
        "split_file": data["split_file"],
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "majority_class": majority,
        "majority_train_count": int(counts[majority]),
        "val_acc": round(accuracy, 6),
        "val_macro_f1": round(macro_f1, 6),
        "train_time_s": round(infer_s, 4),
        "fit_time_s": 0.0,
        "infer_ms_per_img": round(infer_s / max(len(data["y_val"]), 1) * 1000, 6),
        "notes": "Majority class: dự đoán mọi ảnh là lớp nhiều nhất trong train",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info(
        "Majority: lớp %d (%d ảnh train) | val acc %.4f | macro-F1 %.4f",
        majority,
        int(counts[majority]),
        accuracy,
        macro_f1,
    )
    return payload


# ---------------------------------------------------------------------------
# E0b — HOG + LinearSVC
# ---------------------------------------------------------------------------


def run_hog_svc(cfg: Mapping[str, Any], force: bool = False) -> dict[str, Any]:
    """HOG + StandardScaler + LinearSVC; chọn C theo val macro-F1."""
    import joblib
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    out_dir = HOG_DIR
    if (out_dir / "metrics.json").is_file() and not force:
        raise FileExistsError(f"Đã có {out_dir / 'metrics.json'} (dùng --force để ghi đè)")

    data = load_split_arrays(cfg)
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    started = time.perf_counter()
    x_train = hog_features(data["x_train"], cfg)
    x_val = hog_features(data["x_val"], cfg)
    feature_time = time.perf_counter() - started
    LOG.info("HOG xong: %s / %s chiều trong %.1f s", x_train.shape, x_val.shape, feature_time)

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    grid: list[dict[str, Any]] = []
    best: dict[str, Any] = {"C": None, "val_macro_f1": -1.0, "val_acc": 0.0, "fit_time_s": 0.0}
    for c_value in [float(v) for v in cfg["baseline"]["svc_c_grid"]]:
        fit_started = time.perf_counter()
        model = LinearSVC(
            C=c_value,
            max_iter=int(cfg["baseline"]["max_iter"]),
            dual=bool(cfg["baseline"].get("dual", True)),
            random_state=seed,
        ).fit(x_train_scaled, data["y_train"])
        fit_time = time.perf_counter() - fit_started
        predictions = model.predict(x_val_scaled)
        accuracy = float(accuracy_score(data["y_val"], predictions))
        macro_f1 = float(f1_score(data["y_val"], predictions, average="macro", zero_division=0))
        grid.append(
            {
                "C": c_value,
                "val_acc": round(accuracy, 6),
                "val_macro_f1": round(macro_f1, 6),
                "fit_time_s": round(fit_time, 2),
            }
        )
        LOG.info("C=%g | val acc %.4f | macro-F1 %.4f | fit %.1f s", c_value, accuracy, macro_f1, fit_time)
        if macro_f1 > best["val_macro_f1"]:
            best = {
                "C": c_value,
                "val_macro_f1": macro_f1,
                "val_acc": accuracy,
                "fit_time_s": fit_time,
                "model": model,
            }

    final_model = best.pop("model")
    infer_started = time.perf_counter()
    final_predictions = final_model.predict(x_val_scaled)
    infer_s = time.perf_counter() - infer_started

    params = int(final_model.coef_.size + final_model.intercept_.size)
    payload = {
        "run_id": out_dir.name,
        "model": "hog_linear_svc",
        "kind": "baseline",
        "params": params,
        "feature_dim": int(x_train.shape[1]),
        "epochs_run": 0,
        "best_epoch": 0,
        "seed": seed,
        "split": data["split"],
        "split_file": data["split_file"],
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "hog": {k: list(v) if isinstance(v, tuple) else v for k, v in _hog_params(cfg).items()},
        "c_grid": grid,
        "best_C": best["C"],
        "val_acc": round(best["val_acc"], 6),
        "val_macro_f1": round(best["val_macro_f1"], 6),
        "feature_time_s": round(feature_time, 2),
        "fit_time_s": round(best["fit_time_s"], 2),
        "train_time_s": round(feature_time + sum(g["fit_time_s"] for g in grid), 2),
        "infer_ms_per_img": round(infer_s / max(len(data["y_val"]), 1) * 1000, 4),
        "accuracy_from_final_model": round(float(accuracy_score(data["y_val"], final_predictions)), 6),
        "notes": (
            "HOG trên ảnh xám 64x64, StandardScaler fit trên train, LinearSVC OvR; "
            "chọn C theo val macro-F1; KHÔNG augmentation"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"scaler": scaler, "model": final_model, "cfg": payload}, out_dir / "model.joblib")
    LOG.info(
        "HOG+SVC: C=%g | val acc %.4f | macro-F1 %.4f | feature %.1f s | fit %.1f s | %.3f ms/ảnh",
        best["C"],
        best["val_acc"],
        best["val_macro_f1"],
        feature_time,
        best["fit_time_s"],
        payload["infer_ms_per_img"],
    )
    return payload


# ---------------------------------------------------------------------------
# Checklist Giai đoạn 4
# ---------------------------------------------------------------------------


def verify_phase4(cfg: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Kiểm tra checklist 4.5 trên artifact baseline; ghi ``phase4_check.json``."""
    cfg = cfg or load_config()
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        LOG.info("%s %s — %s", "PASS" if passed else "FAIL", name, detail)

    majority = json.loads((MAJORITY_DIR / "metrics.json").read_text(encoding="utf-8"))
    hog = json.loads((HOG_DIR / "metrics.json").read_text(encoding="utf-8"))
    data = load_split_arrays(cfg)
    counts = np.bincount(data["y_train"], minlength=NUM_CLASSES)
    majority_class = int(counts.argmax())

    add(
        "majority_dung_lop",
        majority["majority_class"] == majority_class,
        f"lưu {majority['majority_class']}, np.bincount → {majority_class} "
        f"(lớp {majority_class} có {int(counts[majority_class])} ảnh train)",
    )
    expected_acc = float(np.mean(data["y_val"] == majority_class))
    add(
        "majority_acc_khop",
        abs(majority["val_acc"] - expected_acc) < 1e-6,
        f"val acc {majority['val_acc']:.6f} ≈ tỷ lệ lớp nhiều nhất ở val {expected_acc:.6f}",
    )
    features = hog_features(data["x_val"][:3], cfg)
    add(
        "hog_shape",
        features.shape == (3, 1764),
        f"HOG 3 ảnh → {features.shape} (kỳ vọng (3, 1764), thực tế feature_dim={hog['feature_dim']})",
    )
    add("hog_no_nan", not bool(np.isnan(features).any()), f"NaN trong feature: {bool(np.isnan(features).any())}")

    grid = hog["c_grid"]
    add(
        "c_grid_3_dong",
        len(grid) == 3 and {round(float(g["C"]), 6) for g in grid} == {0.1, 1.0, 10.0},
        "bảng thử C: " + ", ".join(f"C={g['C']} f1={g['val_macro_f1']:.4f}" for g in grid),
    )
    best_from_grid = max(grid, key=lambda g: g["val_macro_f1"])
    add(
        "chon_c_theo_val_f1",
        float(best_from_grid["C"]) == float(hog["best_C"])
        and abs(best_from_grid["val_macro_f1"] - hog["val_macro_f1"]) < 1e-9,
        f"chọn C={hog['best_C']} (macro-F1 {hog['val_macro_f1']:.4f}) = max bảng thử",
    )
    add(
        "json_doc_lai",
        {"run_id", "val_acc", "val_macro_f1", "split", "seed"} <= set(majority) | set(hog),
        "metrics.json của cả hai baseline đọc lại được",
    )
    add(
        "co_thoi_gian",
        hog["feature_time_s"] > 0 and hog["fit_time_s"] > 0 and hog["infer_ms_per_img"] > 0,
        f"trích {hog['feature_time_s']} s | fit {hog['fit_time_s']} s | "
        f"{hog['infer_ms_per_img']:.3f} ms/ảnh",
    )
    add(
        "cung_split",
        majority["n_train"] == hog["n_train"] == data["n_train"] == 31360
        and majority["n_val"] == hog["n_val"] == data["n_val"] == 7841,
        f"train/val = {data['n_train']}/{data['n_val']}, cùng split_{data['split']}, seed "
        f"{majority['seed']}/{hog['seed']}",
    )

    summary = {
        "phase": 4,
        "generated_at": time.strftime("%Y-%m-%d"),
        "passed": sum(1 for c in checks if c["passed"]),
        "total": len(checks),
        "checks": checks,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE4_CHECK_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Kết quả: %d/%d PASS → %s", summary["passed"], summary["total"], PHASE4_CHECK_PATH)
    return checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline majority + HOG/LinearSVC (Giai đoạn 4)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="ghi đè metrics.json/model.joblib đã có")
    parser.add_argument("--verify", action="store_true", help="chạy checklist Giai đoạn 4")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config)
    cfg["experiment"]["seed"] = int(cfg["experiment"].get("seed", 42))

    if args.verify:
        checks = verify_phase4(cfg)
        failed = [c["check"] for c in checks if not c["passed"]]
        if failed:
            LOG.error("Checklist Giai đoạn 4 FAIL: %s", failed)
            return 1
        return 0

    run_majority(cfg, force=args.force)
    run_hog_svc(cfg, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
