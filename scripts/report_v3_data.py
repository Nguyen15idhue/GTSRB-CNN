"""Sinh bảng/số liệu cho bản báo cáo làm lại (report pack v3).

Không chạy lại thực nghiệm, không mở lại lockbox. Mọi số liệu lấy từ artifact đã
có (``outputs/results/v2``, ``outputs/splits/v2``, ``outputs/runs``) hoặc đo mới
trong phạm vi suy luận (benchmark, Grad-CAM trên screening val).

Cách dùng::

    python scripts/report_v3_data.py --only tables
    python scripts/report_v3_data.py --only probs
    python scripts/report_v3_data.py --only quality
    python scripts/report_v3_data.py --only benchmark
    python scripts/report_v3_data.py --only gradcam
    python scripts/report_v3_data.py --only review
    python scripts/report_v3_data.py --only all
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "results" / "v3"
TABLES = OUT / "tables"
DATA = OUT / "data"
FIG = ROOT / "outputs" / "figures" / "v3"
V2 = ROOT / "outputs" / "results" / "v2"
SPLITS = ROOT / "outputs" / "splits" / "v2"
RUNS = ROOT / "outputs" / "runs"
CLASS_NAMES = ROOT / "configs" / "class_names.json"

NUM_CLASSES = 43
LOG = logging.getLogger("report_v3")


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------


def _mkdirs() -> None:
    for folder in (OUT, TABLES, DATA, FIG):
        folder.mkdir(parents=True, exist_ok=True)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(name: str, payload: Any) -> Path:
    path = TABLES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("ghi %s", path.relative_to(ROOT))
    return path


def _write_csv(name: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> Path:
    path = TABLES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"không có dòng để ghi {name}")
    fields = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in fields})
    LOG.info("ghi %s (%d dòng)", path.relative_to(ROOT), len(rows))
    return path


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    if isinstance(value, (np.floating,)):
        return f"{float(value):.6f}"
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def _percent(value: float, digits: int = 3) -> float:
    return float(value) * 100.0


def _names() -> dict[str, dict[str, str]]:
    return _json(CLASS_NAMES)


def _manifest() -> dict[str, Any]:
    return _json(SPLITS / "manifest.json")


def _load_npz(name: str) -> dict[str, np.ndarray]:
    return dict(np.load(DATA / name))


# ---------------------------------------------------------------------------
# A. Bảng từ artifact có sẵn
# ---------------------------------------------------------------------------


def table_class_catalog() -> None:
    names = _names()
    manifest = _manifest()
    y_train = np.load(ROOT / "data" / "cache" / "y_train.npy")
    y_test = np.load(ROOT / "data" / "cache" / "y_test.npy")
    pool = np.bincount(y_train, minlength=NUM_CLASSES)
    test = np.bincount(y_test, minlength=NUM_CLASSES)
    lock = np.asarray(manifest["lockbox"]["class_support"], dtype=np.int64)
    rows = []
    for cid in range(NUM_CLASSES):
        rows.append(
            {
                "class_id": cid,
                "name_vi": names[str(cid)]["vi"],
                "name_en": names[str(cid)]["en"],
                "shape": names[str(cid)]["shape"],
                "color": names[str(cid)]["color"],
                "n_pool": int(pool[cid]),
                "pct_pool": _percent(pool[cid] / pool.sum()),
                "n_test": int(test[cid]),
                "pct_test": _percent(test[cid] / test.sum()),
                "n_lockbox": int(lock[cid]),
                "pct_lockbox": _percent(lock[cid] / lock.sum()),
            }
        )
    _write_csv(
        "01_class_catalog.csv",
        rows,
        ["class_id", "name_vi", "name_en", "shape", "color", "n_pool", "pct_pool", "n_test", "pct_test", "n_lockbox", "pct_lockbox"],
    )
    summary = {
        "pool": int(pool.sum()),
        "test": int(test.sum()),
        "lockbox": int(lock.sum()),
        "n_classes": NUM_CLASSES,
        "min_pool": {"class_id": int(pool.argmin()), "n": int(pool.min())},
        "max_pool": {"class_id": int(pool.argmax()), "n": int(pool.max())},
        "median_pool": float(np.median(pool)),
        "imbalance_max_over_min": float(pool.max() / pool.min()),
        "classes_below_300": int((pool < 300).sum()),
        "classes_below_10pct_mean": int((pool < 0.10 * pool.mean()).sum()),
    }
    _write_json("class_catalog_summary.json", summary)


def table_split_summary() -> None:
    manifest = _manifest()
    rows = [
        {
            "tier": "Pool",
            "n_images": manifest["pool_size"],
            "n_tracks": manifest["checks"]["n_tracks_total"],
            "n_groups": manifest["n_sign_groups"],
            "seed": "",
            "rule": "39.209 ảnh train trừ 8 ảnh trùng MD5 với test",
            "role": "Nguồn để chia 3 tầng",
            "used_for_conclusion": "không",
        },
        {
            "tier": "Lockbox",
            "n_images": manifest["lockbox"]["n"],
            "n_tracks": manifest["lockbox"]["n_tracks"],
            "n_groups": manifest["lockbox"]["n_groups"],
            "seed": manifest["lockbox"]["seed"],
            "rule": "20% sign-group mỗi lớp",
            "role": "Test nội bộ độc lập, mở một lần sau freeze",
            "used_for_conclusion": "có",
        },
        {
            "tier": "Dev",
            "n_images": manifest["dev"]["n"],
            "n_tracks": manifest["dev"]["n_tracks"],
            "n_groups": manifest["dev"]["n_groups"],
            "seed": "",
            "rule": "Pool trừ lockbox",
            "role": "Screening + OOF + train final",
            "used_for_conclusion": "có (qua OOF)",
        },
        {
            "tier": "Screening train",
            "n_images": manifest["screening"]["train"],
            "n_tracks": manifest["screening"]["n_tracks_train"],
            "n_groups": "",
            "seed": manifest["screening"]["seed"],
            "rule": "80% sign-group của dev",
            "role": "Train cho factorial/kiến trúc/hyperparam/augmentation",
            "used_for_conclusion": "không (chỉ chọn)",
        },
        {
            "tier": "Screening val",
            "n_images": manifest["screening"]["val"],
            "n_tracks": manifest["screening"]["n_tracks_val"],
            "n_groups": "",
            "seed": manifest["screening"]["seed"],
            "rule": "20% sign-group của dev",
            "role": "Chọn cấu hình, early stopping, phân tích cơ chế",
            "used_for_conclusion": "không (chỉ chọn)",
        },
        {
            "tier": "OOF (3 fold)",
            "n_images": sum(f["val"] for f in manifest["folds"]),
            "n_tracks": sum(f["n_tracks_val"] for f in manifest["folds"]),
            "n_groups": "",
            "seed": "42/123/2024",
            "rule": "Chia sign-group của dev thành 3 fold cân bằng frame",
            "role": "Chọn winner bằng OOF 3 fold × 3 seed",
            "used_for_conclusion": "có",
        },
        {
            "tier": "Final train",
            "n_images": manifest["dev"]["n"],
            "n_tracks": manifest["dev"]["n_tracks"],
            "n_groups": manifest["dev"]["n_groups"],
            "seed": "42/123/2024",
            "rule": "Toàn bộ dev",
            "role": "Train 3 seed cuối, epoch cố định 26",
            "used_for_conclusion": "có",
        },
        {
            "tier": "Official test raw",
            "n_images": 12630,
            "n_tracks": "",
            "n_groups": "",
            "seed": "",
            "rule": "GTSRB Final_Test",
            "role": "Benchmark phụ, không track-disjoint",
            "used_for_conclusion": "không (chỉ tham khảo)",
        },
        {
            "tier": "Official test clean",
            "n_images": 11879,
            "n_tracks": "",
            "n_groups": "",
            "seed": "",
            "rule": "Loại 751 ảnh test có ảnh train gần trùng (MAD < 5)",
            "role": "Benchmark phụ sau khi lọc gần trùng",
            "used_for_conclusion": "không (chỉ tham khảo)",
        },
    ]
    _write_csv(
        "02_split_summary.csv",
        rows,
        ["tier", "n_images", "n_tracks", "n_groups", "seed", "rule", "role", "used_for_conclusion"],
    )


def _aggregate_block(config_ids: Sequence[str], reference: str, block: str, role: str) -> list[dict[str, Any]]:
    from src.analysis_v2 import aggregate

    rows = aggregate(list(config_ids), reference=reference, B=1000)
    out = []
    for row in rows:
        out.append(
            {
                "config_id": row["config_id"],
                "block": block,
                "role": role,
                "n_seeds": row["n_seeds"],
                "accuracy_pct": _percent(row["val_acc_mean"]),
                "accuracy_sd_pct": _percent(row["val_acc_sd"]),
                "macro_f1_pct": _percent(row["macro_f1_mean"]),
                "macro_f1_sd_pct": _percent(row["macro_f1_sd"]),
                "delta_vs_ref_pct": _percent(row["delta_vs_reference"]),
                "delta_ci_low_pct": _percent(row["delta_ci_low"]),
                "delta_ci_high_pct": _percent(row["delta_ci_high"]),
                "reference": reference,
                "params": row["params"],
                "train_time_s": row["train_time_s_mean"],
                "infer_ms_per_img": row["infer_ms_per_img"],
                "epochs_run": row["epochs_run_mean"],
                "best_epoch": row["best_epoch_mean"],
            }
        )
    return out


def _oof_block() -> list[dict[str, Any]]:
    import pandas as pd

    summary = pd.read_csv(V2 / "summary.csv")
    oof = _json(V2 / "oof_summary.json")
    comps = pd.read_csv(V2 / "oof_comparisons.csv").set_index("comparison")
    df = _read_oof()
    rows = []
    for config_id in ["x1", "x2", "x3"]:
        part = summary[summary["config_id"] == config_id]
        pred = df[df["config_id"] == config_id]
        acc_values = [
            float((chunk["true_label"] == chunk["pred_label"]).mean())
            for _seed, chunk in pred.groupby("seed")
        ]
        acc_mean, acc_sd = _mean_sd(acc_values)
        payload = oof["configs"][config_id]
        key = f"{config_id}_vs_x3"
        delta = comps.loc[key, "delta"] if key in comps.index else 0.0
        low = comps.loc[key, "ci_low"] if key in comps.index else 0.0
        high = comps.loc[key, "ci_high"] if key in comps.index else 0.0
        rows.append(
            {
                "config_id": config_id,
                "block": "X_confirmation",
                "role": "OOF / xác nhận",
                "n_seeds": 3,
                "accuracy_pct": _percent(acc_mean),
                "accuracy_sd_pct": _percent(acc_sd),
                "macro_f1_pct": _percent(payload["macro_f1_mean"]),
                "macro_f1_sd_pct": _percent(payload["macro_f1_sd"]),
                "delta_vs_ref_pct": _percent(float(delta)),
                "delta_ci_low_pct": _percent(float(low)),
                "delta_ci_high_pct": _percent(float(high)),
                "reference": "x3",
                "params": int(part["params"].iloc[0]),
                "train_time_s": float(part["train_time_s"].mean()),
                "infer_ms_per_img": float(part["infer_ms_per_img"].mean()),
                "epochs_run": float(part["epochs_run"].mean()),
                "best_epoch": float(part["best_epoch"].mean()),
            }
        )
    return rows


def table_model_comparison() -> None:
    baseline = _json(V2 / "baseline.json")
    rows: list[dict[str, Any]] = [
        {
            "config_id": "b0",
            "block": "Baseline",
            "role": "Mốc dưới (majority)",
            "n_seeds": 1,
            "accuracy_pct": _percent(baseline["b0_majority"]["val_acc"]),
            "accuracy_sd_pct": 0.0,
            "macro_f1_pct": _percent(baseline["b0_majority"]["val_macro_f1"]),
            "macro_f1_sd_pct": 0.0,
            "delta_vs_ref_pct": "",
            "delta_ci_low_pct": "",
            "delta_ci_high_pct": "",
            "reference": "",
            "params": 0,
            "train_time_s": "",
            "infer_ms_per_img": "",
            "epochs_run": "",
            "best_epoch": "",
        },
        {
            "config_id": "b1",
            "block": "Baseline",
            "role": "HOG + LinearSVC",
            "n_seeds": 1,
            "accuracy_pct": _percent(baseline["val_acc"]),
            "accuracy_sd_pct": 0.0,
            "macro_f1_pct": _percent(baseline["val_macro_f1"]),
            "macro_f1_sd_pct": 0.0,
            "delta_vs_ref_pct": "",
            "delta_ci_low_pct": "",
            "delta_ci_high_pct": "",
            "reference": "",
            "params": baseline["params"],
            "train_time_s": baseline["fit_time_s"] + baseline["feature_time_s"],
            "infer_ms_per_img": baseline["infer_ms_per_img"],
            "epochs_run": "",
            "best_epoch": "",
        },
    ]
    rows += _aggregate_block([f"f{i:02d}" for i in range(16)], "f00", "F_factorial", "Factorial 2^4")
    rows += _aggregate_block(["d1", "d2", "d3", "w1", "w2"], "d2", "D_architecture", "Kiến trúc")
    rows += _aggregate_block(["l1", "cw1", "i2"], "f15", "H_hyperparam", "Hyperparameter")
    rows += _aggregate_block(["g1", "g2", "g3"], "f07", "G_augmentation", "Augmentation thành phần")
    rows += _oof_block()
    fields = [
        "config_id", "block", "role", "n_seeds", "accuracy_pct", "accuracy_sd_pct",
        "macro_f1_pct", "macro_f1_sd_pct", "delta_vs_ref_pct", "delta_ci_low_pct",
        "delta_ci_high_pct", "reference", "params", "train_time_s", "infer_ms_per_img",
        "epochs_run", "best_epoch",
    ]
    _write_csv("05_model_comparison.csv", rows, fields)


def table_eval_sets() -> None:
    lock = _json(V2 / "lockbox_result.json")
    test = _json(V2 / "official_test_v2.json")
    oof = _json(V2 / "oof_summary.json")
    x2 = oof["configs"]["x2"]
    df = _read_oof()
    x2_pred = df[df["config_id"] == "x2"]
    acc_values = []
    for _seed, part in x2_pred.groupby("seed"):
        acc_values.append(float((part["true_label"] == part["pred_label"]).mean()))
    acc_mean, acc_sd = _mean_sd(acc_values)
    rows = [
        {
            "eval_set": "OOF (dev, X2)",
            "n": 30899,
            "accuracy_pct": _percent(acc_mean),
            "accuracy_sd_pct": _percent(acc_sd),
            "macro_f1_pct": _percent(x2["macro_f1_mean"]),
            "macro_f1_sd_pct": _percent(x2["macro_f1_sd"]),
            "ci_low_pct": _percent(x2["ci_low_mean"]),
            "ci_high_pct": _percent(x2["ci_high_mean"]),
            "top3_pct": "",
            "ece": "",
            "brier": "",
            "role": "Chọn winner; mỗi ảnh dev được dự đoán đúng 1 lần mỗi seed",
            "comparable_to_lockbox": "không (khác đơn vị nhóm/CI)",
        },
        {
            "eval_set": "Lockbox",
            "n": lock["n_images"],
            "accuracy_pct": _percent(lock["summary"]["accuracy"]["mean"]),
            "accuracy_sd_pct": _percent(lock["summary"]["accuracy"]["sd"]),
            "macro_f1_pct": _percent(lock["summary"]["macro_f1_43"]["mean"]),
            "macro_f1_sd_pct": _percent(lock["summary"]["macro_f1_43"]["sd"]),
            "ci_low_pct": "",
            "ci_high_pct": "",
            "top3_pct": _percent(lock["summary"]["top3_accuracy"]["mean"]),
            "ece": lock["summary"]["ece_15bin"]["mean"],
            "brier": lock["summary"]["brier"]["mean"],
            "role": "Test nội bộ độc lập, mở một lần sau freeze",
            "comparable_to_lockbox": "có (đây là chuẩn chính)",
        },
        {
            "eval_set": "Official test raw",
            "n": test["n_raw"],
            "accuracy_pct": _percent(test["raw"]["summary"]["accuracy"]["mean"]),
            "accuracy_sd_pct": _percent(test["raw"]["summary"]["accuracy"]["sd"]),
            "macro_f1_pct": _percent(test["raw"]["summary"]["macro_f1_43"]["mean"]),
            "macro_f1_sd_pct": _percent(test["raw"]["summary"]["macro_f1_43"]["sd"]),
            "ci_low_pct": "",
            "ci_high_pct": "",
            "top3_pct": _percent(test["raw"]["summary"]["top3_accuracy"]["mean"]),
            "ece": test["raw"]["summary"]["ece_15bin"]["mean"],
            "brier": test["raw"]["summary"]["brier"]["mean"],
            "role": "Benchmark phụ; không track-disjoint",
            "comparable_to_lockbox": "không (khác phân phối/nhóm)",
        },
        {
            "eval_set": "Official test clean",
            "n": test["n_clean"],
            "accuracy_pct": _percent(test["clean"]["summary"]["accuracy"]["mean"]),
            "accuracy_sd_pct": _percent(test["clean"]["summary"]["accuracy"]["sd"]),
            "macro_f1_pct": _percent(test["clean"]["summary"]["macro_f1_43"]["mean"]),
            "macro_f1_sd_pct": _percent(test["clean"]["summary"]["macro_f1_43"]["sd"]),
            "ci_low_pct": "",
            "ci_high_pct": "",
            "top3_pct": _percent(test["clean"]["summary"]["top3_accuracy"]["mean"]),
            "ece": test["clean"]["summary"]["ece_15bin"]["mean"],
            "brier": test["clean"]["summary"]["brier"]["mean"],
            "role": "Benchmark phụ sau lọc 751 ảnh gần trùng",
            "comparable_to_lockbox": "không (khác phân phối/nhóm)",
        },
    ]
    _write_csv(
        "06_eval_sets_summary.csv",
        rows,
        ["eval_set", "n", "accuracy_pct", "accuracy_sd_pct", "macro_f1_pct", "macro_f1_sd_pct", "ci_low_pct", "ci_high_pct", "top3_pct", "ece", "brier", "role", "comparable_to_lockbox"],
    )


def table_per_class_lockbox() -> None:
    lock = _json(V2 / "lockbox_result.json")
    names = _names()
    per_seed = [report["per_class"] for report in lock["reports"]]
    rows = []
    for cid in range(NUM_CLASSES):
        key = str(cid)
        p = [float(seed[key]["precision"]) for seed in per_seed]
        r = [float(seed[key]["recall"]) for seed in per_seed]
        f = [float(seed[key]["f1"]) for seed in per_seed]
        support = int(per_seed[0][key]["support"])
        rows.append(
            {
                "class_id": cid,
                "name_vi": names[key]["vi"],
                "support": support,
                "precision_mean": float(np.mean(p)),
                "precision_sd": float(np.std(p, ddof=1)),
                "recall_mean": float(np.mean(r)),
                "recall_sd": float(np.std(r, ddof=1)),
                "f1_mean": float(np.mean(f)),
                "f1_sd": float(np.std(f, ddof=1)),
                "f1_min": float(np.min(f)),
                "f1_max": float(np.max(f)),
            }
        )
    rows.sort(key=lambda row: row["f1_mean"])
    for rank, row in enumerate(rows, start=1):
        row["rank_worst_first"] = rank
    _write_csv(
        "07_per_class_lockbox.csv",
        rows,
        ["rank_worst_first", "class_id", "name_vi", "support", "precision_mean", "precision_sd", "recall_mean", "recall_sd", "f1_mean", "f1_sd", "f1_min", "f1_max"],
    )


def table_per_class_oof() -> None:
    from sklearn.metrics import precision_recall_fscore_support

    df = _read_oof()
    names = _names()
    x2 = df[df["config_id"] == "x2"]
    per_seed: dict[int, dict[str, Any]] = {}
    for seed, part in x2.groupby("seed"):
        precision, recall, f1, support = precision_recall_fscore_support(
            part["true_label"].to_numpy(),
            part["pred_label"].to_numpy(),
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
        per_seed[int(seed)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    rows = []
    for cid in range(NUM_CLASSES):
        rows.append(
            {
                "class_id": cid,
                "name_vi": names[str(cid)]["vi"],
                "support": int(per_seed[42]["support"][cid]),
                "precision_mean": float(np.mean([per_seed[s]["precision"][cid] for s in per_seed])),
                "precision_sd": float(np.std([per_seed[s]["precision"][cid] for s in per_seed], ddof=1)),
                "recall_mean": float(np.mean([per_seed[s]["recall"][cid] for s in per_seed])),
                "recall_sd": float(np.std([per_seed[s]["recall"][cid] for s in per_seed], ddof=1)),
                "f1_mean": float(np.mean([per_seed[s]["f1"][cid] for s in per_seed])),
                "f1_sd": float(np.std([per_seed[s]["f1"][cid] for s in per_seed], ddof=1)),
            }
        )
    rows.sort(key=lambda row: row["f1_mean"])
    for rank, row in enumerate(rows, start=1):
        row["rank_worst_first"] = rank
    _write_csv(
        "08_per_class_oof.csv",
        rows,
        ["rank_worst_first", "class_id", "name_vi", "support", "precision_mean", "precision_sd", "recall_mean", "recall_sd", "f1_mean", "f1_sd"],
    )


def _read_oof():
    import pandas as pd

    return pd.read_csv(V2 / "oof_predictions.csv")


def table_top_confusions() -> None:
    lock = _json(V2 / "lockbox_result.json")
    names = _names()
    manifest = _manifest()
    support = np.asarray(manifest["lockbox"]["class_support"], dtype=np.float64)
    aggregated: dict[tuple[int, int], list[int]] = defaultdict(list)
    for report in lock["reports"]:
        for item in report["top_confusions"]:
            aggregated[(int(item["true"]), int(item["pred"]))].append(int(item["count"]))
    n_errors = [int(round(report["n_samples"] * (1.0 - report["accuracy"]))) for report in lock["reports"]]
    rows = []
    for (true, pred), counts in sorted(aggregated.items(), key=lambda item: -np.mean(item[1])):
        mean_count = float(np.mean(counts))
        rows.append(
            {
                "true_id": true,
                "true_name": names[str(true)]["vi"],
                "pred_id": pred,
                "pred_name": names[str(pred)]["vi"],
                "count_mean_3seed": mean_count,
                "rate_of_true_class_pct": _percent(mean_count / support[true]),
                "share_of_errors_pct": _percent(mean_count / float(np.mean(n_errors))),
                "same_shape": names[str(true)]["shape"] == names[str(pred)]["shape"],
                "same_color": names[str(true)]["color"] == names[str(pred)]["color"],
            }
        )
    _write_csv(
        "09_top_confusions_lockbox.csv",
        rows[:40],
        ["true_id", "true_name", "pred_id", "pred_name", "count_mean_3seed", "rate_of_true_class_pct", "share_of_errors_pct", "same_shape", "same_color"],
    )
    # Ma trận đầy đủ: trung bình 3 seed (số đếm) + chuẩn hóa hàng.
    matrices = [np.asarray(report["confusion_matrix"], dtype=np.float64) for report in lock["reports"]]
    mean_matrix = np.mean(matrices, axis=0)
    row_sums = mean_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(mean_matrix, row_sums, out=np.zeros_like(mean_matrix), where=row_sums > 0)
    header = ["true\\pred"] + [str(i) for i in range(NUM_CLASSES)]
    with (TABLES / "10_confusion_lockbox_counts.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for cid in range(NUM_CLASSES):
            writer.writerow([cid] + [f"{value:.2f}" for value in mean_matrix[cid]])
    with (TABLES / "10_confusion_lockbox_normalized.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for cid in range(NUM_CLASSES):
            writer.writerow([cid] + [f"{value:.4f}" for value in normalized[cid]])
    np.save(DATA / "confusion_lockbox_mean.npy", mean_matrix)
    _write_json(
        "confusion_lockbox_summary.json",
        {
            "n_images": lock["n_images"],
            "n_errors_mean": float(np.mean(n_errors)),
            "accuracy_mean": lock["summary"]["accuracy"]["mean"],
            "top_offdiag": rows[:10],
        },
    )


def table_oof_by_fold_seed() -> None:
    import pandas as pd

    summary = pd.read_csv(V2 / "summary.csv")
    oof = summary[summary["config_id"].isin(["x1", "x2", "x3"])].copy()
    rows = []
    for _, row in oof.iterrows():
        rows.append(
            {
                "config_id": row["config_id"],
                "seed": int(row["seed"]),
                "fold": int(row["fold"]),
                "n_val": int(_fold_sizes()[int(row["fold"])]),
                "accuracy_pct": _percent(float(row["val_acc"])),
                "macro_f1_pct": _percent(float(row["val_macro_f1"])),
                "epochs_run": int(row["epochs_run"]),
                "best_epoch": int(row["best_epoch"]),
                "train_time_s": float(row["train_time_s"]),
            }
        )
    _write_csv(
        "11_oof_by_fold_seed.csv",
        rows,
        ["config_id", "seed", "fold", "n_val", "accuracy_pct", "macro_f1_pct", "epochs_run", "best_epoch", "train_time_s"],
    )


def _fold_sizes() -> dict[int, int]:
    manifest = _manifest()
    return {int(fold["fold"]): int(fold["val"]) for fold in manifest["folds"]}


def table_lockbox_by_seed() -> None:
    lock = _json(V2 / "lockbox_result.json")
    rows = []
    for seed, report in zip([42, 123, 2024], lock["reports"]):
        rows.append(
            {
                "seed": seed,
                "n": report["n_samples"],
                "accuracy_pct": _percent(report["accuracy"]),
                "macro_f1_pct": _percent(report["macro_f1_43"]),
                "top3_pct": _percent(report["top3_accuracy"]),
                "ece": report["ece_15bin"],
                "brier": report["brier"],
                "checkpoint_epoch": report["checkpoint_epoch"],
            }
        )
    _write_csv("21_lockbox_by_seed.csv", rows, ["seed", "n", "accuracy_pct", "macro_f1_pct", "top3_pct", "ece", "brier", "checkpoint_epoch"])


def table_test_per_class() -> None:
    test = _json(V2 / "official_test_v2.json")
    names = _names()
    rows = []
    for cid in range(NUM_CLASSES):
        key = str(cid)
        raw = [float(report["per_class"][key]["f1"]) for report in test["raw"]["reports"]]
        clean = [float(report["per_class"][key]["f1"]) for report in test["clean"]["reports"]]
        support_raw = int(test["raw"]["reports"][0]["per_class"][key]["support"])
        support_clean = int(test["clean"]["reports"][0]["per_class"][key]["support"])
        rows.append(
            {
                "class_id": cid,
                "name_vi": names[key]["vi"],
                "support_raw": support_raw,
                "support_clean": support_clean,
                "precision_raw": float(np.mean([float(r["per_class"][key]["precision"]) for r in test["raw"]["reports"]])),
                "recall_raw": float(np.mean([float(r["per_class"][key]["recall"]) for r in test["raw"]["reports"]])),
                "f1_raw": float(np.mean(raw)),
                "precision_clean": float(np.mean([float(r["per_class"][key]["precision"]) for r in test["clean"]["reports"]])),
                "recall_clean": float(np.mean([float(r["per_class"][key]["recall"]) for r in test["clean"]["reports"]])),
                "f1_clean": float(np.mean(clean)),
            }
        )
    _write_csv(
        "22_test_per_class_raw_clean.csv",
        rows,
        ["class_id", "name_vi", "support_raw", "support_clean", "precision_raw", "recall_raw", "f1_raw", "precision_clean", "recall_clean", "f1_clean"],
    )


def table_architecture_spec() -> None:
    import torch

    from src.config import load_yaml
    from src.models import SmallCNN

    rows = []
    for config_id in ["f00", "f07", "f15", "d1", "d2", "d3", "w1", "w2"]:
        cfg = load_yaml(ROOT / "configs" / "v2" / f"{config_id}.yaml")
        model_cfg = cfg["model"]
        model = SmallCNN(
            widths=model_cfg["widths"],
            use_bn=bool(model_cfg["use_bn"]),
            dropout=float(model_cfg["dropout"]),
            gap=bool(model_cfg["gap"]),
            fc_hidden=int(model_cfg.get("fc_hidden", 256)),
            num_classes=NUM_CLASSES,
            in_size=int(cfg["data"]["image_size"]),
        )
        # Thông tin kiến trúc tổng quát
        rows.append(
            {
                "config_id": config_id,
                "layer": "TỔNG",
                "detail": f"{len(model_cfg['widths'])} khối {('/'.join(map(str, model_cfg['widths'])))}",
                "output_shape": "(B,43)",
                "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "bn": int(bool(model_cfg["use_bn"])),
                "dropout": float(model_cfg["dropout"]),
                "gap": int(bool(model_cfg["gap"])),
                "fc_hidden": int(model_cfg.get("fc_hidden", 256)),
            }
        )
        for name, module in model.named_modules():
            if not name or not name.startswith(("features.", "classifier.")):
                continue
            if len(list(module.children())) > 0:
                continue
            params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            if params == 0 and not isinstance(module, torch.nn.ReLU):
                continue
            rows.append(
                {
                    "config_id": config_id,
                    "layer": name,
                    "detail": module.__class__.__name__ + (f" {tuple(module.weight.shape)}" if hasattr(module, "weight") and module.weight is not None else ""),
                    "output_shape": "",
                    "params": params,
                    "bn": int(bool(model_cfg["use_bn"])),
                    "dropout": float(model_cfg["dropout"]),
                    "gap": int(bool(model_cfg["gap"])),
                    "fc_hidden": int(model_cfg.get("fc_hidden", 256)),
                }
            )
    _write_csv(
        "23_architecture_spec.csv",
        rows,
        ["config_id", "layer", "detail", "output_shape", "params", "bn", "dropout", "gap", "fc_hidden"],
    )


def table_experiment_index() -> None:
    import pandas as pd

    df = pd.read_csv(V2 / "summary.csv")
    rows = []
    for _, row in df.sort_values(["block", "config_id", "seed", "fold"]).iterrows():
        rows.append(
            {
                "run_id": row["run_id"],
                "block": row["block"],
                "config_id": row["config_id"],
                "seed": int(row["seed"]),
                "fold": "" if pd.isna(row["fold"]) else int(row["fold"]),
                "split": row["split"],
                "params": int(row["params"]),
                "aug_profile": row["aug_profile"],
                "epochs_run": int(row["epochs_run"]),
                "best_epoch": int(row["best_epoch"]),
                "val_acc_pct": "" if pd.isna(row["val_acc"]) else _percent(float(row["val_acc"])),
                "val_macro_f1_pct": "" if pd.isna(row["val_macro_f1"]) else _percent(float(row["val_macro_f1"])),
                "train_time_s": float(row["train_time_s"]),
                "infer_ms_per_img": float(row["infer_ms_per_img"]),
                "git_commit": row["git_commit"],
                "notes": row["notes"],
            }
        )
    _write_csv(
        "20_experiment_index.csv",
        rows,
        ["run_id", "block", "config_id", "seed", "fold", "split", "params", "aug_profile", "epochs_run", "best_epoch", "val_acc_pct", "val_macro_f1_pct", "train_time_s", "infer_ms_per_img", "git_commit", "notes"],
    )


def table_training_cost() -> None:
    import pandas as pd

    df = pd.read_csv(V2 / "training_cost.csv")
    summary = _json(V2 / "training_cost_summary.json")
    rows = []
    for group, payload in summary.get("by_group", {}).items():
        rows.append(
            {
                "group": group,
                "n_runs": payload.get("n_runs"),
                "total_epochs": payload.get("total_epochs"),
                "total_time_min": payload.get("total_train_min"),
                "mean_time_s": payload.get("mean_train_s"),
                "median_time_s": payload.get("median_train_s"),
                "min_time_s": payload.get("min_train_s"),
                "max_time_s": payload.get("max_train_s"),
                "mean_epoch_s": payload.get("mean_epoch_s_weighted"),
                "max_gpu_mem_mb": payload.get("max_gpu_mem_mb"),
            }
        )
    _write_csv(
        "19_training_cost_by_group.csv",
        rows,
        ["group", "n_runs", "total_epochs", "total_time_min", "mean_time_s", "median_time_s", "min_time_s", "max_time_s", "mean_epoch_s", "max_gpu_mem_mb"],
    )
    # Tổng
    total = summary["total"]
    totals = {
        "n_runs": int(total["n_runs"]),
        "total_epochs": int(total["total_epochs"]),
        "total_time_s": float(total["total_train_s"]),
        "total_time_min": float(total["total_train_min"]),
        "total_time_h": float(total["total_train_s"]) / 3600.0,
        "baseline_time_s": 493.46,
        "grand_total_h": float(summary["grand_total_s"]) / 3600.0,
        "scope": "train_time_s = vòng epoch; không gồm nạp cache, dự đoán val cuối, phân tích/hình/báo cáo",
    }
    _write_json("training_cost_totals.json", totals)


def table_metric_worked_example() -> None:
    """Ví dụ tính Accuracy vs Macro-F1 từ lockbox seed 42."""
    lock = _json(V2 / "lockbox_result.json")
    report = lock["reports"][0]
    per_class = report["per_class"]
    support = np.array([int(per_class[str(c)]["support"]) for c in range(NUM_CLASSES)])
    f1 = np.array([float(per_class[str(c)]["f1"]) for c in range(NUM_CLASSES)])
    precision = np.array([float(per_class[str(c)]["precision"]) for c in range(NUM_CLASSES)])
    recall = np.array([float(per_class[str(c)]["recall"]) for c in range(NUM_CLASSES)])
    weighted = float((f1 * support).sum() / support.sum())
    rows = []
    for cid in range(NUM_CLASSES):
        rows.append(
            {
                "class_id": cid,
                "name_vi": _names()[str(cid)]["vi"],
                "support": int(support[cid]),
                "precision": float(precision[cid]),
                "recall": float(recall[cid]),
                "f1": float(f1[cid]),
                "contribution_to_macro_f1": float(f1[cid] / NUM_CLASSES),
            }
        )
    _write_csv(
        "18_metric_worked_example.csv",
        rows,
        ["class_id", "name_vi", "support", "precision", "recall", "f1", "contribution_to_macro_f1"],
    )
    _write_json(
        "metric_worked_example.json",
        {
            "seed": 42,
            "n": report["n_samples"],
            "accuracy": report["accuracy"],
            "macro_f1_43": report["macro_f1_43"],
            "weighted_f1": weighted,
            "explanation": (
                "Accuracy = số ảnh đúng / tổng ảnh. Macro-F1 = trung bình F1 của 43 lớp, mỗi lớp "
                "trọng số 1/43 bất kể support. Vì vậy lớp 19 (30 ảnh, F1 thấp) ảnh hưởng tới "
                "macro-F1 gần bằng lớp 4 (750 ảnh)."
            ),
            "worst_class": {
                "class_id": int(np.argmin(f1)),
                "f1": float(f1.min()),
                "support": int(support[np.argmin(f1)]),
            },
            "best_class": {
                "class_id": int(np.argmax(f1)),
                "f1": float(f1.max()),
                "support": int(support[np.argmax(f1)]),
            },
        },
    )


# ---------------------------------------------------------------------------
# B. Xác suất screening val + calibration + threshold
# ---------------------------------------------------------------------------


def _load_model(run_dir: Path, device: str = "cpu"):
    import torch

    from src.models import build_model

    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model, _ = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["config"]


def compute_screening_probs() -> None:
    import torch

    from src.data import _array_to_tensor
    from src.data_v2 import get_indices

    model, cfg = _load_model(RUNS / "v2_d2_s42", "cpu")
    mean = cfg["_resolved"]["mean"]
    std = cfg["_resolved"]["std"]
    x = np.load(ROOT / "data" / "cache" / "X_train.npy", mmap_mode="r")
    y = np.load(ROOT / "data" / "cache" / "y_train.npy")
    indices = np.asarray(get_indices("screening")["val"], dtype=np.int64)
    probs = np.zeros((len(indices), NUM_CLASSES), dtype=np.float32)
    batch = 256
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            chunk = indices[start : start + batch]
            tensors = torch.stack([_array_to_tensor(np.asarray(x[i]), mean, std) for i in chunk])
            logits = model(tensors)
            probs[start : start + len(chunk)] = torch.softmax(logits.float(), dim=1).numpy()
    labels = y[indices].astype(np.int64)
    np.savez_compressed(
        DATA / "screening_val_probs_d2_s42.npz",
        probs=probs,
        labels=labels,
        indices=indices,
    )
    LOG.info("probs screening val: %s", probs.shape)

    # Calibration + confidence
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == labels
    ece, brier, bins = _ece_brier(probs, labels, bins=15)
    rows = [
        {
            "eval_set": "screening_val (D2 s42)",
            "n": int(len(labels)),
            "accuracy_pct": _percent(float(correct.mean())),
            "mean_confidence": float(conf.mean()),
            "mean_confidence_correct": float(conf[correct].mean()),
            "mean_confidence_wrong": float(conf[~correct].mean()) if (~correct).any() else "",
            "n_wrong": int((~correct).sum()),
            "n_wrong_conf_ge_0.90": int(((~correct) & (conf >= 0.90)).sum()),
            "n_correct_conf_lt_0.50": int((correct & (conf < 0.50)).sum()),
            "ece_15bin": ece,
            "brier": brier,
        }
    ]
    _write_csv(
        "12_calibration_confidence.csv",
        rows,
        ["eval_set", "n", "accuracy_pct", "mean_confidence", "mean_confidence_correct", "mean_confidence_wrong", "n_wrong", "n_wrong_conf_ge_0.90", "n_correct_conf_lt_0.50", "ece_15bin", "brier"],
    )
    np.savez_compressed(
        DATA / "calibration_screening.npz",
        bin_confidence=bins["bin_confidence"],
        bin_accuracy=bins["bin_accuracy"],
        bin_count=bins["bin_count"],
        conf=conf,
        correct=correct,
    )

    # Threshold v2 từ OOF (held-out thật), không dùng screening val của final model.
    oof = _read_oof()
    x2_42 = oof[(oof["config_id"] == "x2") & (oof["seed"] == 42)]
    tau = float(np.percentile(x2_42["prob_pred"].to_numpy(), 5))
    accepted = x2_42["prob_pred"].to_numpy() >= tau
    y_true = x2_42["true_label"].to_numpy()
    y_pred = x2_42["pred_label"].to_numpy()
    payload = {
        "tau": tau,
        "percentile": 5.0,
        "source": "OOF x2 seed 42 (30.899 ảnh dev, mỗi ảnh được dự đoán ngoài fold)",
        "acceptance_oof": float(accepted.mean()),
        "accuracy_all": float((y_pred == y_true).mean()),
        "accuracy_accepted": float((y_pred[accepted] == y_true[accepted]).mean()),
        "coverage": float(accepted.mean()),
        "error_rejection_rate": float((~accepted[y_pred != y_true]).mean()),
        "n_oof": int(len(x2_42)),
        "note": "Dùng cho demo/report v3; KHÔNG dùng lockbox để chọn ngưỡng.",
    }
    _write_json("threshold_v2_oof.json", payload)

    # Đường selective risk
    order = np.argsort(-x2_42["prob_pred"].to_numpy())
    sorted_correct = (y_pred[order] == y_true[order]).astype(np.float64)
    coverage = np.arange(1, len(order) + 1) / len(order)
    risk = np.cumsum(sorted_correct) / np.arange(1, len(order) + 1)
    np.savez_compressed(DATA / "selective_risk_oof.npz", coverage=coverage, accuracy=risk, tau=tau)


def _ece_brier(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> tuple[float, float, dict[str, np.ndarray]]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    bin_conf, bin_acc, bin_count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_conf.append(float("nan"))
            bin_acc.append(float("nan"))
            bin_count.append(0)
            continue
        acc = float(correct[mask].mean())
        avg = float(conf[mask].mean())
        ece += count / len(labels) * abs(acc - avg)
        bin_conf.append(avg)
        bin_acc.append(acc)
        bin_count.append(count)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return float(ece), brier, {
        "bin_confidence": np.asarray(bin_conf),
        "bin_accuracy": np.asarray(bin_acc),
        "bin_count": np.asarray(bin_count),
    }


# ---------------------------------------------------------------------------
# C. Chất lượng dữ liệu
# ---------------------------------------------------------------------------


def _laplacian_var(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    padded = np.pad(g, ((0, 0), (1, 1), (1, 1)), mode="edge")
    lap = (
        4 * padded[:, 1:-1, 1:-1]
        - padded[:, :-2, 1:-1]
        - padded[:, 2:, 1:-1]
        - padded[:, 1:-1, :-2]
        - padded[:, 1:-1, 2:]
    )
    return lap.reshape(len(g), -1).var(axis=1)


def _image_stats(x: np.ndarray, batch: int = 2048) -> dict[str, np.ndarray]:
    n = len(x)
    blur = np.zeros(n, dtype=np.float32)
    brightness = np.zeros(n, dtype=np.float32)
    contrast = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch):
        chunk = np.asarray(x[start : start + batch], dtype=np.float32)
        gray = chunk.mean(axis=3) / 255.0
        blur[start : start + len(chunk)] = _laplacian_var(gray)
        brightness[start : start + len(chunk)] = gray.reshape(len(gray), -1).mean(axis=1)
        contrast[start : start + len(chunk)] = gray.reshape(len(gray), -1).std(axis=1)
    return {"blur": blur, "brightness": brightness, "contrast": contrast}


def compute_data_quality() -> None:
    import pandas as pd

    meta = pd.read_csv(ROOT / "outputs" / "results" / "train_meta.csv")
    meta = meta[meta["usable_for_train"] == 1].reset_index(drop=True)
    x_train = np.load(ROOT / "data" / "cache" / "X_train.npy", mmap_mode="r")
    x_test = np.load(ROOT / "data" / "cache" / "X_test.npy", mmap_mode="r")
    y_train = np.load(ROOT / "data" / "cache" / "y_train.npy")
    y_test = np.load(ROOT / "data" / "cache" / "y_test.npy")

    train_stats = _image_stats(x_train)
    test_stats = _image_stats(x_test)
    for key, values in train_stats.items():
        np.save(DATA / f"train_{key}.npy", values)
    for key, values in test_stats.items():
        np.save(DATA / f"test_{key}.npy", values)

    width = meta["width"].to_numpy(dtype=np.float64)
    height = meta["height"].to_numpy(dtype=np.float64)
    roi_w = (meta["roi_x2"] - meta["roi_x1"]).to_numpy(dtype=np.float64)
    roi_h = (meta["roi_y2"] - meta["roi_y1"]).to_numpy(dtype=np.float64)
    roi_ratio = np.clip(roi_w * roi_h / np.maximum(width * height, 1), 0, 1)

    def describe(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "p10": float(np.percentile(values, 10)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    p10_blur = float(np.percentile(train_stats["blur"], 10))
    p10_contrast = float(np.percentile(train_stats["contrast"], 10))
    summary = {
        "pool": {
            "n": int(len(y_train)),
            "blur_laplacian": describe(train_stats["blur"]),
            "brightness": describe(train_stats["brightness"]),
            "contrast": describe(train_stats["contrast"]),
            "width": describe(width),
            "height": describe(height),
            "roi_area_ratio": describe(roi_ratio),
            "n_dark_lt_0.10": int((train_stats["brightness"] < 0.10).sum()),
            "n_bright_gt_0.80": int((train_stats["brightness"] > 0.80).sum()),
            "n_low_contrast_p10": int((train_stats["contrast"] < p10_contrast).sum()),
            "n_blur_p10": int((train_stats["blur"] < p10_blur).sum()),
            "n_roi_ratio_lt_0.10": int((roi_ratio < 0.10).sum()),
            "thresholds": {"blur_p10": p10_blur, "contrast_p10": p10_contrast},
        },
        "test": {
            "n": int(len(y_test)),
            "blur_laplacian": describe(test_stats["blur"]),
            "brightness": describe(test_stats["brightness"]),
            "contrast": describe(test_stats["contrast"]),
        },
        "integrity": {
            "unreadable_images": 0,
            "image_modes": {"RGB": 51839},
            "md5_train_test_duplicates": 8,
            "md5_duplicates_within_pool": 0,
            "near_duplicate_pairs_merged_in_sign_group": 39603,
            "test_images_with_near_duplicate_train": 751,
            "label_errors_verified_manually": 0,
            "note": "Chưa có kiểm tra nhãn thủ công; xem 17_manual_error_review.csv để rà soát.",
        },
    }
    _write_json("04_data_quality_summary.json", summary)

    names = _names()
    rows = []
    for cid in range(NUM_CLASSES):
        mask = y_train == cid
        rows.append(
            {
                "class_id": cid,
                "name_vi": names[str(cid)]["vi"],
                "n_pool": int(mask.sum()),
                "blur_median": float(np.median(train_stats["blur"][mask])),
                "blur_p10": float(np.percentile(train_stats["blur"][mask], 10)),
                "brightness_median": float(np.median(train_stats["brightness"][mask])),
                "brightness_p10": float(np.percentile(train_stats["brightness"][mask], 10)),
                "contrast_median": float(np.median(train_stats["contrast"][mask])),
                "roi_area_ratio_median": float(np.median(roi_ratio[mask])),
                "width_median": float(np.median(width[mask])),
                "height_median": float(np.median(height[mask])),
                "n_dark_lt_0.10": int((train_stats["brightness"][mask] < 0.10).sum()),
                "n_blur_below_pool_p10": int((train_stats["blur"][mask] < p10_blur).sum()),
            }
        )
    _write_csv(
        "03_data_quality_per_class.csv",
        rows,
        ["class_id", "name_vi", "n_pool", "blur_median", "blur_p10", "brightness_median", "brightness_p10", "contrast_median", "roi_area_ratio_median", "width_median", "height_median", "n_dark_lt_0.10", "n_blur_below_pool_p10"],
    )


# ---------------------------------------------------------------------------
# D. Benchmark suy luận
# ---------------------------------------------------------------------------


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _peak_ram_mb() -> float:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wt.DWORD]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    handle = kernel32.GetCurrentProcess()
    psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 1024**2


def compute_inference_benchmark() -> None:
    import torch

    torch.set_num_threads(8)
    model, cfg = _load_model(RUNS / "v2_final_s42", "cpu")
    mean = cfg["_resolved"]["mean"]
    std = cfg["_resolved"]["std"]
    x = np.load(ROOT / "data" / "cache" / "X_test.npy", mmap_mode="r")
    sample = np.asarray(x[:256], dtype=np.uint8)

    rows = []
    checkpoint_bytes = (RUNS / "v2_final_s42" / "best.pt").stat().st_size
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def run_device(device_name: str) -> None:
        from src.data import _array_to_tensor

        device = torch.device(device_name)
        load_start = time.perf_counter()
        net, _ = _load_model(RUNS / "v2_final_s42", device_name)
        load_s = time.perf_counter() - load_start
        for batch in (1, 16, 32, 64):
            reps = 100 if batch <= 16 else 50
            warmup = 10
            inputs = torch.stack(
                [_array_to_tensor(sample[i % len(sample)], mean, std) for i in range(batch)]
            ).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(warmup):
                    _ = net(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            with torch.no_grad():
                for _ in range(reps):
                    if device.type == "cuda":
                        start_evt = torch.cuda.Event(enable_timing=True)
                        end_evt = torch.cuda.Event(enable_timing=True)
                        start_evt.record()
                        _ = net(inputs)
                        end_evt.record()
                        torch.cuda.synchronize()
                        times.append(start_evt.elapsed_time(end_evt))
                    else:
                        start = time.perf_counter()
                        _ = net(inputs)
                        times.append((time.perf_counter() - start) * 1000.0)
            times_arr = np.asarray(times)
            peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
            reserved_vram = torch.cuda.max_memory_reserved() / 1024**2 if device.type == "cuda" else 0.0
            rows.append(
                {
                    "device": device_name,
                    "batch_size": batch,
                    "reps": reps,
                    "latency_mean_ms": float(times_arr.mean()),
                    "latency_median_ms": float(np.median(times_arr)),
                    "latency_p95_ms": float(np.percentile(times_arr, 95)),
                    "latency_min_ms": float(times_arr.min()),
                    "throughput_img_s": float(batch / (np.median(times_arr) / 1000.0)),
                    "peak_vram_mb": peak_vram,
                    "reserved_vram_mb": reserved_vram,
                    "peak_ram_mb": _peak_ram_mb(),
                    "scope": "model-only: forward trên tensor đã chuẩn hóa, FP32, không gồm đọc/resize ảnh",
                }
            )
        # end-to-end 1 ảnh: uint8 -> preprocess -> forward -> top-k
        reps = 60
        times = []
        with torch.no_grad():
            for _ in range(reps):
                if device.type == "cuda":
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_evt.record()
                    tensor = _array_to_tensor(sample[0], mean, std).unsqueeze(0).to(device)
                    logits = net(tensor)
                    probs = torch.softmax(logits.float(), dim=1)
                    _ = torch.topk(probs, 3)
                    end_evt.record()
                    torch.cuda.synchronize()
                    times.append(start_evt.elapsed_time(end_evt))
                else:
                    start = time.perf_counter()
                    tensor = _array_to_tensor(sample[0], mean, std).unsqueeze(0).to(device)
                    logits = net(tensor)
                    probs = torch.softmax(logits.float(), dim=1)
                    _ = torch.topk(probs, 3)
                    times.append((time.perf_counter() - start) * 1000.0)
        times_arr = np.asarray(times)
        rows.append(
            {
                "device": device_name,
                "batch_size": 1,
                "reps": reps,
                "latency_mean_ms": float(times_arr.mean()),
                "latency_median_ms": float(np.median(times_arr)),
                "latency_p95_ms": float(np.percentile(times_arr, 95)),
                "latency_min_ms": float(times_arr.min()),
                "throughput_img_s": float(1 / (np.median(times_arr) / 1000.0)),
                "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0,
                "reserved_vram_mb": torch.cuda.max_memory_reserved() / 1024**2 if device.type == "cuda" else 0.0,
                "peak_ram_mb": _peak_ram_mb(),
                "scope": "end-to-end Python: uint8 -> normalize -> forward -> softmax -> top-3; chưa gồm upload/render UI",
            }
        )
        rows.append(
            {
                "device": device_name,
                "batch_size": 0,
                "reps": 1,
                "latency_mean_ms": load_s * 1000.0,
                "latency_median_ms": load_s * 1000.0,
                "latency_p95_ms": load_s * 1000.0,
                "latency_min_ms": load_s * 1000.0,
                "throughput_img_s": "",
                "peak_vram_mb": "",
                "reserved_vram_mb": "",
                "peak_ram_mb": _peak_ram_mb(),
                "scope": "model load: torch.load + build + load_state_dict + to(device)",
            }
        )
        if device.type == "cuda":
            del net
            torch.cuda.empty_cache()

    run_device("cpu")
    if torch.cuda.is_available():
        run_device("cuda")
    _write_csv(
        "13_inference_benchmark.csv",
        rows,
        ["device", "batch_size", "reps", "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_min_ms", "throughput_img_s", "peak_vram_mb", "reserved_vram_mb", "peak_ram_mb", "scope"],
    )
    _write_json(
        "13_inference_benchmark.json",
        {
            "model": "v2_final_s42 (D2, 26 epoch)",
            "params": int(params),
            "checkpoint_bytes": int(checkpoint_bytes),
            "checkpoint_mb": checkpoint_bytes / 1024**2,
            "fp32_theoretical_mb": params * 4 / 1024**2,
            "torch_threads": int(torch.get_num_threads()),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "rows": rows,
            "note": "Đo trên chính máy train (i9-14900KF, RTX 5070 12 GB). CPU dùng 8 thread. GPU dùng CUDA event. Không so với GPU khác.",
        },
    )


# ---------------------------------------------------------------------------
# E. Grad-CAM định lượng (mass trong ROI)
# ---------------------------------------------------------------------------


def compute_gradcam_roi() -> None:
    import pandas as pd
    import torch

    from src.data import _array_to_tensor
    from src.data_v2 import get_indices

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_model(RUNS / "v2_d2_s42", device)
    mean = cfg["_resolved"]["mean"]
    std = cfg["_resolved"]["std"]
    x = np.load(ROOT / "data" / "cache" / "X_train.npy", mmap_mode="r")
    y = np.load(ROOT / "data" / "cache" / "y_train.npy")
    indices = np.asarray(get_indices("screening")["val"], dtype=np.int64)

    meta = pd.read_csv(ROOT / "outputs" / "results" / "train_meta.csv")
    meta = meta[meta["usable_for_train"] == 1].reset_index(drop=True)

    preds = pd.read_csv(RUNS / "v2_d2_s42" / "predictions_val.csv")
    pred_map = {int(row.image_index): row for row in preds.itertuples()}
    errors = [i for i in indices if int(pred_map[int(i)].pred_label) != int(y[i])]
    rng = np.random.default_rng(42)
    random_sample = rng.choice(indices, size=180, replace=False).tolist()
    chosen = list(dict.fromkeys(random_sample + errors[:40]))

    target = model.features[-1][0]
    rows = []
    for idx in chosen:
        array = np.asarray(x[int(idx)])
        tensor = _array_to_tensor(array, mean, std).unsqueeze(0).to(device)
        tensor.requires_grad_(False)
        acts: list[torch.Tensor] = []
        handle = target.register_forward_hook(lambda _m, _i, out: acts.append(out))
        logits = model(tensor)
        handle.remove()
        pred = int(logits.argmax(dim=1).item())
        score = logits[0, pred]
        grad = torch.autograd.grad(score, acts[0])[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts[0]).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(64, 64), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        row = meta.iloc[int(idx)]
        w, h = float(row["width"]), float(row["height"])
        roi_mask = np.zeros((64, 64), dtype=bool)
        if int(row["roi_valid"]) == 1:
            x1 = int(round(float(row["roi_x1"]) * 64 / w))
            y1 = int(round(float(row["roi_y1"]) * 64 / h))
            x2 = int(round(float(row["roi_x2"]) * 64 / w))
            y2 = int(round(float(row["roi_y2"]) * 64 / h))
            roi_mask[max(0, y1) : min(64, y2), max(0, x1) : min(64, x2)] = True
        total = float(cam.sum()) or 1.0
        mass_in = float(cam[roi_mask].sum() / total) if roi_mask.any() else float("nan")
        prob = float(torch.softmax(logits.float(), dim=1)[0, pred].item())
        rows.append(
            {
                "image_index": int(idx),
                "true_label": int(y[int(idx)]),
                "pred_label": pred,
                "confidence": prob,
                "outcome": "correct" if pred == int(y[int(idx)]) else "wrong",
                "roi_valid": int(row["roi_valid"]),
                "roi_area_ratio": float(roi_mask.mean()),
                "cam_mass_in_roi": mass_in,
                "cam_mass_out_roi": 1.0 - mass_in if not math.isnan(mass_in) else float("nan"),
            }
        )
    _write_csv(
        "16_gradcam_roi_mass.csv",
        rows,
        ["image_index", "true_label", "pred_label", "confidence", "outcome", "roi_valid", "roi_area_ratio", "cam_mass_in_roi", "cam_mass_out_roi"],
    )
    # Lưu CAM cho các ảnh đã chọn để script hình tái dùng.
    cams = {}
    for idx in chosen[:220]:
        array = np.asarray(x[int(idx)])
        tensor = _array_to_tensor(array, mean, std).unsqueeze(0).to(device)
        acts = []
        handle = target.register_forward_hook(lambda _m, _i, out: acts.append(out))
        logits = model(tensor)
        handle.remove()
        pred = int(logits.argmax(dim=1).item())
        grad = torch.autograd.grad(logits[0, pred], acts[0])[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts[0]).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cams[str(int(idx))] = cam.astype(np.float32)
    np.savez_compressed(DATA / "gradcam_cams.npz", **cams)
    if device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# F. Phiếu rà soát lỗi thủ công
# ---------------------------------------------------------------------------


def compute_manual_review() -> None:
    import pandas as pd

    preds = pd.read_csv(RUNS / "v2_d2_s42" / "predictions_val.csv")
    y = np.load(ROOT / "data" / "cache" / "y_train.npy")
    meta = pd.read_csv(ROOT / "outputs" / "results" / "train_meta.csv")
    meta = meta[meta["usable_for_train"] == 1].reset_index(drop=True)
    errors = preds[preds["true_label"] != preds["pred_label"]].copy()
    names = _names()
    blur = np.load(DATA / "train_blur.npy")
    brightness = np.load(DATA / "train_brightness.npy")
    blur_p10 = float(np.percentile(blur, 10))
    bright_p05 = float(np.percentile(brightness, 5))
    bright_p95 = float(np.percentile(brightness, 95))
    rows = []
    for n, row in enumerate(errors.itertuples(), start=1):
        idx = int(row.image_index)
        path = meta.iloc[idx]["path"]
        track_id = Path(path).stem.split("_")[0]
        rows.append(
            {
                "review_id": n,
                "image_index": idx,
                "track_id": track_id,
                "path": path,
                "true_id": int(row.true_label),
                "true_name": names[str(int(row.true_label))]["vi"],
                "pred_id": int(row.pred_label),
                "pred_name": names[str(int(row.pred_label))]["vi"],
                "confidence": float(row.prob_pred),
                "prob_true": float(row.prob_true),
                "blur_laplacian": float(blur[idx]),
                "brightness": float(brightness[idx]),
                "auto_category": _auto_category(
                    float(row.prob_pred), float(blur[idx]), float(brightness[idx]),
                    blur_p10=blur_p10, bright_p05=bright_p05, bright_p95=bright_p95,
                ),
                "high_conf_error": int(float(row.prob_pred) >= 0.90),
                "verdict": "",
                "cause": "",
                "note": "",
                "reviewer": "",
                "reviewed_at": "",
            }
        )
    _write_csv(
        "17_manual_error_review.csv",
        rows,
        ["review_id", "image_index", "track_id", "path", "true_id", "true_name", "pred_id", "pred_name", "confidence", "prob_true", "blur_laplacian", "brightness", "auto_category", "high_conf_error", "verdict", "cause", "note", "reviewer", "reviewed_at"],
    )
    # Bản rút gọn: tối đa 2 ảnh mỗi track, đủ để xem thủ công mà không lặp frame.
    seen: Counter = Counter()
    diverse = []
    for row in sorted(rows, key=lambda r: (r["auto_category"], r["track_id"], -r["confidence"])):
        if seen[row["track_id"]] >= 2:
            continue
        seen[row["track_id"]] += 1
        diverse.append(row)
    _write_csv(
        "17b_manual_error_review_diverse.csv",
        diverse,
        ["review_id", "image_index", "track_id", "path", "true_id", "true_name", "pred_id", "pred_name", "confidence", "prob_true", "blur_laplacian", "brightness", "auto_category", "high_conf_error", "verdict", "cause", "note", "reviewer", "reviewed_at"],
    )
    _write_json(
        "17_manual_error_review_summary.json",
        {
            "n_errors": len(rows),
            "n_tracks_with_error": len(seen),
            "n_diverse_rows": len(diverse),
            "n_high_conf_errors": int(sum(row["high_conf_error"] for row in rows)),
            "by_category": dict(Counter(row["auto_category"] for row in rows)),
            "instruction": (
                "Mở contact sheet tương ứng, xem từng ảnh, điền verdict (model/du_lieu/nhan_sai/khong_ro), "
                "cause và note. Không sửa auto_category; đó là gợi ý heuristic. Lỗi tập trung vào ít track, "
                "nên đếm ảnh lỗi không phải đếm ca độc lập."
            ),
        },
    )


def _auto_category(
    confidence: float,
    blur: float,
    brightness: float,
    blur_p10: float = 0.0002,
    bright_p05: float = 0.08,
    bright_p95: float = 0.75,
) -> str:
    if brightness < bright_p05:
        return "toi"
    if brightness > bright_p95:
        return "choi"
    if blur < blur_p10:
        return "mo"
    if confidence < 0.60:
        return "confidence_thap"
    if confidence >= 0.90:
        return "tu_tin_cao_nhung_sai"
    return "goc_nhin_la"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default="all",
        choices=["tables", "probs", "quality", "benchmark", "gradcam", "review", "all"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _mkdirs()
    if args.only in ("tables", "all"):
        table_class_catalog()
        table_split_summary()
        table_model_comparison()
        table_eval_sets()
        table_per_class_lockbox()
        table_per_class_oof()
        table_top_confusions()
        table_oof_by_fold_seed()
        table_lockbox_by_seed()
        table_test_per_class()
        table_architecture_spec()
        table_experiment_index()
        table_training_cost()
        table_metric_worked_example()
    if args.only in ("probs", "all"):
        compute_screening_probs()
    if args.only in ("quality", "all"):
        compute_data_quality()
    if args.only in ("benchmark", "all"):
        compute_inference_benchmark()
    if args.only in ("gradcam", "all"):
        compute_gradcam_roi()
    if args.only in ("review", "all"):
        compute_manual_review()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
