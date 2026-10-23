"""Đánh giá checkpoint: metric, dự đoán chi tiết và confusion matrix.

Là **nơi duy nhất** tính metric cho val/test (khớp với ``train.py::evaluate`` khi
chọn checkpoint). Quy tắc:

- Macro-F1 **luôn tính trên đủ 43 lớp** (``labels=range(43)``), lớp vắng = 0.
- ``predictions.csv`` theo schema Phụ lục C (kèm ROI ratio, độ mờ, độ sáng, top-3).
- ``test`` chỉ chạy khi có ``--confirm-freeze``, ghi log và **không ghi đè** report cũ
  (chạy lại do bug phải dùng ``--allow-rerun --reason "..."`` → sinh bản ``_v2``).

Cách dùng::

    python -m src.evaluate --run outputs/runs/final_s42 --split val
    python -m src.evaluate --run outputs/runs/final_s42 --split test --confirm-freeze
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    CLASS_NAMES_PATH,
    NUM_CLASSES,
    PROJECT_ROOT,
    RESULTS_DIR,
    get_device,
    git_commit,
    load_yaml,
    seed_worker,
)
from .data import GTSRBTensorDataset, TRAIN_META_PATH, load_cache, load_split, read_test_annotations
from .models import build_model

LOG = logging.getLogger(__name__)

CONFUSION_FIG = "confusion_matrix.png"
PREDICTIONS_CSV = "predictions.csv"
METRICS_JSON = "metrics.json"
TEST_REPORT = RESULTS_DIR / "test_report.json"
TEST_FINAL = RESULTS_DIR / "test_final.json"
TEST_LOG = RESULTS_DIR / "test_runs.log"

PREDICTION_COLUMNS = (
    "path",
    "true_id",
    "true_name",
    "pred_id",
    "pred_name",
    "prob_true",
    "prob_pred",
    "margin",
    "top3",
    "blur_laplacian",
    "brightness_mean",
    "roi_area_ratio",
    "is_correct",
)


# ---------------------------------------------------------------------------
# Nạp model và dữ liệu
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 của file (dùng cho ``checkpoint_sha256`` trong report)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Nạp ``best.pt`` (dict thuần) → model đã ``eval()`` trên ``device``."""
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy checkpoint {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model, _ = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _ratio_train(path: str, index: Mapping[str, tuple[int, int]], meta: Mapping[str, tuple[int, int, int]]) -> float:
    """Tỷ lệ diện tích ROI / diện tích ảnh gốc cho ảnh train."""
    width, height = index.get(path, (0, 0))
    x1, y1, x2, y2 = meta.get(path, (0, 0, 0, 0))
    if width <= 0 or height <= 0:
        return float("nan")
    return float(max(0, x2 - x1) * max(0, y2 - y1)) / float(width * height)


def _load_train_roi() -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int, int, int]]]:
    sizes: dict[str, tuple[int, int]] = {}
    boxes: dict[str, tuple[int, int, int, int]] = {}
    with TRAIN_META_PATH.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            path = row["path"]
            sizes[path] = (int(row["width"]), int(row["height"]))
            if int(row["roi_valid"]) == 1:
                boxes[path] = (
                    int(row["roi_x1"]),
                    int(row["roi_y1"]),
                    int(row["roi_x2"]),
                    int(row["roi_y2"]),
                )
    return sizes, boxes


def build_eval_loader(cfg: Mapping[str, Any], split: str = "val") -> dict[str, Any]:
    """Loader chỉ-đọc cho ``val`` hoặc ``test``; kèm đường dẫn, nhãn, ROI ratio, ảnh uint8."""
    if split not in ("val", "test"):
        raise ValueError("split phải là 'val' hoặc 'test'")
    data_cfg = cfg["data"]
    size = int(data_cfg["image_size"])
    cache = load_cache(size)
    annotations = {Path(r["Filename"]).stem: r for r in read_test_annotations()}

    if split == "val":
        payload = load_split(data_cfg["split"])
        indices = payload["val"]
        images = cache["train"]["x"]
        labels = cache["train"]["y"]
        paths_all = cache["train"]["paths"]
        sizes, boxes = _load_train_roi()
        ratios = np.array([_ratio_train(paths_all[i], sizes, boxes) for i in indices], dtype=np.float32)
    else:
        indices = list(range(len(cache["test"]["y"])))
        images = cache["test"]["x"]
        labels = cache["test"]["y"]
        paths_all = cache["test"]["paths"]
        ratios = np.empty(len(indices), dtype=np.float32)
        for i, path in enumerate(paths_all):
            row = annotations.get(Path(path).stem)
            if row is None or int(row["Width"]) <= 0:
                ratios[i] = np.nan
                continue
            area = max(0, int(row["Roi.X2"]) - int(row["Roi.X1"])) * max(
                0, int(row["Roi.Y2"]) - int(row["Roi.Y1"])
            )
            ratios[i] = area / (int(row["Width"]) * int(row["Height"]))

    dataset = GTSRBTensorDataset(
        images,
        labels,
        indices,
        transform=None,
        mean=list(data_cfg["mean"]),
        std=list(data_cfg["std"]),
    )
    generator = torch.Generator()
    generator.manual_seed(int(cfg["experiment"].get("seed", 42)))
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return {
        "loader": loader,
        "paths": [paths_all[i] for i in indices],
        "labels": labels[indices],
        "ratios": ratios,
        "images": images[indices],
        "split_name": data_cfg["split"],
    }


def quality_features(images: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """``(blur_laplacian, brightness_mean)`` cho từng ảnh uint8 HWC (64×64)."""
    from skimage.color import rgb2hsv, rgb2gray
    from skimage.filters import laplace

    n = len(images)
    blur = np.empty(n, dtype=np.float32)
    brightness = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        block = images[start : start + chunk].astype(np.float32) / 255.0
        for offset, image in enumerate(block):
            gray = rgb2gray(image)
            blur[start + offset] = float(np.var(laplace(gray)))
            brightness[start + offset] = float(rgb2hsv(image)[..., 2].mean())
    return blur, brightness


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả ``(y_true, y_pred, y_prob)``; luôn ``eval()`` + ``no_grad``."""
    model.eval()
    if model.training:
        raise RuntimeError("model vẫn ở chế độ train khi đánh giá")
    if torch.is_grad_enabled():
        raise RuntimeError("gradient đang bật khi đánh giá (thiếu no_grad)")
    truths, preds, probs = [], [], []
    for inputs, targets in loader:
        probability = torch.softmax(model(inputs.to(device, non_blocking=True)).float(), dim=1)
        probability = probability.cpu().numpy()
        probs.append(probability)
        preds.append(probability.argmax(axis=1))
        truths.append(targets.numpy())
    return np.concatenate(truths), np.concatenate(preds), np.concatenate(probs)


# ---------------------------------------------------------------------------
# Metric, hình, dự đoán
# ---------------------------------------------------------------------------


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Metric trên **đủ 43 lớp** + per-class + top cặp nhầm."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    labels = list(range(NUM_CLASSES))
    support = np.bincount(y_true, minlength=NUM_CLASSES)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present = [c for c in labels if support[c] > 0]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], matrix.shape))[0]
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1_43": float(np.mean(f1)),
        "macro_f1_present_classes": float(np.mean(f1[present])) if present else 0.0,
        "n_classes_present": len(present),
        "missing_classes": [int(c) for c in labels if support[c] == 0],
        "macro_f1_ceiling_43": (NUM_CLASSES - len([c for c in labels if support[c] == 0])) / NUM_CLASSES,
        "top3_accuracy": None,
        "per_class": {
            str(c): {
                "precision": round(float(precision[c]), 6),
                "recall": round(float(recall[c]), 6),
                "f1": round(float(f1[c]), 6),
                "support": int(support[c]),
            }
            for c in labels
        },
        "top_confusions": [
            {"true": int(t), "pred": int(p), "count": int(off[t, p])}
            for t, p in order[:10]
            if off[t, p] > 0
        ],
        "confusion_matrix": matrix.tolist(),
    }


def plot_confusion(matrix: np.ndarray, out_path: Path, title: str) -> None:
    """Confusion matrix chuẩn hóa theo hàng (recall mỗi lớp), 43×43."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(11, 10))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="tỷ lệ theo hàng (recall)")
    ax.set_xlabel("nhãn dự đoán")
    ax.set_ylabel("nhãn thật")
    ax.set_title(f"{title}\nConfusion matrix chuẩn hóa theo hàng, 43 lớp")
    ticks = np.arange(NUM_CLASSES)
    ax.set_xticks(ticks, ticks, fontsize=6, rotation=90)
    ax.set_yticks(ticks, ticks, fontsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    LOG.info("Đã ghi %s", out_path)


def write_predictions(
    out_path: Path,
    paths: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    ratios: np.ndarray,
    blur: np.ndarray,
    brightness: np.ndarray,
    class_names: Mapping[str, Mapping[str, str]],
) -> None:
    """Ghi ``predictions.csv`` đúng schema Phụ lục C (không sửa tay)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(PREDICTION_COLUMNS)
        for i, path in enumerate(paths):
            prob = y_prob[i]
            top = np.argsort(prob)[::-1][:3]
            top3 = "|".join(f"{int(c)}:{float(prob[c]):.6f}" for c in top)
            true_id, pred_id = int(y_true[i]), int(y_pred[i])
            writer.writerow(
                [
                    path,
                    true_id,
                    class_names.get(str(true_id), {}).get("vi", ""),
                    pred_id,
                    class_names.get(str(pred_id), {}).get("vi", ""),
                    f"{float(prob[true_id]):.6f}",
                    f"{float(prob[pred_id]):.6f}",
                    f"{float(prob[pred_id] - np.partition(prob, -2)[-2]):.6f}",
                    top3,
                    f"{float(blur[i]):.6f}",
                    f"{float(brightness[i]):.6f}",
                    "nan" if not np.isfinite(ratios[i]) else f"{float(ratios[i]):.6f}",
                    int(true_id == pred_id),
                ]
            )
    LOG.info("Đã ghi %s (%d dòng)", out_path, len(paths))


# ---------------------------------------------------------------------------
# Hai entrypoint: val và test
# ---------------------------------------------------------------------------


def evaluate_run(run_dir: Path, split: str = "val", device: torch.device | None = None) -> dict[str, Any]:
    """Đánh giá ``best.pt`` trên val; lưu metric + dự đoán + confusion matrix."""
    device = device or get_device()
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
    cfg = checkpoint["config"]
    data = build_eval_loader(cfg, split=split)
    y_true, y_pred, y_prob = predict_loader(model, data["loader"], device)
    blur, brightness = quality_features(data["images"])

    report = classification_metrics(y_true, y_pred)
    top3_hits = 0
    for i in range(len(y_true)):
        top3_hits += int(y_true[i] in np.argsort(y_prob[i])[::-1][:3])
    report["top3_accuracy"] = top3_hits / len(y_true)
    report.update(
        {
            "run_id": run_dir.name,
            "split": cfg["data"]["split"],
            "evaluated_on": split,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_val_macro_f1": float(checkpoint["val_macro_f1"]),
            "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
            "git_commit": git_commit(short=False),
            "device": str(device),
        }
    )
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    out_dir = run_dir / f"eval_{split}"
    plot_confusion(np.asarray(report["confusion_matrix"]), out_dir / CONFUSION_FIG, run_dir.name)
    write_predictions(
        out_dir / PREDICTIONS_CSV,
        data["paths"],
        y_true,
        y_pred,
        y_prob,
        data["ratios"],
        blur,
        brightness,
        class_names,
    )
    (out_dir / METRICS_JSON).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info(
        "%s [%s/%s] acc %.6f | macro-F1(43) %.6f | top-3 %.6f | thiếu %d lớp",
        run_dir.name,
        cfg["data"]["split"],
        split,
        report["accuracy"],
        report["macro_f1_43"],
        report["top3_accuracy"],
        len(report["missing_classes"]),
    )
    return report


def _check_freeze(run_dir: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Chặn chạy test nếu chưa đóng băng hoặc run không khớp final.yaml."""
    from .config import load_config

    final_cfg = PROJECT_ROOT / "configs" / "final.yaml"
    if not final_cfg.is_file():
        raise RuntimeError("Chưa có configs/final.yaml — không được chạy test (Giai đoạn 6.3.1)")
    # load_config merge với default.yaml để so sánh đủ khóa (không so file thô).
    frozen = load_config(final_cfg)
    if checkpoint["config"]["experiment"].get("id") != "final":
        raise RuntimeError(
            f"Checkpoint của run '{checkpoint['config']['experiment'].get('id')}' không phải run final"
        )
    for key in ("model", "augmentation", "train", "data"):
        current = dict(checkpoint["config"][key])
        expected = dict(frozen[key])
        for field in ("mean", "std"):
            current.pop(field, None)
            expected.pop(field, None)
        if current != expected:
            raise RuntimeError(f"Checkpoint khác configs/final.yaml ở nhóm '{key}' — từ chối chạy test")
    return frozen


def run_test_evaluation(
    run_dir: Path,
    device: torch.device | None = None,
    allow_rerun: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    """Đánh giá test đúng một lần: report + predictions + log; không ghi đè."""
    if TEST_REPORT.is_file() and not allow_rerun:
        raise RuntimeError(
            f"{TEST_REPORT} đã tồn tại — test chỉ được đánh giá 1 lần. "
            "Nếu thực sự phải chạy lại do bug: --allow-rerun --reason \"...\" (sinh bản _v2)."
        )
    if allow_rerun and not reason.strip():
        raise RuntimeError("--allow-rerun bắt buộc kèm --reason để ghi lý do vào log")
    device = device or get_device()
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
    _check_freeze(run_dir, checkpoint)

    data = build_eval_loader(checkpoint["config"], split="test")
    y_true, y_pred, y_prob = predict_loader(model, data["loader"], device)
    blur, brightness = quality_features(data["images"])
    report = classification_metrics(y_true, y_pred)
    top3_hits = sum(int(y_true[i] in np.argsort(y_prob[i])[::-1][:3]) for i in range(len(y_true)))
    report["top3_accuracy"] = top3_hits / len(y_true)
    report.update(
        {
            "run_id": run_dir.name,
            "split": "test",
            "evaluated_on": "test",
            "n_test_images": int(len(y_true)),
            "unique_paths": len(set(data["paths"])),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_val_macro_f1": float(checkpoint["val_macro_f1"]),
            "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
            "git_commit": git_commit(short=False),
            "torch_version": str(torch.__version__),
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "rerun": bool(allow_rerun),
            "rerun_reason": reason if allow_rerun else "",
        }
    )
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    suffix = "_v2" if allow_rerun else ""
    out_dir = run_dir / "eval_test"
    plot_confusion(np.asarray(report["confusion_matrix"]), out_dir / CONFUSION_FIG, f"{run_dir.name} — TEST")
    write_predictions(
        out_dir / PREDICTIONS_CSV,
        data["paths"],
        y_true,
        y_pred,
        y_prob,
        data["ratios"],
        blur,
        brightness,
        class_names,
    )
    (out_dir / METRICS_JSON).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    (RESULTS_DIR / f"test_report{suffix}.json").write_text(payload, encoding="utf-8")
    (RESULTS_DIR / f"test_final{suffix}.json").write_text(payload, encoding="utf-8")
    with TEST_LOG.open("a", encoding="utf-8") as fh:
        fh.write(
            f"{report['evaluated_at']} | run={run_dir.name} | n={len(y_true)} | "
            f"acc={report['accuracy']:.6f} | macro_f1={report['macro_f1_43']:.6f} | "
            f"sha256={report['checkpoint_sha256'][:16]} | git={report['git_commit'][:10]} | "
            f"rerun={report['rerun']} | reason={report['rerun_reason'] or '-'}\n"
        )
    LOG.info(
        "TEST %s: acc %.6f | macro-F1(43) %.6f | top-3 %.6f | %d ảnh",
        run_dir.name,
        report["accuracy"],
        report["macro_f1_43"],
        report["top3_accuracy"],
        len(y_true),
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint trên val/test")
    parser.add_argument("--run", type=str, required=True, help="thư mục run, ví dụ outputs/runs/final_s42")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--confirm-freeze", action="store_true", help="bắt buộc để chạy test")
    parser.add_argument("--allow-rerun", action="store_true", help="chạy lại test do bug (sinh bản _v2)")
    parser.add_argument("--reason", type=str, default="", help="lý do chạy lại test (bắt buộc với --allow-rerun)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = parse_args(argv)
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    device = get_device(args.device)

    if args.split == "test":
        if not args.confirm_freeze:
            LOG.error("Chạy test bắt buộc có --confirm-freeze (kỷ luật test 1 lần)")
            return 2
        run_test_evaluation(
            run_dir, device=device, allow_rerun=args.allow_rerun, reason=args.reason
        )
    else:
        evaluate_run(run_dir, split="val", device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
