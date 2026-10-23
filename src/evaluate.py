"""Đánh giá một checkpoint trên tập val: metric, dự đoán và confusion matrix.

Module này là **nơi duy nhất** tính metric để Giai đoạn 6 dùng lại, tránh hai nơi
tính khác nhau. Ở Giai đoạn 5, `train.py` vẫn dùng `evaluate()` nội bộ cho vòng
train; sau Giai đoạn 6 nên chuyển hẳn sang đây.

Nguyên tắc:
- Macro-F1 **luôn tính trên đủ 43 lớp** (``labels=range(43)``), lớp vắng mặt = 0.
- Chỉ đánh giá ``val``. ``test`` bị chặn cho tới Giai đoạn 6 (cần ``--confirm-freeze``).

Cách dùng::

    python -m src.evaluate --run outputs/runs/e3_cnnb_aug_s42 --split val
    python -m src.evaluate --run outputs/runs/diag_e3_group_s42 --split val --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
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
    get_device,
    seed_worker,
)
from .data import GTSRBTensorDataset, load_cache, load_split
from .models import build_model

LOG = logging.getLogger(__name__)

CONFUSION_FIG = "confusion_matrix.png"
PREDICTIONS_CSV = "predictions.csv"
METRICS_JSON = "metrics.json"


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Nạp ``best.pt`` (dict thuần) → model đã ở ``eval()`` trên ``device``."""
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy checkpoint {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model, _ = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def build_eval_loader(
    cfg: Mapping[str, Any], split: str = "val", batch_size: int = 256
) -> tuple[DataLoader, list[str], np.ndarray]:
    """Loader chỉ-đọc cho val: không augmentation, có chuẩn hóa; trả cả đường dẫn."""
    if split != "val":
        raise NotImplementedError(
            "Chỉ hỗ trợ split='val'. Tập test chỉ được đánh giá ở Giai đoạn 6 "
            "sau khi đóng băng configs/final.yaml (cờ --confirm-freeze)."
        )
    data_cfg = cfg["data"]
    size = int(data_cfg["image_size"])
    cache = load_cache(size)
    payload = load_split(data_cfg["split"])
    indices = payload["val"]
    dataset = GTSRBTensorDataset(
        cache["train"]["x"],
        cache["train"]["y"],
        indices,
        transform=None,
        mean=list(data_cfg["mean"]),
        std=list(data_cfg["std"]),
    )
    generator = torch.Generator()
    generator.manual_seed(int(cfg["experiment"].get("seed", 42)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    paths = [cache["train"]["paths"][i] for i in indices]
    labels = cache["train"]["y"][indices]
    return loader, paths, labels


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
    truths: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for inputs, targets in loader:
        logits = model(inputs.to(device, non_blocking=True))
        probability = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probs.append(probability)
        preds.append(probability.argmax(axis=1))
        truths.append(targets.numpy())
    return np.concatenate(truths), np.concatenate(preds), np.concatenate(probs)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    """Metric trên **đủ 43 lớp** + per-class + top cặp nhầm."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    labels = list(range(NUM_CLASSES))
    support = np.bincount(y_true, minlength=NUM_CLASSES)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present = [c for c in labels if support[c] > 0]
    macro_f1_43 = float(np.mean(f1))
    macro_f1_present = float(np.mean(f1[present])) if present else 0.0
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off_diagonal.ravel())[::-1], matrix.shape))[0]
    top_confusions = [
        {"true": int(t), "pred": int(p), "count": int(off_diagonal[t, p])}
        for t, p in pairs[:10]
        if off_diagonal[t, p] > 0
    ]
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1_43": macro_f1_43,
        "macro_f1_present_classes": macro_f1_present,
        "n_classes_present": len(present),
        "missing_classes": [int(c) for c in labels if support[c] == 0],
        "macro_f1_ceiling_43": (NUM_CLASSES - len([c for c in labels if support[c] == 0])) / NUM_CLASSES,
        "per_class": {
            str(c): {
                "precision": round(float(precision[c]), 6),
                "recall": round(float(recall[c]), 6),
                "f1": round(float(f1[c]), 6),
                "support": int(support[c]),
            }
            for c in labels
        },
        "top_confusions": top_confusions,
        "confusion_matrix": matrix.tolist(),
    }


def plot_confusion(
    matrix: np.ndarray, out_path: Path, title: str, class_names: Mapping[str, Any] | None = None
) -> None:
    """Confusion matrix chuẩn hóa theo hàng (recall mỗi lớp), ghi rõ trong tiêu đề."""
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
) -> None:
    """Ghi ``predictions.csv``: đường dẫn, nhãn thật, nhãn dự đoán, xác suất, đúng/sai."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "path", "label", "pred", "prob_pred", "correct"])
        for i, (path, truth, pred) in enumerate(zip(paths, y_true, y_pred)):
            writer.writerow([i, path, int(truth), int(pred), f"{float(y_prob[i, pred]):.6f}", int(truth == pred)])
    LOG.info("Đã ghi %s (%d dòng)", out_path, len(y_true))


def evaluate_run(run_dir: Path, split: str = "val", device: torch.device | None = None) -> dict[str, Any]:
    """Đánh giá ``run_dir/best.pt`` trên split val; lưu metric + dự đoán + hình."""
    device = device or get_device()
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
    cfg = checkpoint["config"]
    loader, paths, labels = build_eval_loader(cfg, split=split)
    y_true, y_pred, y_prob = predict_loader(model, loader, device)

    report = classification_metrics(y_true, y_pred)
    report.update(
        {
            "run_id": run_dir.name,
            "split": cfg["data"]["split"],
            "evaluated_on": split,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_val_macro_f1": float(checkpoint["val_macro_f1"]),
            "device": str(device),
        }
    )
    out_dir = run_dir / f"eval_{split}"
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8")) if CLASS_NAMES_PATH.is_file() else {}
    plot_confusion(np.asarray(report["confusion_matrix"]), out_dir / CONFUSION_FIG, run_dir.name, class_names)
    write_predictions(out_dir / PREDICTIONS_CSV, paths, y_true, y_pred, y_prob)
    (out_dir / METRICS_JSON).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info(
        "%s [%s/%s] acc %.6f | macro-F1(43) %.6f | macro-F1(lớp có mặt) %.6f | thiếu %d lớp",
        run_dir.name,
        cfg["data"]["split"],
        split,
        report["accuracy"],
        report["macro_f1_43"],
        report["macro_f1_present_classes"],
        len(report["missing_classes"]),
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint trên val (Giai đoạn 5–6)")
    parser.add_argument("--run", type=str, required=True, help="thư mục run, ví dụ outputs/runs/e3_cnnb_aug_s42")
    parser.add_argument("--split", choices=("val",), default="val", help="chỉ hỗ trợ val ở giai đoạn này")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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
    evaluate_run(run_dir, split=args.split, device=get_device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
