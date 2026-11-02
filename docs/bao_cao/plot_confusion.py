"""Vẽ lại confusion matrix test có ghi số ảnh, chỉ đọc artifact đã lưu.

Chạy: python docs/bao_cao/plot_confusion.py

Ra:
    outputs/figures/10_confusion_full_counts.png
    outputs/figures/10_confusion_zoom15_counts.png
    outputs/results/confusion_matrix_test.csv

KHÔNG train, KHÔNG chạy lại test: ma trận lấy từ ``test_final.json`` và được
kiểm chứng chéo với ``predictions_test_final.csv`` (12.630 dòng) trước khi vẽ.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"
CLASS_NAMES = ROOT / "configs" / "class_names.json"
N_CLASSES = 43

LOG = logging.getLogger("plot_confusion")


def load_report() -> dict:
    return json.loads((RES / "test_final.json").read_text(encoding="utf-8"))


def load_predictions() -> tuple[np.ndarray, np.ndarray]:
    path = RES / "predictions_test_final.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    y_true = np.array([int(r["true_id"]) for r in rows], dtype=np.int64)
    y_pred = np.array([int(r["pred_id"]) for r in rows], dtype=np.int64)
    return y_true, y_pred


def load_class_names() -> dict:
    return json.loads(CLASS_NAMES.read_text(encoding="utf-8"))


def build_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Dựng 43x43 số đếm không cần sklearn."""
    flat = y_true * N_CLASSES + y_pred
    return np.bincount(flat, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES).astype(np.int64)


def macro_f1_from_matrix(matrix: np.ndarray) -> float:
    """Macro-F1/43 tính trực tiếp từ ma trận (zero_division=0 như sklearn)."""
    true_positive = np.diag(matrix).astype(float)
    row_sum = matrix.sum(axis=1).astype(float)
    col_sum = matrix.sum(axis=0).astype(float)
    precision = np.divide(true_positive, col_sum, out=np.zeros_like(true_positive), where=col_sum > 0)
    recall = np.divide(true_positive, row_sum, out=np.zeros_like(true_positive), where=row_sum > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(true_positive), where=denom > 0)
    return float(f1.mean())


def verify(matrix: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, report: dict) -> None:
    """Chặn mọi sai lệch giữa ma trận và file test đã đóng băng."""
    if matrix.shape != (N_CLASSES, N_CLASSES):
        raise RuntimeError(f"Ma trận phải 43x43, nhận {matrix.shape}")
    if int(matrix.sum()) != int(report["n_samples"]) or len(y_true) != int(report["n_samples"]):
        raise RuntimeError(
            f"Tổng ma trận {int(matrix.sum())} / n_samples {report['n_samples']} "
            f"/ số dòng dự đoán {len(y_true)} không khớp"
        )
    rebuilt = build_matrix(y_true, y_pred)
    if not np.array_equal(matrix, rebuilt):
        diff = int(np.abs(matrix - rebuilt).sum())
        raise RuntimeError(f"Ma trận JSON lệch {diff} so với dựng lại từ predictions CSV")
    if not np.array_equal(matrix.sum(axis=1), np.bincount(y_true, minlength=N_CLASSES)):
        raise RuntimeError("Tổng hàng ma trận không bằng support từng lớp")

    accuracy = float(np.trace(matrix) / matrix.sum())
    macro_f1 = macro_f1_from_matrix(matrix)
    if abs(accuracy - float(report["accuracy"])) > 1e-12:
        raise RuntimeError(f"Accuracy tính lại {accuracy} lệch report {report['accuracy']}")
    if abs(macro_f1 - float(report["macro_f1_43"])) > 1e-12:
        raise RuntimeError(f"Macro-F1 tính lại {macro_f1} lệch report {report['macro_f1_43']}")
    LOG.info(
        "Đã kiểm chứng: %d ảnh | đúng %d | sai %d | acc %.7f | macro-F1 %.7f",
        int(matrix.sum()),
        int(np.trace(matrix)),
        int(matrix.sum() - np.trace(matrix)),
        accuracy,
        macro_f1,
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums > 0,
    )


def plot_full(matrix: np.ndarray, report: dict, out_path: Path) -> Path:
    """43x43: màu = tỷ lệ theo hàng, số = số ảnh (chỉ ghi ô khác 0)."""
    normalized = _normalize_rows(matrix)
    accuracy = float(np.trace(matrix) / matrix.sum())
    fig, ax = plt.subplots(figsize=(14, 13))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="tỷ lệ theo hàng (recall)")

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            count = int(matrix[i, j])
            if count == 0:
                continue
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if normalized[i, j] < 0.5 else "black",
            )

    ticks = np.arange(N_CLASSES)
    ax.set_xticks(ticks, ticks, fontsize=6, rotation=90)
    ax.set_yticks(ticks, ticks, fontsize=6)
    ax.set_xlabel("nhãn dự đoán")
    ax.set_ylabel("nhãn thật")
    ax.set_title(
        f"{report['run_id']} — test {int(matrix.sum()):,} ảnh, acc {accuracy:.6f}\n"
        "Màu: tỷ lệ theo hàng · Số trong ô: số ảnh thật (ô trống = 0)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def weakest_ids(report: dict, k: int = 15) -> list[int]:
    """Lấy k lớp F1 thấp nhất (tăng dần) từ per_class của test_final.json."""
    items = [(int(cid), float(payload["f1"])) for cid, payload in report["per_class"].items()]
    items.sort(key=lambda item: item[1])
    return [cid for cid, _ in items[:k]]


def plot_zoom(matrix: np.ndarray, ids: list[int], names: dict, out_path: Path) -> Path:
    """Zoom 15 lớp F1 thấp nhất, ghi số ảnh để đọc trên báo cáo/slide."""
    sub = matrix[np.ix_(ids, ids)]
    normalized = _normalize_rows(sub)
    fig, ax = plt.subplots(figsize=(11.5, 9))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="magma")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="tỷ lệ theo hàng")

    for i in range(len(ids)):
        for j in range(len(ids)):
            count = int(sub[i, j])
            if count == 0:
                continue
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if normalized[i, j] < 0.6 else "black",
            )

    labels = [f"{cid} · {names.get(str(cid), {}).get('vi', '')}" for cid in ids]
    ax.set_xticks(range(len(ids)), ids, fontsize=8, rotation=90)
    ax.set_yticks(range(len(ids)), labels, fontsize=7)
    ax.set_xlabel("nhãn dự đoán")
    ax.set_ylabel("nhãn thật")
    ax.set_title(
        "Zoom 15 lớp F1 thấp nhất trên test\n"
        "Màu: tỷ lệ theo hàng · Số trong ô: số ảnh thật"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def write_csv(matrix: np.ndarray, out_path: Path) -> Path:
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *range(N_CLASSES)])
        for i in range(N_CLASSES):
            writer.writerow([i, *matrix[i].tolist()])
    return out_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
    report = load_report()
    matrix = np.asarray(report["confusion_matrix"], dtype=np.int64)
    y_true, y_pred = load_predictions()
    verify(matrix, y_true, y_pred, report)

    names = load_class_names()
    weakest = weakest_ids(report, 15)
    full = plot_full(matrix, report, FIG / "10_confusion_full_counts.png")
    zoom = plot_zoom(matrix, weakest, names, FIG / "10_confusion_zoom15_counts.png")
    csv_path = write_csv(matrix, RES / "confusion_matrix_test.csv")

    for path in (full, zoom, csv_path):
        LOG.info("Đã ghi %s", path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
