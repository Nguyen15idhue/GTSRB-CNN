"""Hình cho lockbox: confusion matrix chuẩn hóa hàng (trung bình 3 seed) + per-class F1.

Cách dùng::

    python scripts/lockbox_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "results" / "v2"
FIG = ROOT / "outputs" / "figures" / "v2"
NUM_CLASSES = 43


def main() -> int:
    result = json.loads((OUT / "lockbox_result.json").read_text(encoding="utf-8"))
    matrices = [np.asarray(r["confusion_matrix"], dtype=float) for r in result["reports"]]
    matrix = np.mean(matrices, axis=0)
    row = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row, out=np.zeros_like(matrix), where=row > 0)
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="tỷ lệ theo hàng")
    ax.set_xticks(range(NUM_CLASSES), range(NUM_CLASSES), fontsize=6, rotation=90)
    ax.set_yticks(range(NUM_CLASSES), range(NUM_CLASSES), fontsize=6)
    ax.set_title("Lockbox 8.302 ảnh — confusion chuẩn hóa hàng (trung bình 3 seed)")
    fig.tight_layout()
    path = FIG / "10_confusion_lockbox.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Per-class F1 trung bình 3 seed + support.
    f1 = np.mean([[r["per_class"][str(c)]["f1"] for c in range(NUM_CLASSES)] for r in result["reports"]], axis=0)
    support = np.array([result["reports"][0]["per_class"][str(c)]["support"] for c in range(NUM_CLASSES)])
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(NUM_CLASSES), f1[order], color="#2e86c1")
    ax.set_xticks(range(NUM_CLASSES), [str(c) for c in order], fontsize=7, rotation=90)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("F1 (trung bình 3 seed)")
    ax.set_title("Per-class F1 trên lockbox — sắp tăng dần (nhãn = class id)")
    for pos, idx in enumerate(order):
        if f1[idx] < 0.9:
            ax.text(pos, f1[idx] + 0.01, f"n={support[idx]}", ha="center", fontsize=6, rotation=90)
    fig.tight_layout()
    fig.savefig(FIG / "11_per_class_f1_lockbox.png", dpi=150)
    plt.close(fig)

    with (OUT / "lockbox_per_class.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class_id", "f1_mean", "f1_sd", "support"])
        for c in range(NUM_CLASSES):
            values = [r["per_class"][str(c)]["f1"] for r in result["reports"]]
            writer.writerow([c, round(float(np.mean(values)), 6), round(float(np.std(values, ddof=1)), 6), int(support[c])])
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
