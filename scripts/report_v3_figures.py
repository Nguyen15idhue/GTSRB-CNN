"""Sinh hình cho bản báo cáo làm lại (report pack v3).

Chỉ đọc artifact đã có trong ``outputs/results/v3`` + ``outputs/results/v2``;
không chạy lại train, không mở lockbox.

Cách dùng::

    python scripts/report_v3_figures.py --only all
    python scripts/report_v3_figures.py --only 1,2,3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "results" / "v3"
TABLES = OUT / "tables"
DATA = OUT / "data"
FIG = ROOT / "outputs" / "figures" / "v3"
V2 = ROOT / "outputs" / "results" / "v2"
RUNS = ROOT / "outputs" / "runs"
CLASS_NAMES = ROOT / "configs" / "class_names.json"

LOG = logging.getLogger("report_v3_fig")
NUM_CLASSES = 43

BLUE = "#2E86C1"
ORANGE = "#E67E22"
RED = "#C0392B"
GREEN = "#27AE60"
PURPLE = "#8E44AD"
GRAY = "#7F8C8D"
LIGHT = "#D5DBDB"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _names() -> dict[str, dict[str, str]]:
    return _json(CLASS_NAMES)


def _save(fig, name: str) -> Path:
    path = FIG / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("ghi %s", path.relative_to(ROOT))
    return path


def _short(text: str, limit: int = 26) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _load_original_crop(path: str, roi: Sequence[float] | None, size: int = 128) -> np.ndarray:
    """Đọc ảnh gốc, crop ROI, resize để hiển thị (không dùng cho model)."""
    from PIL import Image

    image = Image.open(ROOT / path).convert("RGB")
    if roi is not None and len(roi) == 4 and all(float(v) >= 0 for v in roi):
        x1, y1, x2, y2 = [int(round(float(v))) for v in roi]
        if x2 > x1 and y2 > y1:
            image = image.crop((x1, y1, x2, y2))
    return np.asarray(image.resize((size, size), Image.BILINEAR))


# ---------------------------------------------------------------------------
# 01. Phân bố 43 lớp
# ---------------------------------------------------------------------------


def fig_class_distribution() -> None:
    rows = _read_csv(TABLES / "01_class_catalog.csv")
    counts = np.array([int(r["n_pool"]) for r in rows])
    names = [f'{r["class_id"]} · {_short(r["name_vi"], 24)}' for r in rows]
    order = np.argsort(counts)
    counts_sorted = counts[order]
    names_sorted = [names[i] for i in order]
    colors = [RED if value < 300 else BLUE for value in counts_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(13, 10), gridspec_kw={"width_ratios": [2.4, 1]})
    ax = axes[0]
    bars = ax.barh(np.arange(len(counts_sorted)), counts_sorted, color=colors, edgecolor="white")
    ax.set_yticks(np.arange(len(counts_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=7.2)
    ax.set_xlabel("Số ảnh trong pool 39.201")
    ax.set_title("Số ảnh từng lớp (đỏ: dưới 300 ảnh)")
    for bar, value in zip(bars, counts_sorted):
        ax.text(value + 20, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=6.5)
    ax.set_xlim(0, counts_sorted.max() * 1.12)

    ax = axes[1]
    cumulative = np.cumsum(np.sort(counts)) / counts.sum()
    share = np.arange(1, len(counts) + 1) / len(counts)
    ax.plot(share * 100, cumulative * 100, color=BLUE, lw=2)
    ax.plot([0, 100], [0, 100], ls="--", color=GRAY, lw=1, label="cân bằng lý tưởng")
    ax.fill_between(share * 100, cumulative * 100, share * 100, color=BLUE, alpha=0.12)
    top10 = np.sort(counts)[-10:].sum() / counts.sum() * 100
    ax.axvline(100 * 33 / 43, color=ORANGE, ls=":", lw=1.2)
    ax.annotate(
        f"23% số lớp lớn nhất\nchiếm {top10:.1f}% ảnh",
        xy=(100 * 33 / 43, cumulative[32] * 100),
        xytext=(14, 72),
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=ORANGE,
    )
    ax.set_xlabel("% số lớp (xếp tăng dần)")
    ax.set_ylabel("% ảnh tích lũy")
    ax.set_title("Mức mất cân bằng")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.suptitle("Phân bố 43 lớp GTSRB — pool dùng chia split (39.201 ảnh)", fontsize=13)
    _save(fig, "01_class_distribution.png")


# ---------------------------------------------------------------------------
# 02. Sơ đồ chia tập
# ---------------------------------------------------------------------------


def _box(ax, x, y, w, h, title, lines, color, edge=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=edge or color,
        facecolor=color + "1A",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.038, title, ha="center", va="top", fontsize=10, weight="bold", color=edge or color)
    ax.text(
        x + w / 2,
        y + h - 0.098,
        "\n".join(lines),
        ha="center",
        va="top",
        fontsize=7.9,
        color="#222222",
        linespacing=1.35,
    )


def _arrow(ax, start, end, color=GRAY, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=14, linewidth=1.3,
            color=color, linestyle=ls, shrinkA=3, shrinkB=3,
        )
    )


def fig_split_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.36, 0.84, 0.28, 0.12, "Pool 39.201 ảnh", ["1.307 track · 1.183 sign-group · 43 lớp"], GRAY)
    _box(ax, 0.04, 0.58, 0.30, 0.16, "LOCKBOX 8.302 ảnh", ["277 track · 238 sign-group", "seed 2024 · mở 1 lần sau freeze"], RED)
    _box(ax, 0.55, 0.58, 0.30, 0.16, "DEV 30.899 ảnh", ["1.030 track · 945 sign-group", "dùng để chọn và train final"], GREEN)
    _box(ax, 0.02, 0.30, 0.29, 0.16, "Screening 24.719 / 6.180", ["80/20 sign-group của dev · seed 42", "factorial, kiến trúc, hyperparam, aug"], BLUE)
    _box(ax, 0.36, 0.30, 0.28, 0.16, "OOF 3 fold × 3 seed", ["19.949–21.120 train", "9.779–10.950 val / fold"], PURPLE)
    _box(ax, 0.69, 0.30, 0.29, 0.16, "Final train (3 seed)", ["toàn bộ dev · 26 epoch", "đánh giá trên lockbox"], ORANGE)
    _box(ax, 0.36, 0.06, 0.28, 0.14, "Official test 12.630", ["không track-disjoint", "751 ảnh gần trùng · benchmark phụ"], GRAY)

    _arrow(ax, (0.44, 0.84), (0.20, 0.74), color=RED)
    _arrow(ax, (0.56, 0.84), (0.70, 0.74), color=GREEN)
    _arrow(ax, (0.60, 0.58), (0.20, 0.46), color=BLUE)
    _arrow(ax, (0.69, 0.58), (0.50, 0.46), color=PURPLE)
    _arrow(ax, (0.80, 0.58), (0.83, 0.46), color=ORANGE)

    ax.text(0.50, 0.02, "Nguyên tắc: lockbox không dùng để chọn mô hình, epoch, ngưỡng hay augmentation. Official test chỉ là benchmark phụ.",
            ha="center", fontsize=8.8, style="italic", color=RED)
    ax.set_title("Luồng chia dữ liệu 3 tầng theo sign-group", fontsize=14, pad=10)
    _save(fig, "02_split_diagram.png")


# ---------------------------------------------------------------------------
# 03. Kiến trúc D2
# ---------------------------------------------------------------------------


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top = [
        ("Input", "3×64×64", "0"),
        ("Conv 3×3 + BN\nReLU + MaxPool", "32×32×32", "928"),
        ("Conv 3×3 + BN\nReLU + MaxPool", "64×16×16", "18.560"),
        ("Conv 3×3 + BN\nReLU + MaxPool", "128×8×8", "73.984"),
        ("Conv 3×3 + BN\nReLU + MaxPool", "256×4×4", "295.424"),
    ]
    bottom = [
        ("Flatten", "4.096", "0"),
        ("FC 256 + ReLU\nDropout 0,3", "256", "1.048.832"),
        ("FC 43", "43 logits", "11.051"),
        ("Softmax\n(chỉ khi suy luận)", "43 xác suất", "0"),
    ]
    w, h = 0.155, 0.24
    for i, (title, shape, params) in enumerate(top):
        x = 0.02 + i * 0.19
        _box(ax, x, 0.58, w, h, title, [f"đầu ra: {shape}", f"params: {params}"], BLUE)
        if i < len(top) - 1:
            _arrow(ax, (x + w, 0.70), (x + 0.19, 0.70), color=BLUE)
    for i, (title, shape, params) in enumerate(bottom):
        x = 0.02 + i * 0.24
        _box(ax, x, 0.16, 0.19, h, title, [f"đầu ra: {shape}", f"params: {params}"], ORANGE if i < 3 else GRAY)
        if i < len(bottom) - 1:
            _arrow(ax, (x + 0.19, 0.28), (x + 0.24, 0.28), color=ORANGE if i < 2 else GRAY)
    _arrow(ax, (0.80, 0.58), (0.10, 0.40), color=GRAY, ls=":")
    ax.text(0.5, 0.04,
            "D2: 4 khối conv 32/64/128/256 · BN bật · dropout 0,3 · không GAP · 1.448.779 params · đầu ra logits",
            ha="center", fontsize=9.5, color="#333333")
    ax.set_title("Kiến trúc D2 (mô hình thắng) — đường đi của một ảnh RGB 64×64", fontsize=14, pad=10)
    _save(fig, "03_architecture_d2.png")


# ---------------------------------------------------------------------------
# 04. Learning curve train vs val
# ---------------------------------------------------------------------------


def _metrics(run_id: str) -> list[dict[str, str]]:
    return _read_csv(RUNS / run_id / "metrics.csv")


def fig_learning_curves() -> None:
    configs = [("D2 (winner)", "v2_d2_s42"), ("F07 (control)", "v2_f07_s42")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    for row, (label, run_id) in enumerate(configs):
        rows = _metrics(run_id)
        epochs = np.array([int(r["epoch"]) for r in rows])
        for col, (key_train, key_val, title, ylabel) in enumerate(
            [
                ("train_loss", "val_loss", "Loss", "loss"),
                ("train_acc", "val_acc", "Accuracy", "accuracy"),
            ]
        ):
            ax = axes[row, col]
            ax.plot(epochs, [float(r[key_train]) for r in rows], color=BLUE, lw=1.8, label="train")
            ax.plot(epochs, [float(r[key_val]) for r in rows], color=ORANGE, lw=1.8, label="validation")
            ax.set_title(f"{label} — {title}")
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            if key_val == "val_acc":
                ax.set_ylim(0.4, 1.02)
            else:
                ax.set_ylim(0, max(1.0, max(float(r[key_train]) for r in rows) * 1.05))
    fig.suptitle("Learning curve trên screening val: train so với validation (seed 42)", fontsize=13)
    fig.tight_layout()
    _save(fig, "04_learning_curves.png")


# ---------------------------------------------------------------------------
# 05. So sánh mô hình
# ---------------------------------------------------------------------------


def fig_model_comparison() -> None:
    rows = {r["config_id"]: r for r in _read_csv(TABLES / "05_model_comparison.csv")}
    order = ["b1", "f00", "f03", "f07", "f15", "d1", "d2", "d3", "w1", "w2", "l1", "cw1", "i2", "g1", "g2", "g3", "x1", "x2", "x3"]
    labels = {
        "b1": "B1 HOG+SVC", "f00": "F00 CNN-A", "f03": "F03 BN+DO", "f07": "F07 control",
        "f15": "F15 +aug", "d1": "D1 3 khối+GAP", "d2": "D2 4 khối", "d3": "D3 4 khối+GAP",
        "w1": "W1 width nhỏ", "w2": "W2 width lớn", "l1": "L1 LR 1e-4", "cw1": "CW1 class weight",
        "i2": "I2 input 96", "g1": "G1 affine", "g2": "G2 color", "g3": "G3 perspective",
        "x1": "X1 OOF G2", "x2": "X2 OOF D2", "x3": "X3 OOF control",
    }
    selected_all = [cid for cid in order if cid in rows]
    excluded = [cid for cid in selected_all if float(rows[cid]["macro_f1_pct"]) < 90]
    selected = [cid for cid in selected_all if float(rows[cid]["macro_f1_pct"]) >= 90]
    selected.sort(key=lambda cid: float(rows[cid]["macro_f1_pct"]))

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    y = np.arange(len(selected))
    for yi, cid in zip(y, selected):
        f1 = float(rows[cid]["macro_f1_pct"])
        acc = float(rows[cid]["accuracy_pct"])
        f1_sd = float(rows[cid]["macro_f1_sd_pct"] or 0)
        acc_sd = float(rows[cid]["accuracy_sd_pct"] or 0)
        ax.plot([acc, f1], [yi, yi], color=LIGHT, lw=2, zorder=1)
        ax.errorbar(acc, yi, xerr=acc_sd, fmt="o", color=BLUE, ms=5.5, capsize=2.5, lw=1.1, zorder=3)
        ax.errorbar(f1, yi, xerr=f1_sd, fmt="s", color=ORANGE, ms=5.5, capsize=2.5, lw=1.1, zorder=3)
        ax.text(88.2, yi, f'{int(float(rows[cid]["params"]))/1e6:.2f}M', va="center", fontsize=7, color=GRAY)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[c] for c in selected], fontsize=8.6)
    for tick, cid in zip(ax.get_yticklabels(), selected):
        if cid in ("d2", "x2"):
            tick.set_fontweight("bold")
            tick.set_color(GREEN)
    ax.set_xlim(88, 100.3)
    ax.set_xlabel("Điểm % (trung bình 3 seed; thanh lỗi = độ lệch chuẩn)")
    ax.set_title("So sánh cấu hình: Accuracy (●) và Macro-F1/43 (■)\ntrục phóng to 88–100; cột trái là số tham số")
    ax.axvline(97.158, color=GREEN, ls=":", lw=1.0)
    ax.plot([], [], "o", color=BLUE, label="Accuracy")
    ax.plot([], [], "s", color=ORANGE, label="Macro-F1/43")
    ax.legend(loc="lower right", fontsize=8.5)
    note = "Ngoài thang: " + "; ".join(
        f'{labels[c]} {float(rows[c]["accuracy_pct"]):.1f}/{float(rows[c]["macro_f1_pct"]):.1f}' for c in excluded
    )
    ax.text(88.3, -0.9, note + "  (Accuracy/Macro-F1)", fontsize=7.4, color=GRAY)
    ax.set_ylim(-1.4, len(selected) - 0.4)
    fig.tight_layout()
    _save(fig, "05_model_comparison.png")


# ---------------------------------------------------------------------------
# 06. Confusion matrix
# ---------------------------------------------------------------------------


def fig_confusion() -> None:
    from matplotlib.colors import PowerNorm

    matrix = np.load(DATA / "confusion_lockbox_mean.npy")
    names = _names()
    row_sum = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.05], wspace=0.62)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(norm, cmap="magma_r", norm=PowerNorm(gamma=0.45, vmin=0, vmax=1), interpolation="nearest")
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title("Confusion matrix lockbox — chuẩn hóa theo hàng (trung bình 3 seed)")
    ax.set_xticks(range(0, NUM_CLASSES, 2))
    ax.set_yticks(range(0, NUM_CLASSES, 2))
    ax.tick_params(labelsize=7)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.012)
    cbar.set_label("Tỷ lệ trong lớp thật", fontsize=8)
    for cid in range(NUM_CLASSES):
        ax.text(cid, cid, f"{int(round(row_sum[cid, 0]))}", ha="center", va="center", fontsize=4.2, color="white")

    ax2 = fig.add_subplot(gs[0, 1])
    rows = _read_csv(TABLES / "09_top_confusions_lockbox.csv")[:10]
    labels = [f'{r["true_id"]}→{r["pred_id"]}' for r in rows]
    values = [float(r["rate_of_true_class_pct"]) for r in rows]
    ax2.barh(np.arange(len(rows))[::-1], values, color=RED, alpha=0.8)
    ax2.set_yticks(np.arange(len(rows))[::-1])
    ax2.set_yticklabels(labels, fontsize=7.5)
    ax2.set_xlabel("% của lớp thật bị đoán nhầm")
    ax2.set_title("10 cặp nhầm nhiều nhất")
    for yi, value in zip(np.arange(len(rows))[::-1], values):
        ax2.text(value + 0.15, yi, f"{value:.1f}%", va="center", fontsize=7)
    ax2.set_xlim(0, max(values) * 1.25)
    fig.suptitle("Phân tích nhầm lẫn trên lockbox 8.302 ảnh", fontsize=13)
    _save(fig, "06_confusion_matrix.png")


# ---------------------------------------------------------------------------
# 07. F1 từng lớp
# ---------------------------------------------------------------------------


def fig_per_class_f1() -> None:
    rows = _read_csv(TABLES / "07_per_class_lockbox.csv")
    rows = sorted(rows, key=lambda r: float(r["f1_mean"]))
    labels = [f'{r["class_id"]} · {_short(r["name_vi"], 26)}' for r in rows]
    f1 = np.array([float(r["f1_mean"]) for r in rows])
    sd = np.array([float(r["f1_sd"]) for r in rows])
    support = [int(r["support"]) for r in rows]
    colors = [RED if value < 0.95 else (ORANGE if value < 0.99 else BLUE) for value in f1]

    fig, ax = plt.subplots(figsize=(10, 11))
    y = np.arange(len(rows))
    ax.axvspan(0.99, 1.005, color=GREEN, alpha=0.07)
    ax.hlines(y, 0.70, f1, color=LIGHT, lw=1.2)
    ax.errorbar(f1, y, xerr=sd, fmt="o", color="black", ms=0, lw=1, capsize=2)
    for yi, value, color in zip(y, f1, colors):
        ax.plot(value, yi, "o", ms=6, color=color, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{label}  (n={n})" for label, n in zip(labels, support)], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlim(0.70, 1.005)
    ax.set_xlabel("F1 trung bình 3 seed (thanh lỗi = độ lệch chuẩn)")
    ax.set_title("F1 từng lớp trên lockbox (8.302 ảnh) — lớp yếu nhất ở trên")
    ax.axvline(0.975981, color=GREEN, ls=":", lw=1.2, label="macro-F1 0,9760")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.992, 0.5, "≥ 0,99", fontsize=7.5, color=GREEN, ha="center")
    fig.tight_layout()
    _save(fig, "07_per_class_f1.png")


# ---------------------------------------------------------------------------
# 08. Ảnh hưởng factorial
# ---------------------------------------------------------------------------


def _factorial_means() -> dict[str, float]:
    import pandas as pd

    df = pd.read_csv(V2 / "summary.csv")
    df = df[df["config_id"].str.match(r"f\d\d")]
    means = df.groupby("config_id")["val_macro_f1"].mean()
    return {k: float(v) * 100 for k, v in means.items()}


def fig_factor_effects() -> None:
    effects = _json(V2 / "factorial_effects.json")
    labels = ["BN", "Dropout 0,3", "Weight decay", "Augmentation", "BN × Aug", "DO × WD"]
    keys = ["bn", "do", "wd", "aug", "bn_x_aug", "do_x_wd"]
    values = np.array([effects[k]["effect_mean"] * 100 for k in keys])
    lows = np.array([effects[k]["ci_low"] * 100 for k in keys])
    highs = np.array([effects[k]["ci_high"] * 100 for k in keys])
    colors = [GREEN if low > 0 else (RED if high < 0 else GRAY) for low, high in zip(lows, highs)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    y = np.arange(len(keys))[::-1]
    ax.errorbar(values, y, xerr=[values - lows, highs - values], fmt="o", color="black", ms=6, capsize=3)
    for yi, value, color in zip(y, values, colors):
        ax.plot(value, yi, "o", ms=7, color=color)
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ macro-F1 (điểm %) · 95% CI")
    ax.set_title("Hiệu ứng chính và tương tác")
    for yi, value in zip(y, values):
        ax.annotate(f"{value:+.2f}", (value, yi), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#333333")
    ax.set_xlim(min(lows) - 1.5, max(highs) + 1.8)

    means = _factorial_means()
    def mean_of(bit_bn: int, bit_aug: int) -> float:
        values_ = []
        for cid, value in means.items():
            idx = int(cid[1:])
            bn = (idx >> 3) & 1
            aug = (idx >> 0) & 1
            if bn == bit_bn and aug == bit_aug:
                values_.append(value)
        return float(np.mean(values_))

    ax = axes[1]
    x = np.array([0, 1])
    ax.plot(x, [mean_of(0, 0), mean_of(0, 1)], "o-", color=BLUE, label="BN tắt")
    ax.plot(x, [mean_of(1, 0), mean_of(1, 1)], "o-", color=ORANGE, label="BN bật")
    ax.set_xticks(x)
    ax.set_xticklabels(["Aug tắt", "Aug all"])
    ax.set_ylabel("Macro-F1 trung bình (điểm %)")
    ax.set_title("Tương tác BN × Augmentation")
    ax.legend(fontsize=8)
    points = [
        (0, mean_of(0, 0), (0, 9)),
        (1, mean_of(0, 1), (0, -16)),
        (0, mean_of(1, 0), (0, -16)),
        (1, mean_of(1, 1), (0, 9)),
    ]
    for xi, yi, offset in points:
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=offset, ha="center", fontsize=8)
    fig.suptitle("Factorial 2⁴: BN và Dropout có bằng chứng; WD và Augmentation thì chưa", fontsize=13)
    fig.tight_layout()
    _save(fig, "08_factor_effects.png")


# ---------------------------------------------------------------------------
# 09. OOF theo fold và seed
# ---------------------------------------------------------------------------


def fig_oof_fold_seed() -> None:
    rows = _read_csv(TABLES / "11_oof_by_fold_seed.csv")
    variance = _json(V2 / "variance.json")
    configs = ["x1", "x2", "x3"]
    colors = {42: BLUE, 123: ORANGE, 2024: GREEN}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    for ax, cid in zip(axes.flat[:3], configs):
        part = [r for r in rows if r["config_id"] == cid]
        for seed in (42, 123, 2024):
            values = [
                float(next(r["macro_f1_pct"] for r in part if int(r["seed"]) == seed and int(r["fold"]) == fold))
                for fold in (0, 1, 2)
            ]
            ax.plot([0, 1, 2], values, "o-", color=colors[seed], label=f"seed {seed}", lw=1.6, ms=5)
        fold_means = [
            np.mean([float(r["macro_f1_pct"]) for r in part if int(r["fold"]) == fold]) for fold in (0, 1, 2)
        ]
        ax.plot([0, 1, 2], fold_means, "k--", lw=1.2, label="trung bình fold")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["fold 0", "fold 1", "fold 2"])
        ax.set_ylabel("Macro-F1 (điểm %)")
        ax.set_title(f"{cid.upper()} — ICC fold {variance[cid]['icc_fold']:.2f}, ICC seed {variance[cid]['icc_seed']:.2f}")
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(90, 100)
    ax = axes[1, 1]
    pooled = variance["_pooled"]
    parts = [pooled["var_seed"], pooled["var_fold"], pooled["var_resid"]]
    total = sum(parts)
    labels = ["seed", "fold", "residual"]
    colors_bar = [ORANGE, BLUE, GRAY]
    ax.bar(labels, [p / total * 100 for p in parts], color=colors_bar)
    for i, p in enumerate(parts):
        ax.text(i, p / total * 100 + 1, f"{p/total*100:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("% phương sai")
    ax.set_title("Phân rã phương sai (pooled 3 cấu hình)")
    ax.set_ylim(0, 100)
    fig.suptitle("OOF 3 fold × 3 seed: fold là nguồn biến thiên chính, không phải seed", fontsize=13)
    fig.tight_layout()
    _save(fig, "09_oof_fold_seed.png")


# ---------------------------------------------------------------------------
# 10. Robustness
# ---------------------------------------------------------------------------


def fig_robustness() -> None:
    import pandas as pd

    df = pd.read_csv(V2 / "corruption.csv")
    labels = {
        "blur": "Blur", "noise": "Noise", "brightness": "Brightness",
        "contrast": "Contrast", "jpeg": "JPEG", "occlusion": "Occlusion",
    }
    configs = ["f07", "f15", "g1", "g2", "g3", "d2"]
    table = df.groupby(["config_id", "corruption"])["accuracy"].mean().unstack() * 100
    table = table.loc[configs, list(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    im = ax.imshow(table.values, cmap="RdYlGn", vmin=15, vmax=100, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([labels[c] for c in labels], rotation=20, ha="right")
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(["F07 control", "F15 +aug", "G1 affine", "G2 color", "G3 perspective", "D2 winner"])
    ax.set_title("Accuracy trung bình 3 mức corruption (%)")
    ax.grid(False)
    for i in range(len(configs)):
        for j in range(len(labels)):
            value = table.values[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if value < 55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    ax = axes[1]
    d2 = df[df["config_id"] == "d2"]
    for corruption in labels:
        part = d2[d2["corruption"] == corruption].sort_values("severity")
        ax.plot(part["severity"], part["accuracy"] * 100, "o-", label=labels[corruption], lw=1.5)
    ax.axhline(98.004, color=GRAY, ls="--", lw=1, label="clean 98,0")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Mức corruption (1 nhẹ → 3 nặng)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("D2: accuracy giảm theo mức corruption")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 102)
    fig.suptitle("Robustness: occlusion là điểm yếu lớn nhất (trung bình 3 mức chỉ 17,6%)", fontsize=13)
    fig.tight_layout()
    _save(fig, "10_robustness.png")


# ---------------------------------------------------------------------------
# 11. Grad-CAM
# ---------------------------------------------------------------------------


def fig_gradcam() -> None:
    import pandas as pd

    rows = pd.read_csv(TABLES / "16_gradcam_roi_mass.csv")
    cams = np.load(DATA / "gradcam_cams.npz")
    x = np.load(ROOT / "data" / "cache" / "X_train.npy", mmap_mode="r")
    y = np.load(ROOT / "data" / "cache" / "y_train.npy")
    meta = pd.read_csv(ROOT / "outputs" / "results" / "train_meta.csv")
    meta = meta[meta["usable_for_train"] == 1].reset_index(drop=True)
    names = _names()

    def pick(mask, n=4, sort_key="confidence", ascending=False):
        part = rows[mask].copy()
        if part.empty:
            return part
        part = part.sort_values(sort_key, ascending=ascending)
        return part.head(n)

    correct = rows["outcome"] == "correct"
    wrong = rows["outcome"] == "wrong"
    groups = [
        ("Đúng, tự tin cao", pick(correct & (rows["confidence"] >= 0.95), 4, "confidence", False), GREEN),
        ("Đúng, tự tin thấp", pick(correct & (rows["confidence"] < 0.80), 4, "confidence", True), ORANGE),
        ("Sai", pick(wrong, 4, "confidence", False), RED),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.4))
    for row_i, (title, part, color) in enumerate(groups):
        for col_i in range(4):
            ax = axes[row_i, col_i]
            ax.axis("off")
            if col_i >= len(part):
                continue
            record = part.iloc[col_i]
            idx = int(record["image_index"])
            image = np.asarray(x[idx])
            cam = cams[str(idx)] if str(idx) in cams else np.zeros((64, 64), dtype=np.float32)
            ax.imshow(image, interpolation="bilinear")
            ax.imshow(cam, cmap="jet", alpha=0.45, interpolation="bilinear")
            # ROI
            m = meta.iloc[idx]
            if int(m["roi_valid"]) == 1:
                w, h = float(m["width"]), float(m["height"])
                x1 = float(m["roi_x1"]) * 64 / w
                y1 = float(m["roi_y1"]) * 64 / h
                x2 = float(m["roi_x2"]) * 64 / w
                y2 = float(m["roi_y2"]) * 64 / h
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", lw=1.2, ls="--"))
            ax.set_title(
                f'thật {int(record["true_label"])} · đoán {int(record["pred_label"])}\n'
                f'conf {record["confidence"]:.2f} · ROI {record["cam_mass_in_roi"]:.2f}',
                fontsize=7.8, color=color,
            )
            if col_i == 0:
                ax.text(-0.10, 0.5, title, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10, weight="bold", color=color)
    fig.suptitle("Grad-CAM của D2: mô hình nhìn vào biển (mass trong ROI trung bình 0,85 khi đúng, 0,73 khi sai)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "11_gradcam.png")


# ---------------------------------------------------------------------------
# 12. Calibration
# ---------------------------------------------------------------------------


def fig_calibration() -> None:
    cal = np.load(DATA / "calibration_screening.npz")
    risk = np.load(DATA / "selective_risk_oof.npz")
    stats = _read_csv(TABLES / "12_calibration_confidence.csv")[0]
    tau = float(_json(TABLES / "threshold_v2_oof.json")["tau"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    conf = cal["bin_confidence"]
    acc = cal["bin_accuracy"]
    counts = cal["bin_count"]
    valid = ~np.isnan(conf)
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1, label="hoàn hảo")
    err = 1.96 * np.sqrt(np.clip(acc[valid] * (1 - acc[valid]) / np.maximum(counts[valid], 1), 0, None))
    ax.errorbar(conf[valid], acc[valid], yerr=err, fmt="o-", color=BLUE, ms=4.5, lw=1.5,
                capsize=2, label="D2 screening val (CI 95%)")
    ax2 = ax.twinx()
    ax2.bar(conf[valid], counts[valid], width=0.06, alpha=0.18, color=GRAY)
    ax2.set_ylabel("số ảnh / bin", color=GRAY, fontsize=8)
    ax2.tick_params(axis="y", colors=GRAY, labelsize=7)
    ax2.grid(False)
    ax.set_xlabel("Confidence dự đoán")
    ax.set_ylabel("Accuracy thực tế")
    ax.set_title(f"Reliability diagram · ECE {float(stats['ece_15bin']):.4f}")
    ax.legend(fontsize=7.2, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    correct = cal["correct"].astype(bool)
    ax.hist(cal["conf"][correct], bins=40, alpha=0.65, color=GREEN, label="đúng", log=True)
    ax.hist(cal["conf"][~correct], bins=40, alpha=0.75, color=RED, label="sai", log=True)
    ax.axvline(tau, color=ORANGE, ls="--", lw=1.2, label=f"τ = {tau:.3f}")
    ax.set_xlabel("Confidence (max prob)")
    ax.set_ylabel("Số ảnh (log)")
    ax.set_title("Phân bố confidence: đúng vs sai")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    ax.plot(risk["coverage"] * 100, risk["accuracy"] * 100, color=BLUE, lw=1.8)
    ax.axvline(95, color=ORANGE, ls="--", lw=1.2, label="coverage 95%")
    ax.set_xlabel("Coverage (% ảnh được chấp nhận)")
    ax.set_ylabel("Accuracy trên phần chấp nhận (%)")
    ax.set_title("Selective risk trên OOF (X2)")
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(90, 100.5)
    fig.suptitle("Calibration và selective prediction: tin cậy cao chưa chắc đúng, nhưng ngưỡng τ lọc được 80,5% lỗi",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "12_calibration.png")


# ---------------------------------------------------------------------------
# 13. Precision–Recall
# ---------------------------------------------------------------------------


def fig_precision_recall() -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    payload = np.load(DATA / "screening_val_probs_d2_s42.npz")
    probs = payload["probs"]
    labels = payload["labels"]
    names = _names()
    ap = np.array([average_precision_score((labels == c).astype(int), probs[:, c]) for c in range(NUM_CLASSES)])
    worst = np.argsort(ap)[:5]
    micro_ap = average_precision_score((labels[:, None] == np.arange(NUM_CLASSES)).ravel(), probs.ravel())

    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    precision, recall, _ = precision_recall_curve((labels[:, None] == np.arange(NUM_CLASSES)).ravel(), probs.ravel())
    ax.plot(recall, precision, color=BLUE, lw=2.2, label=f"micro-average (AP {micro_ap:.3f})")
    palette = [RED, ORANGE, PURPLE, GREEN, "#16A085"]
    for color, cid in zip(palette, worst):
        precision, recall, _ = precision_recall_curve((labels == cid).astype(int), probs[:, cid])
        ax.plot(recall, precision, color=color, lw=1.6, label=f'{cid} · AP {ap[cid]:.3f}')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall (screening val): 5 lớp AP thấp nhất + micro")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_xlim(0.70, 1.001)
    ax.set_ylim(0.70, 1.001)

    ax = fig.add_subplot(gs[0, 1])
    order = np.argsort(ap)
    y = np.arange(len(order))
    values = ap[order]
    colors = [RED if value < 0.99 else (ORANGE if value < 0.998 else BLUE) for value in values]
    ax.hlines(y, 0.96, values, color=LIGHT, lw=1.0)
    for yi, value, color in zip(y, values, colors):
        ax.plot(value, yi, "o", ms=5, color=color, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{cid} · {_short(names[str(cid)]['vi'], 22)}" for cid in order], fontsize=6.8)
    ax.set_xlim(0.96, 1.001)
    ax.set_xlabel("Average Precision (AP) — trục bắt đầu 0,96")
    ax.set_title("AP từng lớp — 35/43 lớp đạt 1,000")
    for yi, value in zip(y, values):
        if value < 0.999:
            ax.annotate(f"{value:.3f}", (value, yi), textcoords="offset points", xytext=(-4, 6),
                        ha="right", fontsize=6.5)
    fig.suptitle("Chất lượng nhận diện từng lớp: micro-AP 0,997; thấp nhất là lớp 8 (0,973), 16 (0,986), 7 (0,991)", fontsize=12)
    _save(fig, "13_precision_recall.png")


# ---------------------------------------------------------------------------
# 14. Hiệu năng suy luận
# ---------------------------------------------------------------------------


def fig_inference_perf() -> None:
    rows = _read_csv(TABLES / "13_inference_benchmark.csv")
    model_only = [r for r in rows if r["scope"].startswith("model-only")]
    e2e = [r for r in rows if r["scope"].startswith("end-to-end")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for device, color, label in [("cpu", BLUE, "CPU (8 thread)"), ("cuda", RED, "GPU RTX 5070")]:
        part = sorted([r for r in model_only if r["device"] == device], key=lambda r: int(r["batch_size"]))
        batches = [int(r["batch_size"]) for r in part]
        latency = [float(r["latency_median_ms"]) for r in part]
        throughput = [float(r["throughput_img_s"]) for r in part]
        axes[0].plot(batches, latency, "o-", color=color, label=label)
        axes[1].plot(batches, throughput, "o-", color=color, label=label)
        e2e_part = next(r for r in e2e if r["device"] == device)
        axes[0].scatter([1], [float(e2e_part["latency_median_ms"])], marker="*", s=140, color=color, zorder=5)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xticks([1, 16, 32, 64])
    axes[0].set_xticklabels(["1", "16", "32", "64"])
    axes[0].set_xlabel("Batch size (log)")
    axes[0].set_ylabel("Latency trung vị (ms, log)")
    axes[0].set_title("Latency model-only (★ = end-to-end 1 ảnh)")
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks([1, 16, 32, 64])
    axes[1].set_xticklabels(["1", "16", "32", "64"])
    axes[1].set_xlabel("Batch size (log)")
    axes[1].set_ylabel("Ảnh/giây (log)")
    axes[1].set_title("Throughput model-only")
    axes[1].legend(fontsize=8)
    # Chú thích end-to-end
    for device, color, label, offset in [
        ("cpu", BLUE, "CPU", (8, -14)),
        ("cuda", RED, "GPU", (8, 10)),
    ]:
        e2e_part = next(r for r in e2e if r["device"] == device)
        axes[0].annotate(
            f'{label} e2e {float(e2e_part["latency_median_ms"]):.2f} ms',
            (1, float(e2e_part["latency_median_ms"])),
            textcoords="offset points", xytext=offset, fontsize=7.2, color=color,
        )
    fig.suptitle("Hiệu năng suy luận D2 (FP32, máy train; batch 1 GPU không nhanh hơn CPU do overhead)", fontsize=12)
    fig.tight_layout()
    _save(fig, "14_inference_perf.png")


# ---------------------------------------------------------------------------
# 15. Ví dụ dự đoán
# ---------------------------------------------------------------------------


def fig_prediction_examples() -> None:
    import pandas as pd

    payload = np.load(DATA / "screening_val_probs_d2_s42.npz")
    probs = payload["probs"]
    labels = payload["labels"]
    indices = payload["indices"]
    import pandas as pd

    meta = pd.read_csv(ROOT / "outputs/results/train_meta.csv")
    meta = meta[meta["usable_for_train"] == 1].reset_index(drop=True)
    names = _names()
    preds = pd.read_csv(RUNS / "v2_d2_s42" / "predictions_val.csv")
    track_by_index = dict(zip(preds["image_index"].astype(int), preds["track_key"].astype(str)))
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == labels

    def pick_distinct(pool, n=3, used=None):
        rng = np.random.default_rng(7)
        pool = list(rng.permutation(pool))
        chosen, tracks = [], used if used is not None else set()
        for pos in pool:
            track = track_by_index.get(int(indices[pos]), str(pos))
            if track in tracks:
                continue
            chosen.append(int(pos))
            tracks.add(track)
            if len(chosen) == n:
                break
        return chosen

    groups = [
        ("Đúng · confidence ≥ 0,95", np.where(correct & (conf >= 0.95))[0], GREEN),
        ("Đúng · confidence < 0,60", np.where(correct & (conf < 0.60))[0], ORANGE),
        ("Sai · confidence ≥ 0,90", np.where(~correct & (conf >= 0.90))[0], RED),
        ("Sai · confidence < 0,60", np.where(~correct & (conf < 0.60))[0], PURPLE),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 13.2))
    used_tracks: set[str] = set()
    for row, (title, pool, color) in enumerate(groups):
        chosen = pick_distinct(pool, used=used_tracks)
        for col in range(3):
            ax = axes.flat[row * 3 + col]
            ax.axis("off")
            if col >= len(chosen):
                continue
            pos = int(chosen[col])
            idx = int(indices[pos])
            meta_row = meta.iloc[idx]
            roi = (
                [meta_row["roi_x1"], meta_row["roi_y1"], meta_row["roi_x2"], meta_row["roi_y2"]]
                if int(meta_row["roi_valid"]) == 1
                else None
            )
            ax.imshow(_load_original_crop(meta_row["path"], roi), interpolation="bilinear")
            true_id = int(labels[pos])
            pred_id = int(pred[pos])
            ax.set_title(
                f'thật: {true_id} {_short(names[str(true_id)]["vi"], 18)}\n'
                f'đoán: {pred_id} {_short(names[str(pred_id)]["vi"], 18)} · {conf[pos]:.2f}',
                fontsize=7.6, color=color,
            )
            if col == 0:
                ax.text(-0.08, 0.5, title, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9.5, weight="bold", color=color)
    fig.suptitle(
        "Ví dụ dự đoán trên screening val (D2, seed 42): đúng/sai và confidence\n"
        "Ảnh hiển thị là crop gốc quanh biển (ROI); mô hình nhận ảnh 64×64",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "15_prediction_examples.png")


# ---------------------------------------------------------------------------
# 16. Chất lượng dữ liệu
# ---------------------------------------------------------------------------


def fig_data_quality() -> None:
    summary = _json(TABLES / "04_data_quality_summary.json")
    blur = np.load(DATA / "train_blur.npy")
    brightness = np.load(DATA / "train_brightness.npy")
    contrast = np.load(DATA / "train_contrast.npy")
    x = np.load(ROOT / "data" / "cache" / "X_train.npy", mmap_mode="r")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    ax = axes[0]
    ax.hist(blur, bins=80, color=BLUE, log=True)
    ax.axvline(summary["pool"]["thresholds"]["blur_p10"], color=RED, ls="--", lw=1.2,
               label=f'p10 = {summary["pool"]["thresholds"]["blur_p10"]:.5f}')
    ax.set_xlabel("Laplacian variance (độ nét; thấp = mờ)")
    ax.set_ylabel("Số ảnh (log)")
    ax.set_title(f'Độ nét · {summary["pool"]["n_blur_p10"]:,} ảnh dưới p10')
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.hist(brightness, bins=80, color=ORANGE, log=True)
    ax.axvline(0.10, color=RED, ls="--", lw=1.2, label="0,10")
    ax.axvline(0.80, color=GREEN, ls="--", lw=1.2, label="0,80")
    ax.set_xlabel("Độ sáng trung bình (0–1)")
    ax.set_ylabel("Số ảnh (log)")
    ax.set_title(f'Độ sáng · {summary["pool"]["n_dark_lt_0.10"]:,} ảnh tối < 0,10')
    ax.legend(fontsize=7.5)

    ax = axes[2]
    ax.hist(contrast, bins=80, color=GREEN, log=True)
    ax.axvline(summary["pool"]["thresholds"]["contrast_p10"], color=RED, ls="--", lw=1.2,
               label=f'p10 = {summary["pool"]["thresholds"]["contrast_p10"]:.3f}')
    ax.set_xlabel("Độ tương phản (std xám)")
    ax.set_ylabel("Số ảnh (log)")
    ax.set_title(f'Tương phản · {summary["pool"]["n_low_contrast_p10"]:,} ảnh dưới p10')
    ax.legend(fontsize=7.5)
    fig.suptitle("Chất lượng dữ liệu pool 39.201 ảnh: ảnh tối/mờ phổ biến nhưng không có ảnh hỏng", fontsize=12.5)
    fig.tight_layout()
    _save(fig, "16_data_quality.png")

    picks = [
        ("Tối nhất", int(np.argmin(brightness))),
        ("Sáng nhất", int(np.argmax(brightness))),
        ("Mờ nhất", int(np.argmin(blur))),
        ("Nét nhất", int(np.argmax(blur))),
        ("Tương phản thấp", int(np.argmin(contrast))),
        ("Trung vị", int(np.argsort(brightness)[len(brightness) // 2])),
    ]
    fig2, axes2 = plt.subplots(1, 6, figsize=(14, 3.1))
    for ax, (label, idx) in zip(axes2, picks):
        ax.imshow(np.asarray(x[idx]), interpolation="bilinear")
        ax.set_title(f"{label}\nbr {brightness[idx]:.2f} · bl {blur[idx]:.4f}", fontsize=7.5)
        ax.axis("off")
    fig2.suptitle("Ví dụ các cực trị chất lượng ảnh trong pool", fontsize=11)
    fig2.tight_layout()
    _save(fig2, "16b_data_quality_examples.png")


# ---------------------------------------------------------------------------
# 17. Lưới ảnh mẫu 43 lớp
# ---------------------------------------------------------------------------


def fig_class_samples() -> None:
    import pandas as pd

    y = np.load(ROOT / "data" / "cache" / "y_train.npy")
    blur = np.load(DATA / "train_blur.npy")
    brightness = np.load(DATA / "train_brightness.npy")
    names = _names()
    meta = pd.read_csv(ROOT / "outputs/results/train_meta.csv")
    meta = meta[meta["usable_for_train"] == 1].reset_index(drop=True)

    fig, axes = plt.subplots(7, 7, figsize=(14, 14))
    for cid in range(NUM_CLASSES):
        mask = np.where(y == cid)[0]
        med_b = np.median(brightness[mask])
        med_blur = np.median(blur[mask])
        candidates = mask[blur[mask] >= med_blur]
        if len(candidates) == 0:
            candidates = mask
        idx = int(candidates[np.argmin(np.abs(brightness[candidates] - med_b))])
        row = meta.iloc[idx]
        roi = (
            [row["roi_x1"], row["roi_y1"], row["roi_x2"], row["roi_y2"]]
            if int(row["roi_valid"]) == 1
            else None
        )
        ax = axes[cid // 7, cid % 7]
        ax.imshow(_load_original_crop(row["path"], roi), interpolation="bilinear")
        ax.set_title(f'{cid}: {_short(names[str(cid)]["vi"], 26)}', fontsize=7)
        ax.axis("off")
    for cid in range(NUM_CLASSES, 49):
        axes[cid // 7, cid % 7].axis("off")
    fig.suptitle("Mỗi lớp một ảnh mẫu (chọn ảnh có độ nét ≥ trung vị lớp và độ sáng gần trung vị)", fontsize=13)
    fig.text(0.5, 0.005, "Ảnh hiển thị là crop gốc quanh biển (ROI); mô hình nhận ảnh 64×64.",
             ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    _save(fig, "17_class_samples.png")


# ---------------------------------------------------------------------------
# 18. Contact sheet rà soát lỗi
# ---------------------------------------------------------------------------


def fig_error_contact_sheets() -> None:
    import pandas as pd

    rows = pd.read_csv(TABLES / "17b_manual_error_review_diverse.csv")
    x = np.load(ROOT / "data" / "cache" / "X_train.npy", mmap_mode="r")
    per_sheet = 30
    n_sheets = math.ceil(len(rows) / per_sheet)
    for sheet_idx in range(n_sheets):
        part = rows.iloc[sheet_idx * per_sheet : (sheet_idx + 1) * per_sheet]
        fig, axes = plt.subplots(5, 6, figsize=(15, 13.5))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, row) in zip(axes.ravel(), part.iterrows()):
            idx = int(row["image_index"])
            ax.imshow(np.asarray(x[idx]), interpolation="bilinear")
            ax.set_title(
                f'#{int(row["review_id"])} · {int(row["true_id"])}→{int(row["pred_id"])} · {row["confidence"]:.2f}\n'
                f'track {row["track_id"]} · {row["auto_category"]}',
                fontsize=6.6,
            )
            ax.axis("off")
        fig.suptitle(
            f"Phiếu rà soát lỗi D2 — sheet {sheet_idx + 1}/{n_sheets} "
            f"(mỗi track tối đa 2 ảnh; điền vào 17b_manual_error_review_diverse.csv)",
            fontsize=12, y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        _save(fig, f"contact_sheets/error_sheet_{sheet_idx + 1:02d}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


FIGURES = {
    "1": fig_class_distribution,
    "2": fig_split_diagram,
    "3": fig_architecture,
    "4": fig_learning_curves,
    "5": fig_model_comparison,
    "6": fig_confusion,
    "7": fig_per_class_f1,
    "8": fig_factor_effects,
    "9": fig_oof_fold_seed,
    "10": fig_robustness,
    "11": fig_gradcam,
    "12": fig_calibration,
    "13": fig_precision_recall,
    "14": fig_inference_perf,
    "15": fig_prediction_examples,
    "16": fig_data_quality,
    "17": fig_class_samples,
    "18": fig_error_contact_sheets,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all", help="all hoặc danh sách số, ví dụ 1,2,3")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    FIG.mkdir(parents=True, exist_ok=True)
    if args.only == "all":
        keys = list(FIGURES)
    else:
        keys = [key.strip() for key in args.only.split(",") if key.strip()]
    for key in keys:
        if key not in FIGURES:
            raise SystemExit(f"không có hình {key}")
        FIGURES[key]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
