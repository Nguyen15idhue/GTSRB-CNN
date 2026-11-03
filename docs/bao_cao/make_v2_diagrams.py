"""Vẽ 3 sơ đồ cho báo cáo v2: split 3 tầng, pipeline, kiến trúc D2.

Chạy::

    python docs/bao_cao/make_v2_diagrams.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "outputs/figures/v2"
BLUE = "#2e86c1"
GREEN = "#27ae60"
ORANGE = "#e67e22"
RED = "#c0392b"
GRAY = "#7f8c8d"


def box(ax, x, y, w, h, text, color=BLUE, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                linewidth=1.2, edgecolor=color, facecolor=color + "22"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color=color))


def split_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
    box(ax, 0.2, 2.1, 2.2, 1.0, "Pool\n39.201 ảnh\n1.307 track / 1.183 sign-group", GRAY)
    box(ax, 3.0, 3.6, 2.6, 1.0, "LOCKBOX\n8.302 ảnh · 277 track\n238 sign-group · seed 2024", RED)
    box(ax, 3.0, 1.9, 2.6, 1.0, "DEV\n30.899 ảnh · 1.030 track\n945 sign-group", GREEN)
    box(ax, 6.2, 3.6, 2.5, 1.0, "Screening\n24.719 train / 6.180 val\nseed 42", BLUE)
    box(ax, 6.2, 1.9, 2.5, 1.0, "3-fold OOF\n19.949–21.120 train\n9.779–10.950 val", ORANGE)
    box(ax, 9.2, 3.6, 1.6, 1.0, "Factorial\nKiến trúc\nHyperparam\nAug", BLUE, 8)
    box(ax, 9.2, 1.9, 1.6, 1.0, "Chọn top 2\n+ control\nOOF 31k", ORANGE, 8)
    arrow(ax, 2.4, 2.9, 3.0, 4.0); arrow(ax, 2.4, 2.4, 3.0, 2.4)
    arrow(ax, 5.6, 4.1, 6.2, 4.1); arrow(ax, 5.6, 2.4, 6.2, 2.4)
    arrow(ax, 8.7, 4.1, 9.2, 4.1); arrow(ax, 8.7, 2.4, 9.2, 2.4)
    ax.text(5.5, 0.5, "Lockbox chỉ mở một lần sau freeze; official test 12.630 chỉ là benchmark phụ",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Hình 2.1. Split 3 tầng theo sign-group (số thật từ manifest.json)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/12_split_v2.png", dpi=160); plt.close(fig)


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.4); ax.axis("off")
    steps = [
        ("0\nSplit + audit\nC1–C12", 0.1, BLUE),
        ("1\nBaseline +\nfactorial 2⁴", 2.3, BLUE),
        ("2\nKiến trúc +\nhyperparam + aug", 4.5, BLUE),
        ("3\nOOF 3×3\nfreeze + lockbox", 6.7, ORANGE),
        ("4\nThống kê +\nhình + báo cáo", 8.9, GREEN),
    ]
    for text, x, color in steps:
        box(ax, x, 1.4, 1.8, 1.0, text, color, 9)
    for x in (1.9, 4.1, 6.3, 8.5):
        arrow(ax, x, 1.9, x + 0.4, 1.9)
    ax.text(5.5, 0.6, "Mỗi bước: artifact + test + commit; lockbox không dùng để chọn",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Hình 3.1. Pipeline thực nghiệm v2", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/13_pipeline_v2.png", dpi=160); plt.close(fig)


def architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.8); ax.axis("off")
    blocks = [
        ("Input\n3×64×64", 0.1, GRAY),
        ("Conv 3×3\n32 + BN\nReLU + pool\n32×32", 1.6, BLUE),
        ("Conv 3×3\n64 + BN\nReLU + pool\n16×16", 3.3, BLUE),
        ("Conv 3×3\n128 + BN\nReLU + pool\n8×8", 5.0, BLUE),
        ("Conv 3×3\n256 + BN\nReLU + pool\n4×4", 6.7, BLUE),
        ("Flatten\n4×4×256 = 4096", 8.4, ORANGE),
        ("FC 256\nReLU\nDropout 0,3", 9.4, GREEN),
        ("FC 43\nlogits", 10.3, RED),
    ]
    for text, x, color in blocks:
        width = 1.05 if x < 9.3 else 0.85
        box(ax, x, 1.2, width, 1.5, text, color, 7.5)
    for x in (1.15, 2.85, 4.55, 6.25, 7.95, 9.25, 10.15):
        arrow(ax, x, 1.95, x + 0.3, 1.95)
    ax.text(5.5, 0.5, "D2: 1.448.779 params · BN bật · dropout 0,3 · không GAP · đầu ra logits (không softmax)",
            ha="center", fontsize=8.5)
    ax.set_title("Hình 3.2. Kiến trúc D2 (winner v2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/14_architecture_d2.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    split_diagram(); pipeline_diagram(); architecture_diagram()
    print("Đã ghi 12_split_v2.png, 13_pipeline_v2.png, 14_architecture_d2.png")
