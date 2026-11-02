"""Hình cơ chế v2: learning curve, interaction, Pareto, per-class Δ, Grad-CAM,
reliability/ECE, corruption, confusion matrix, error taxonomy.

Tất cả hình ghi vào ``outputs/figures/v2/``; số liệu đi kèm ghi CSV/JSON trong
``outputs/results/v2/``.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import NUM_CLASSES, PROJECT_ROOT, RESULTS_DIR
from .data_v2 import get_indices, get_mean_std, load_images, track_map
from .stats_v2 import macro_f1_43, read_predictions

LOG = logging.getLogger(__name__)

FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "v2"
V2_RESULTS = RESULTS_DIR / "v2"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
CONFIG_DIR = PROJECT_ROOT / "configs" / "v2"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    LOG.info("Đã ghi %s", path)
    return path


def learning_curves(runs: Mapping[str, str]) -> Path:
    """4 panel loss/acc/macro-F1 cho các run đại diện (A/B/C)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, run_id in runs.items():
        path = RUNS_DIR / run_id / "metrics.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        epochs = [int(r["epoch"]) for r in rows]
        axes[0].plot(epochs, [float(r["val_loss"]) for r in rows], label=f"{label} val")
        axes[1].plot(epochs, [float(r["val_acc"]) for r in rows], label=f"{label} val")
        axes[2].plot(epochs, [float(r["val_macro_f1"]) for r in rows], label=label)
    axes[0].set_title("Val loss"); axes[1].set_title("Val accuracy"); axes[2].set_title("Val macro-F1/43")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Learning curves trên screening val (3 seed, seed 42)")
    return _save(fig, "01_learning_curves.png")


def interaction_plot(factorial: Mapping[str, Any]) -> Path:
    """Interaction BN×Aug và DO×WD từ hiệu ứng factorial."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["BN×Aug", "DO×WD"]
    values = [factorial["bn_x_aug"]["effect_mean"], factorial["do_x_wd"]["effect_mean"]]
    lows = [factorial["bn_x_aug"]["ci_low"], factorial["do_x_wd"]["ci_low"]]
    highs = [factorial["bn_x_aug"]["ci_high"], factorial["do_x_wd"]["ci_high"]]
    axes[0].bar(labels, values, color=["#2e86c1", "#e67e22"])
    axes[0].errorbar(labels, values, yerr=[np.array(values) - np.array(lows), np.array(highs) - np.array(values)],
                     fmt="none", ecolor="black", capsize=4)
    axes[0].axhline(0, color="gray", lw=1)
    axes[0].set_ylabel("Δ macro-F1 (điểm %)")
    axes[0].set_title("Interaction (điểm %)")
    main = ["bn", "do", "wd", "aug"]
    vals = [factorial[k]["effect_mean"] * 100 for k in main]
    err_low = [vals[i] - factorial[k]["ci_low"] * 100 for i, k in enumerate(main)]
    err_high = [factorial[k]["ci_high"] * 100 - vals[i] for i, k in enumerate(main)]
    axes[1].bar(main, vals, color="#27ae60")
    axes[1].errorbar(main, vals, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=4)
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].set_title("Main effect (điểm %)")
    return _save(fig, "02_factorial_effects.png")


def pareto_plot(table: Sequence[Mapping[str, Any]]) -> Path:
    """Params vs macro-F1, đánh dấu Pareto frontier."""
    points = [(int(r["params"]), float(r["macro_f1_mean"]), r["config_id"]) for r in table]
    points.sort()
    frontier, best = [], -1.0
    for params, f1, config_id in points:
        if f1 > best:
            frontier.append((params, f1, config_id))
            best = f1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter([p[0] for p in points], [p[1] for p in points], color="#2e86c1")
    for params, f1, config_id in points:
        ax.annotate(config_id, (params, f1), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.plot([p[0] for p in frontier], [p[1] for p in frontier], color="#c0392b", marker="o", label="Pareto")
    ax.set_xlabel("Params"); ax.set_ylabel("Macro-F1/43 (mean 3 seed)")
    ax.set_title("Pareto: params vs macro-F1 (screening val)")
    ax.grid(alpha=0.3); ax.legend()
    return _save(fig, "03_pareto_params_f1.png")


def per_class_heatmap(config_ids: Sequence[str], reference: str = "f15", seed: int = 42) -> Path:
    """Δ F1 từng lớp so với reference, hàng = config."""
    mapping = track_map()
    ref_path = RUNS_DIR / f"v2_{reference}_s{seed}" / "predictions_val.csv"
    index, truth, ref_pred = read_predictions(ref_path)
    from sklearn.metrics import f1_score

    support = np.bincount(truth, minlength=NUM_CLASSES)
    ref_f1 = f1_score(truth, ref_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    rows = []
    for config_id in config_ids:
        path = RUNS_DIR / f"v2_{config_id}_s{seed}" / "predictions_val.csv"
        if not path.is_file():
            continue
        idx, truth_i, pred = read_predictions(path)
        if not np.array_equal(idx, index):
            order = {int(i): p for i, p in zip(idx, pred)}
            pred = np.array([order[int(i)] for i in index])
        f1 = f1_score(truth, pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        rows.append((config_id, f1 - ref_f1))
    fig, ax = plt.subplots(figsize=(14, max(3, 0.4 * len(rows) + 2)))
    matrix = np.array([r[1] for r in rows])
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.set_xticks(range(NUM_CLASSES), [str(c) for c in range(NUM_CLASSES)], fontsize=6, rotation=90)
    fig.colorbar(image, ax=ax, label="Δ F1 so với " + reference)
    ax.set_title(f"Per-class ΔF1 so với {reference} (seed 42, screening val)")
    with (V2_RESULTS / "per_class_delta.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config_id", "class_id", "delta_f1", "support"])
        for config_id, delta in rows:
            for c in range(NUM_CLASSES):
                writer.writerow([config_id, c, round(float(delta[c]), 6), int(support[c])])
    return _save(fig, "04_per_class_delta.png")


def reliability_plot(config_ids: Sequence[str], seed: int = 42, bins: int = 15) -> Path:
    """Reliability diagram + ECE cho các config (dùng prob_pred trên screening val)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ece_table = []
    for config_id in config_ids:
        path = RUNS_DIR / f"v2_{config_id}_s{seed}" / "predictions_val.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        conf = np.array([float(r["prob_pred"]) for r in rows])
        correct = np.array([int(r["true_label"]) == int(r["pred_label"]) for r in rows], dtype=float)
        edges = np.linspace(0, 1, bins + 1)
        xs, ys, ece = [], [], 0.0
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (conf > low) & (conf <= high)
            if mask.sum() == 0:
                continue
            xs.append(float(conf[mask].mean()))
            ys.append(float(correct[mask].mean()))
            ece += float(mask.mean()) * abs(ys[-1] - xs[-1])
        ax.plot(xs, ys, marker="o", ms=3, label=f"{config_id} (ECE {ece:.3f})")
        ece_table.append({"config_id": config_id, "ece_15bin": ece})
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Reliability diagram (screening val, seed 42)")
    (V2_RESULTS / "calibration.csv").write_text(
        "\n".join(["config_id,ece_15bin"] + [f"{r['config_id']},{r['ece_15bin']:.6f}" for r in ece_table]),
        encoding="utf-8",
    )
    return _save(fig, "05_reliability.png")


def corruption_curves(path: Path | None = None) -> Path:
    """6 panel accuracy vs severity cho các config có trong corruption.csv."""
    path = path or (V2_RESULTS / "corruption.csv")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    kinds = ["blur", "noise", "brightness", "contrast", "jpeg", "occlusion"]
    configs = sorted({r["config_id"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, kind in zip(axes.ravel(), kinds):
        for config_id in configs:
            subset = sorted([r for r in rows if r["config_id"] == config_id and r["corruption"] == kind],
                            key=lambda r: int(r["severity"]))
            ax.plot([int(r["severity"]) for r in subset], [float(r["accuracy"]) for r in subset],
                    marker="o", label=config_id)
        ax.set_title(kind); ax.set_xlabel("severity"); ax.set_ylabel("accuracy"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Corruption robustness trên screening val (seed 42)")
    return _save(fig, "06_corruption_curves.png")


def confusion_from_predictions(config_id: str, seed: int = 42, title: str = "") -> Path:
    """Confusion matrix chuẩn hóa theo hàng từ predictions_val.csv."""
    from sklearn.metrics import confusion_matrix

    path = RUNS_DIR / f"v2_{config_id}_s{seed}" / "predictions_val.csv"
    _, truth, pred = read_predictions(path)
    matrix = confusion_matrix(truth, pred, labels=list(range(NUM_CLASSES)))
    row = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row, out=np.zeros_like(matrix, dtype=float), where=row > 0)
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="tỷ lệ theo hàng")
    ax.set_xticks(range(NUM_CLASSES), range(NUM_CLASSES), fontsize=6, rotation=90)
    ax.set_yticks(range(NUM_CLASSES), range(NUM_CLASSES), fontsize=6)
    ax.set_title(title or f"Confusion matrix chuẩn hóa hàng — {config_id} s{seed}")
    return _save(fig, f"07_confusion_{config_id}_s{seed}.png")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------


def _gradcam(model, image_tensor, device):
    """Grad-CAM trên conv cuối; trả heatmap 0–1 và lớp dự đoán."""
    import torch

    activations, gradients = {}, {}

    def forward_hook(module, inputs, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    target = model.features[-1]
    handle_f = target.register_forward_hook(forward_hook)
    handle_b = target.register_full_backward_hook(backward_hook)
    model.zero_grad(set_to_none=True)
    logits = model(image_tensor.to(device))
    pred = int(logits.argmax(1).item())
    logits[0, pred].backward()
    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
    handle_f.remove(); handle_b.remove()
    return cam, pred


def gradcam_grid(config_id: str = "f15", compare_id: str = "d3", seed: int = 42, n_classes: int = 4) -> Path:
    """12 ảnh × 2 model (B vs C): đúng tự tin, sai, đúng nhưng confidence thấp."""
    import torch

    from .config import get_device
    from .models import build_model

    device = get_device()
    payload = load_images(64)
    indices = get_indices("screening")["val"]
    images = payload["x"][indices]
    labels = payload["y"][indices]
    mean, std = get_mean_std("screening")
    tensor = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    tensor = (tensor - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)

    models = {}
    for name in (config_id, compare_id):
        ckpt = torch.load(RUNS_DIR / f"v2_{name}_s{seed}" / "best.pt", map_location="cpu", weights_only=True)
        model, _ = build_model(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        models[name] = model.to(device).eval()

    with torch.no_grad():
        probs = torch.softmax(models[config_id](tensor.to(device)).float(), dim=1).cpu().numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    selected: list[int] = []
    classes = sorted({int(c) for c in labels})[:n_classes]
    for class_id in classes:
        idx = np.flatnonzero((labels == class_id) & (pred == labels))
        if len(idx):
            selected.append(int(idx[np.argmax(conf[idx])]))
        wrong = np.flatnonzero((labels == class_id) & (pred != labels))
        if len(wrong):
            selected.append(int(wrong[0]))
        low = np.flatnonzero((labels == class_id) & (pred == labels))
        if len(low):
            selected.append(int(low[np.argmin(conf[low])]))
    selected = selected[:12]

    fig, axes = plt.subplots(3, len(selected), figsize=(1.9 * len(selected), 6.4))
    if len(selected) == 1:
        axes = axes.reshape(3, 1)
    for col, pos in enumerate(selected):
        array = images[pos]
        axes[0, col].imshow(array)
        axes[0, col].set_title(f"y={int(labels[pos])}", fontsize=7)
        axes[0, col].axis("off")
        for row, name in enumerate((config_id, compare_id), start=1):
            cam, p = _gradcam(models[name], tensor[pos:pos + 1], device)
            axes[row, col].imshow(array)
            axes[row, col].imshow(cam, cmap="jet", alpha=0.45)
            axes[row, col].set_title(f"{name}: pred={p}", fontsize=6)
            axes[row, col].axis("off")
    fig.suptitle("Grad-CAM 12 ảnh: CNN-B (f15) vs CNN-C (d3), seed 42")
    return _save(fig, "08_gradcam_grid.png")


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


def error_taxonomy(config_id: str = "f15", seed: int = 42) -> Path:
    """Phân nhóm lỗi theo heuristic + lưới 12 mẫu (có ghi rõ là heuristic)."""
    from skimage.color import rgb2gray
    from skimage.filters import laplace

    payload = load_images(64)
    indices = get_indices("screening")["val"]
    images = payload["x"][indices]
    path = RUNS_DIR / f"v2_{config_id}_s{seed}" / "predictions_val.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    truth = np.array([int(r["true_label"]) for r in rows])
    pred = np.array([int(r["pred_label"]) for r in rows])
    conf = np.array([float(r["prob_pred"]) for r in rows])
    wrong = np.flatnonzero(truth != pred)
    blur = np.array([float(np.var(laplace(rgb2gray(image.astype(np.float32) / 255.0)))) for image in images])
    brightness = images.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
    categories: list[str] = []
    for i in wrong:
        if blur[i] < 20:
            categories.append("mờ")
        elif brightness[i] < 0.15:
            categories.append("tối")
        elif brightness[i] > 0.85:
            categories.append("chói")
        elif conf[i] < 0.5:
            categories.append("nhầm cặp giống nhau / confidence thấp")
        else:
            categories.append("góc nhìn lạ / nhãn khó")
    counts = {name: categories.count(name) for name in sorted(set(categories))}
    chosen = wrong[:12]
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for ax, i in zip(axes.ravel(), chosen):
        ax.imshow(images[i]); ax.axis("off")
        ax.set_title(f"y={truth[i]} → {pred[i]}\nconf={conf[i]:.2f}", fontsize=7)
    for ax in axes.ravel()[len(chosen):]:
        ax.axis("off")
    fig.suptitle(f"12 mẫu lỗi — {config_id} seed {seed} (phân nhóm heuristic)")
    with (V2_RESULTS / "error_taxonomy.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_index", "true", "pred", "confidence", "blur_laplacian", "brightness", "category"])
        for i, cat in zip(wrong, categories):
            writer.writerow([int(indices[i]), int(truth[i]), int(pred[i]), round(float(conf[i]), 6),
                             round(float(blur[i]), 3), round(float(brightness[i]), 4), cat])
    (V2_RESULTS / "error_taxonomy_summary.json").write_text(
        json.dumps({"config_id": config_id, "seed": seed, "n_errors": int(len(wrong)),
                    "counts": counts, "method": "heuristic trên blur/brightness/confidence; cần xem thủ công"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _save(fig, "09_error_samples.png")
