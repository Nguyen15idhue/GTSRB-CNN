"""Huấn luyện một run v2 (screening hoặc fold) — Giai đoạn 1–3.

Khác ``src/train.py`` ở chỗ: đọc split v2 (lockbox/dev/screening/fold), mean/std
theo đúng split, ghi ``config_sha256``/``split_sha256``, lưu
``predictions_val.csv`` kèm ``image_index``/``track_key`` để dựng OOF.

Cách dùng::

    python -m src.train_v2 --config configs/v2/f00.yaml --seed 42
    python -m src.train_v2 --config configs/v2/f15.yaml --seed 42 --fold 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .config import (
    CLASS_NAMES_PATH,
    NUM_CLASSES,
    RUNS_DIR,
    collect_env,
    get_device,
    git_commit,
    load_yaml,
    set_seed,
)
from .data_v2 import (
    get_indices,
    load_manifest,
    make_loaders_v2,
    manifest_sha256,
    split_sha256,
    track_map,
)
from .models import build_model

LOG = logging.getLogger(__name__)

METRIC_COLUMNS = (
    "epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
    "val_macro_f1", "epoch_time_s", "gpu_mem_mb",
)


def sha256_file(path: Path) -> str:
    """SHA-256 của file cấu hình (ghi vào checkpoint/meta)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_split(cfg: Mapping[str, Any], fold: int | None = None) -> tuple[str, int | None]:
    """Trả ``(split_name, fold)`` từ config (screening | dev_all | fold)."""
    data = cfg["data"]
    split = str(data.get("split", "screening"))
    used_fold = data.get("fold") if fold is None else fold
    if split in ("fold", "dev_fold"):
        if used_fold is None:
            raise ValueError("data.split='fold' nhưng thiếu fold")
        return f"fold_{int(used_fold)}", int(used_fold)
    if split in ("screening", "dev_all", "lockbox"):
        return split, None
    raise ValueError(f"data.split v2 không hợp lệ: {split!r}")


def run_id_v2(cfg: Mapping[str, Any], seed: int, fold: int | None = None) -> str:
    """``v2_<config_id>_s<seed>[_f<fold>]``."""
    name = f"v2_{cfg['experiment']['id']}_s{int(seed)}"
    return name if fold is None else f"{name}_f{int(fold)}"


def build_criterion(cfg: Mapping[str, Any], labels: np.ndarray, device: torch.device) -> nn.Module:
    """CrossEntropyLoss + class weight (none | inv_sqrt | inv_freq)."""
    mode = str(cfg["train"].get("class_weight", "none"))
    weights = None
    if mode in ("inv_sqrt", "inv_sqrt_freq", "inv_freq"):
        counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
        raw = 1.0 / np.sqrt(np.maximum(counts, 1.0)) if "sqrt" in mode else 1.0 / np.maximum(counts, 1.0)
        raw = raw * (len(raw) / raw.sum())
        weights = torch.tensor(raw, dtype=torch.float32, device=device)
        LOG.info("Class weight %s: min=%.3f max=%.3f", mode, float(raw.min()), float(raw.max()))
    elif mode != "none":
        raise ValueError(f"class_weight không hợp lệ: {mode!r}")
    return nn.CrossEntropyLoss(
        weight=weights, label_smoothing=float(cfg["train"].get("label_smoothing", 0.0))
    )


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp) -> tuple[float, float]:
    """Một epoch train; trả (loss, accuracy)."""
    model.train()
    total, correct, seen = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(targets)
        correct += int((logits.argmax(1) == targets).sum())
        seen += len(targets)
    return total / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device) -> tuple[float, float, float]:
    """Đánh giá val: (loss, accuracy, macro-F1/43)."""
    from sklearn.metrics import f1_score

    model.eval()
    total, seen = 0.0, 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(inputs)
        total += float(criterion(logits, targets).item()) * len(targets)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
        seen += len(targets)
    accuracy = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))
    return total / max(seen, 1), accuracy, macro_f1


@torch.no_grad()
def predict_val(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả (y_true, y_pred, y_prob) trên val."""
    model.eval()
    truths, preds, probs = [], [], []
    for inputs, targets in loader:
        probability = torch.softmax(model(inputs.to(device)).float(), dim=1).cpu().numpy()
        probs.append(probability)
        preds.append(probability.argmax(1))
        truths.append(targets.numpy())
    return np.concatenate(truths), np.concatenate(preds), np.concatenate(probs)


def plot_curves(rows: Sequence[Mapping[str, Any]], out_path: Path, title: str) -> None:
    """3 panel: loss, accuracy, macro-F1 (bỏ panel val nếu run không có val)."""
    epochs = [int(r["epoch"]) for r in rows]

    def values(key: str) -> list[float]:
        return [float(r[key]) if str(r[key]).strip() not in ("", "None") else float("nan") for r in rows]

    has_val = any(np.isfinite(values("val_acc")))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    axes[0].plot(epochs, values("train_loss"), label="train")
    if has_val:
        axes[0].plot(epochs, values("val_loss"), label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch")
    axes[1].plot(epochs, values("train_acc"), label="train")
    if has_val:
        axes[1].plot(epochs, values("val_acc"), label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1.02)
    axes[2].plot(epochs, values("val_macro_f1"), color="tab:green", label="val macro-F1")
    axes[2].set_title("Val macro-F1/43"); axes[2].set_xlabel("epoch")
    if has_val:
        axes[2].set_ylim(0, 1.02)
    else:
        axes[2].axis("off")
    for ax in axes[:2]:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    if has_val:
        axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_checkpoint(
    out_dir: Path,
    model: nn.Module,
    resolved: Mapping[str, Any],
    epoch: int,
    val_macro_f1: float,
    config_sha: str,
    split_sha: str,
) -> None:
    """Lưu ``best.pt`` dạng dict thuần (không pickle model)."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dict(resolved),
            "class_names": json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8")),
            "epoch": int(epoch),
            "val_macro_f1": float(val_macro_f1),
            "git_commit": git_commit(short=False),
            "torch_version": str(torch.__version__),
            "config_sha256": config_sha,
            "split_sha256": split_sha,
        },
        out_dir / "best.pt",
    )


def run_training_v2(
    config_path: Path | str,
    seed: int | None = None,
    fold: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Huấn luyện một run v2 và ghi đầy đủ artifact."""
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    used_seed = int(cfg["experiment"].get("seed", 42) if seed is None else seed)
    split_name, used_fold = resolve_split(cfg, fold)
    # Ghi fold đã dùng vào cfg để make_loaders_v2 đọc đúng fold (bug 3.1 đã sửa).
    cfg["data"]["fold"] = used_fold
    device = device or get_device()
    name = run_id_v2(cfg, used_seed, used_fold)
    out_dir = RUNS_DIR / name
    if (out_dir / "meta.json").is_file():
        LOG.info("Run %s đã có meta.json — bỏ qua", name)
        return json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(used_seed)
    manifest = load_manifest()
    loaders = make_loaders_v2(cfg, split_name, used_seed)
    model, params = build_model(cfg)
    model.to(device)
    train_cfg = cfg["train"]
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    indices = loaders["indices"]
    has_val = len(indices["val"]) > 0
    train_labels = loaders["train"].dataset.labels[indices["train"]]
    criterion = build_criterion(cfg, train_labels, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0))
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["max_epochs"]))
        if train_cfg.get("scheduler", "cosine") == "cosine" else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config_sha = sha256_file(config_path)
    split_sha = split_sha256(split_name, manifest)
    resolved = json.loads(json.dumps(cfg))
    resolved["_resolved"] = {
        "seed": used_seed, "split_name": split_name, "fold": used_fold,
        "mean": loaders["meta"]["mean"], "std": loaders["meta"]["std"],
        "config_sha256": config_sha, "split_sha256": split_sha,
        "manifest_sha256": manifest_sha256(), "git_commit": git_commit(short=False),
    }
    (out_dir / "config.yaml").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "env.json").write_text(json.dumps(collect_env(), ensure_ascii=False, indent=2), encoding="utf-8")

    LOG.info(
        "Run %s | split %s | train %d / val %d | batch %d | aug %s | device %s",
        name, split_name, loaders["meta"]["n_train"], loaders["meta"]["n_val"],
        loaders["meta"]["batch_size"], loaders["meta"]["augmentation_profile"], device,
    )

    rows: list[dict[str, Any]] = []
    best_f1, best_epoch, best_row = -1.0, 0, {}
    bad, started = 0, time.perf_counter()
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_COLUMNS)
        writer.writeheader(); fh.flush()
        for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            lr_used = float(optimizer.param_groups[0]["lr"])
            epoch_started = time.perf_counter()
            train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, scaler, device, use_amp)
            if scheduler is not None:
                scheduler.step()
            if has_val:
                val_loss, val_acc, val_f1 = evaluate_epoch(model, loaders["val"], criterion, device)
            else:
                # Final train trên dev_all: không có val; không early stop, đánh giá sau trên lockbox.
                val_loss = val_acc = val_f1 = float("nan")
            gpu_mem = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1) if device.type == "cuda" else 0.0
            row = {
                "epoch": epoch, "lr": f"{lr_used:.6f}", "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": "" if not has_val else round(val_loss, 6),
                "val_acc": "" if not has_val else round(val_acc, 6),
                "val_macro_f1": "" if not has_val else round(val_f1, 6),
                "epoch_time_s": round(time.perf_counter() - epoch_started, 2), "gpu_mem_mb": gpu_mem,
            }
            writer.writerow(row); fh.flush(); rows.append(row)
            LOG.info(
                "epoch %02d | lr %.2e | train %.4f/%.4f | val %s/%s f1 %s | %.1fs",
                epoch, lr_used, train_loss, train_acc,
                "–" if not has_val else f"{val_loss:.4f}", "–" if not has_val else f"{val_acc:.4f}",
                "–" if not has_val else f"{val_f1:.4f}", row["epoch_time_s"],
            )
            if has_val:
                if val_f1 > best_f1:
                    best_f1, best_epoch, best_row = val_f1, epoch, dict(row)
                    bad = 0
                    _save_checkpoint(out_dir, model, resolved, epoch, float(val_f1), config_sha, split_sha)
                else:
                    bad += 1
                    if bad >= int(train_cfg["patience"]):
                        LOG.info("Early stopping tại epoch %d (patience=%d)", epoch, bad)
                        break
            else:
                # Lưu checkpoint mỗi epoch; best.pt cuối cùng = epoch cuối.
                best_epoch, best_row = epoch, dict(row)
                _save_checkpoint(out_dir, model, resolved, epoch, 0.0, config_sha, split_sha)

    train_time = time.perf_counter() - started
    # Nạp lại checkpoint tốt nhất để dự đoán val (OOF dùng file này).
    checkpoint = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    if has_val:
        y_true, y_pred, y_prob = predict_val(model, loaders["val"], device)
        mapping = track_map()
        val_indices = list(indices["val"])
        with (out_dir / "predictions_val.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image_index", "track_key", "true_label", "pred_label", "prob_true", "prob_pred", "fold"])
            for pos, image_index in enumerate(val_indices):
                true_id, pred_id = int(y_true[pos]), int(y_pred[pos])
                writer.writerow([
                    int(image_index), mapping[int(image_index)]["track_key"], true_id, pred_id,
                    f"{float(y_prob[pos, true_id]):.6f}", f"{float(y_prob[pos, pred_id]):.6f}",
                    "" if used_fold is None else int(used_fold),
                ])
        from sklearn.metrics import accuracy_score, f1_score

        val_macro = float(f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))
        val_acc = float(accuracy_score(y_true, y_pred))
    else:
        with (out_dir / "predictions_val.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image_index", "track_key", "true_label", "pred_label", "prob_true", "prob_pred", "fold"])
        val_macro = None
        val_acc = None

    # Latency model-only: batch 1, 100 lần, median.
    latency_ms = _latency_ms(model, int(cfg["data"]["image_size"]), device)
    plot_curves(rows, out_dir / "curves.png", f"{name} (best epoch {best_epoch})")
    meta = {
        "run_id": name,
        "protocol": "v2_clean",
        "block": cfg["experiment"].get("block", ""),
        "config_id": cfg["experiment"]["id"],
        "seed": used_seed,
        "fold": used_fold,
        "split_name": split_name,
        "model": cfg["model"].get("name", "smallcnn"),
        "params": int(params),
        "augmentation_profile": loaders["meta"]["augmentation_profile"],
        "epochs_run": len(rows),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": None if not has_val else round(best_f1, 6),
        "best_val_acc": None if not has_val else round(float(best_row.get("val_acc", 0.0)), 6),
        "best_train_acc": round(float(best_row.get("train_acc", 0.0)), 6),
        "train_val_acc_gap": None if not has_val else round(float(best_row.get("train_acc", 0.0)) - float(best_row.get("val_acc", 0.0)), 6),
        "val_acc": None if val_acc is None else round(val_acc, 6),
        "val_macro_f1": None if val_macro is None else round(val_macro, 6),
        "val_macro_f1_checkpoint": None if not has_val else round(best_f1, 6),
        "n_train": loaders["meta"]["n_train"],
        "n_val": loaders["meta"]["n_val"],
        "train_time_s": round(train_time, 2),
        "infer_ms_per_img": round(latency_ms, 4),
        "git_commit": git_commit(short=False),
        "config_sha256": config_sha,
        "split_sha256": split_sha,
        "manifest_sha256": manifest_sha256(),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "amp": use_amp,
        "num_workers": loaders["meta"]["num_workers"],
        "mean": loaders["meta"]["mean"],
        "std": loaders["meta"]["std"],
        "notes": cfg["experiment"].get("notes", ""),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Xong %s: best epoch %d | val macro-F1 %s | %.1f s",
             name, best_epoch, "– (dev_all)" if not has_val else f"{val_macro:.4f}", train_time)
    return meta


def _latency_ms(model: nn.Module, size: int, device: torch.device, repeats: int = 100) -> float:
    """Latency model-only batch 1 (median của 100 lần), đơn vị ms/ảnh."""
    model.eval()
    dummy = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện run v2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = parse_args(argv)
    run_training_v2(args.config, seed=args.seed, fold=args.fold, device=get_device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
