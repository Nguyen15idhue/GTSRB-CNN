"""Vòng huấn luyện, early stopping, logging và kiểm tra hạ tầng (Giai đoạn 3).

Một run tạo ra ``outputs/runs/<run_id>/`` gồm: ``config.yaml``, ``env.json``,
``metrics.csv`` (1 dòng/epoch, flush ngay), ``best.pt`` (theo val macro-F1),
``curves.png`` và ``meta.json``.

Cách dùng::

    python -m src.train --config configs/smoke.yaml
    python -m src.train --config configs/e3_cnnb_aug.yaml --seed 42
    python -m src.train --config configs/smoke.yaml --overfit-batch
    python -m src.train --config configs/smoke.yaml --determinism-check
"""

from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader

from .config import (
    CLASS_NAMES_PATH,
    DEFAULT_SEED,
    NUM_CLASSES,
    RESULTS_DIR,
    RUNS_DIR,
    collect_env,
    get_device,
    git_commit,
    load_config,
    run_id,
    set_seed,
)
from .data import load_split, make_loaders
from .models import build_model

LOG = logging.getLogger(__name__)

METRIC_COLUMNS = (
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "epoch_time_s",
    "gpu_mem_mb",
)
PHASE3_OVERFIT_PATH = RESULTS_DIR / "phase3_overfit.json"
PHASE3_DETERMINISM_PATH = RESULTS_DIR / "phase3_determinism.json"


# ---------------------------------------------------------------------------
# Thành phần huấn luyện
# ---------------------------------------------------------------------------


def build_criterion(cfg: Mapping[str, Any], labels: np.ndarray, device: torch.device) -> nn.Module:
    """CrossEntropyLoss (tuỳ chọn class weight ``1/sqrt(freq)`` cho E7)."""
    mode = cfg["train"].get("class_weight", "none")
    weights = None
    if mode == "inv_sqrt_freq":
        counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
        raw = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        raw = raw * (len(raw) / raw.sum())  # chuẩn hóa để trung bình trọng số = 1
        weights = torch.tensor(raw, dtype=torch.float32, device=device)
        LOG.info("Class weight 1/sqrt(freq): min=%.3f max=%.3f", float(raw.min()), float(raw.max()))
    elif mode != "none":
        raise ValueError(f"class_weight không hợp lệ: {mode!r}")
    return nn.CrossEntropyLoss(
        weight=weights, label_smoothing=float(cfg["train"].get("label_smoothing", 0.0))
    )


def build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """AdamW/Adam/SGD theo config (không hard-code hyperparameter)."""
    train_cfg = cfg["train"]
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    name = train_cfg["optimizer"]
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"optimizer không hợp lệ: {name!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any]):
    """CosineAnnealingLR theo ``max_epochs``; ``none`` thì không dùng scheduler."""
    if cfg["train"].get("scheduler") == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["train"]["max_epochs"])
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Một epoch train; trả ``(loss trung bình, accuracy)``."""
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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
        total_loss += float(loss.item()) * len(targets)
        correct += int((logits.argmax(dim=1) == targets).sum())
        seen += len(targets)
    return total_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Đánh giá trên val: ``(loss, accuracy, macro-F1)``; luôn ``eval()`` + ``no_grad``."""
    from sklearn.metrics import f1_score

    model.eval()
    if model.training:
        raise RuntimeError("model vẫn ở chế độ train khi đánh giá — sai hạ tầng")
    if torch.is_grad_enabled():
        raise RuntimeError("gradient đang bật khi đánh giá — sai hạ tầng (thiếu no_grad)")

    total_loss = 0.0
    seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        total_loss += float(criterion(logits, targets).item()) * len(targets)
        predictions = logits.argmax(dim=1)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        seen += len(targets)
    accuracy = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return total_loss / max(seen, 1), accuracy, macro_f1


def plot_curves(rows: Sequence[Mapping[str, Any]], out_path: Path, title: str) -> None:
    """Vẽ 3 subplot: loss, accuracy, macro-F1 theo epoch."""
    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], marker="o", label="train loss")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], marker="s", label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [r["train_acc"] for r in rows], marker="o", label="train acc")
    axes[1].plot(epochs, [r["val_acc"] for r in rows], marker="s", label="val acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Accuracy")
    axes[2].plot(
        epochs, [r["val_macro_f1"] for r in rows], marker="d", color="tab:green", label="val macro-F1"
    )
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("macro-F1")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("Val macro-F1 (tiêu chí chọn checkpoint)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: Mapping[str, Any],
    epoch: int,
    val_macro_f1: float,
) -> None:
    """Lưu checkpoint dạng dict (không pickle object model)."""
    with CLASS_NAMES_PATH.open(encoding="utf-8") as fh:
        class_names = json.load(fh)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dict(cfg),
            "class_names": class_names,
            "epoch": int(epoch),
            "val_macro_f1": float(val_macro_f1),
            "git_commit": git_commit(short=False),
            "torch_version": str(torch.__version__),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Một run huấn luyện
# ---------------------------------------------------------------------------


def run_training(
    cfg: Mapping[str, Any],
    device: torch.device | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Huấn luyện một cấu hình; ghi đầy đủ artifact vào thư mục run."""
    used_seed = int(cfg["experiment"].get("seed", DEFAULT_SEED) if seed is None else seed)
    device = device or get_device()
    name = run_id(cfg, used_seed)
    out_dir = RUNS_DIR / name
    if out_dir.exists():
        raise FileExistsError(
            f"Run đã tồn tại: {out_dir}. Quy ước: không ghi đè run cũ — "
            "đổi experiment.id hoặc seed."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(used_seed)
    loaders = make_loaders(cfg, seed=used_seed)
    model, params = build_model(cfg)
    model.to(device)
    train_cfg = cfg["train"]
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    criterion = build_criterion(cfg, loaders["train"].dataset.labels[load_split(cfg["data"]["split"])["train"]], device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    cfg_to_write = {k: v for k, v in dict(cfg).items() if not k.startswith("_")}
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        import yaml

        yaml.safe_dump(cfg_to_write, fh, allow_unicode=True, sort_keys=False)
    (out_dir / "env.json").write_text(
        json.dumps(collect_env(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    LOG.info(
        "Run %s | train %d / val %d | batch %d | amp=%s | device=%s",
        name,
        loaders["meta"]["n_train"],
        loaders["meta"]["n_val"],
        loaders["meta"]["batch_size"],
        use_amp,
        device,
    )

    rows: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_row: dict[str, Any] = {}
    bad_epochs = 0
    started = time.perf_counter()

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        fh.flush()
        for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            lr_used = float(optimizer.param_groups[0]["lr"])
            epoch_started = time.perf_counter()
            train_loss, train_acc = train_one_epoch(
                model, loaders["train"], criterion, optimizer, scaler, device, use_amp
            )
            if scheduler is not None:
                scheduler.step()
            val_loss, val_acc, val_f1 = evaluate(model, loaders["val"], criterion, device)
            epoch_time = time.perf_counter() - epoch_started
            gpu_mem = (
                round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
                if device.type == "cuda"
                else 0.0
            )
            row = {
                "epoch": epoch,
                "lr": f"{lr_used:.6f}",
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "val_macro_f1": round(val_f1, 6),
                "epoch_time_s": round(epoch_time, 2),
                "gpu_mem_mb": gpu_mem,
            }
            writer.writerow(row)
            fh.flush()  # mất điện vẫn còn dữ liệu
            rows.append(row)
            LOG.info(
                "epoch %02d | lr %.2e | train loss %.4f acc %.4f | val loss %.4f acc %.4f f1 %.4f | %.1fs",
                epoch,
                lr_used,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_f1,
                epoch_time,
            )
            if val_f1 > best_f1:
                best_f1, best_epoch, best_row = val_f1, epoch, dict(row)
                bad_epochs = 0
                _save_checkpoint(out_dir / "best.pt", model, cfg_to_write, epoch, val_f1)
            else:
                bad_epochs += 1
                if bad_epochs >= int(train_cfg["patience"]):
                    LOG.info("Early stopping tại epoch %d (patience=%d)", epoch, bad_epochs)
                    break

    train_time = time.perf_counter() - started
    plot_curves(rows, out_dir / "curves.png", f"{name} (best epoch {best_epoch})")
    meta = {
        "run_id": name,
        "experiment": cfg["experiment"],
        "model": cfg["model"]["name"],
        "params": params,
        "seed": used_seed,
        "split": cfg["data"]["split"],
        "epochs_run": len(rows),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_f1, 6),
        "best_val_acc": round(float(best_row.get("val_acc", 0.0)), 6),
        "best_train_acc": round(float(best_row.get("train_acc", 0.0)), 6),
        "train_val_acc_gap": round(
            float(best_row.get("train_acc", 0.0)) - float(best_row.get("val_acc", 0.0)), 6
        ),
        "train_time_s": round(train_time, 2),
        "epoch_time_mean_s": round(float(np.mean([r["epoch_time_s"] for r in rows])), 2),
        "params": params,
        "device": str(device),
        "amp": use_amp,
        "num_workers": loaders["meta"]["num_workers"],
        "git_commit": git_commit(short=False),
        "torch_version": torch.__version__,
        "notes": cfg["experiment"].get("notes", ""),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info(
        "Xong %s: best epoch %d, val acc %.4f, val macro-F1 %.4f, %.1f s",
        name,
        best_epoch,
        meta["best_val_acc"],
        best_f1,
        train_time,
    )
    return meta


# ---------------------------------------------------------------------------
# Test hạ tầng (Giai đoạn 3)
# ---------------------------------------------------------------------------


def run_overfit_test(cfg: Mapping[str, Any], device: torch.device | None = None, steps: int = 50) -> dict[str, Any]:
    """Overfit 1 batch 32 ảnh trong ``steps`` vòng: kỳ vọng loss → 0, acc = 100%."""
    device = device or get_device()
    set_seed(int(cfg["experiment"]["seed"]))
    cfg_overfit = json.loads(json.dumps(cfg))
    cfg_overfit["augmentation"]["enabled"] = False  # học thuộc batch, không augment
    loaders = make_loaders(cfg_overfit, seed=int(cfg["experiment"]["seed"]))
    inputs, targets = next(iter(loaders["train"]))
    inputs, targets = inputs[:32].to(device), targets[:32].to(device)

    model, params = build_model(cfg)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    model.train()
    history: list[float] = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        if step in (1, steps // 2, steps):
            LOG.info("overfit step %02d | loss %.5f", step, float(loss.item()))

    model.eval()
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1)
    accuracy = float((predictions == targets).float().mean())
    payload = {
        "batch_size": int(inputs.shape[0]),
        "steps": steps,
        "params": params,
        "loss_first": round(history[0], 6),
        "loss_mid": round(history[len(history) // 2], 6),
        "loss_last": round(history[-1], 6),
        "accuracy": round(accuracy, 6),
        "lr": float(cfg["train"]["lr"]),
        "passed_loss": history[-1] < 0.05,
        "passed_acc": accuracy == 1.0,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_OVERFIT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Overfit 1 batch: loss %.5f, acc %.3f → %s", history[-1], accuracy, PHASE3_OVERFIT_PATH)
    return payload


def run_determinism_check(
    cfg: Mapping[str, Any], device: torch.device | None = None, epochs: int = 3
) -> dict[str, Any]:
    """Chạy 2 lần cùng seed, cùng số epoch; so sánh loss/acc từng epoch."""
    device = device or get_device()
    seed = int(cfg["experiment"]["seed"])
    runs: list[list[dict[str, float]]] = []
    for attempt in (1, 2):
        set_seed(seed)
        loaders = make_loaders(cfg, seed=seed)
        model, _ = build_model(cfg)
        model.to(device)
        use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        per_epoch: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, loaders["train"], criterion, optimizer, scaler, device, use_amp
            )
            per_epoch.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
            LOG.info("determinism lần %d epoch %d: loss %.8f acc %.8f", attempt, epoch, train_loss, train_acc)
        runs.append(per_epoch)

    diffs = [
        abs(a["train_loss"] - b["train_loss"]) for a, b in zip(runs[0], runs[1])
    ]
    payload = {
        "seed": seed,
        "epochs": epochs,
        "attempt_1": [{k: round(v, 8) for k, v in row.items()} for row in runs[0]],
        "attempt_2": [{k: round(v, 8) for k, v in row.items()} for row in runs[1]],
        "max_abs_diff": round(max(diffs), 8),
        "threshold": 1e-4,
        "passed": max(diffs) < 1e-4,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_DETERMINISM_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("Determinism: lệch lớn nhất %.3e → %s", payload["max_abs_diff"], PHASE3_DETERMINISM_PATH)
    return payload


# ---------------------------------------------------------------------------
# Checklist hạ tầng (Giai đoạn 3)
# ---------------------------------------------------------------------------


class _ModeSpy(nn.Module):
    """Bọc model để ghi lại ``training`` và trạng thái grad tại mỗi lần forward."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.training_flags: list[bool] = []
        self.grad_flags: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        self.training_flags.append(bool(self.inner.training))
        self.grad_flags.append(bool(torch.is_grad_enabled()))
        return self.inner(x)


def verify_phase3(cfg_path: str = "configs/smoke.yaml", device: torch.device | None = None) -> list[dict[str, Any]]:
    """Chạy checklist 3.5 trên artifact của run smoke; ghi ``phase3_check.json``."""
    device = device or torch.device("cpu")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        LOG.info("%s %s — %s", "PASS" if passed else "FAIL", name, detail)

    cfg = load_config(cfg_path)
    name = run_id(cfg)
    run_dir = RUNS_DIR / name
    metrics_path = run_dir / "metrics.csv"

    artifacts = ["metrics.csv", "best.pt", "curves.png", "config.yaml", "env.json", "meta.json"]
    missing = [a for a in artifacts if not (run_dir / a).is_file()]
    add("artifact_run", not missing, f"{name}: thiếu {missing}" if missing else f"đủ {len(artifacts)} file")

    rows: list[dict[str, str]] = []
    header: list[str] = []
    if metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            rows = [dict(zip(header, r)) for r in reader if r]
    add(
        "metrics_schema",
        header == list(METRIC_COLUMNS) and bool(rows) and all(len(r) == len(METRIC_COLUMNS) for r in rows),
        f"{len(header)} cột, {len(rows)} dòng, khớp schema: {header == list(METRIC_COLUMNS)}",
    )
    add(
        "metrics_day_du",
        bool(rows) and rows[0]["epoch"] == "1" and all(all(str(v).strip() for v in r.values()) for r in rows),
        f"epoch đầu = {rows[0]['epoch'] if rows else 'NA'}, ô trống = 0"
        if rows and all(all(str(v).strip() for v in r.values()) for r in rows)
        else "có ô trống hoặc thiếu epoch 1",
    )
    f1_values = [float(r["val_macro_f1"]) for r in rows]
    add(
        "val_macro_f1_range",
        bool(f1_values) and all(0.0 <= v <= 1.0 for v in f1_values),
        f"min={min(f1_values):.4f} max={max(f1_values):.4f}",
    )
    first_three = [float(r["train_loss"]) for r in rows[:3]]
    decreasing = len(first_three) == 3 and first_three[0] > first_three[1] > first_three[2]
    add("train_loss_giam", decreasing, f"3 epoch đầu: {['%.4f' % v for v in first_three]}")

    ckpt_path = run_dir / "best.pt"
    checkpoint_ok = False
    checkpoint_detail = "không có best.pt"
    if ckpt_path.is_file():
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        required = {"state_dict", "config", "class_names", "epoch", "val_macro_f1", "git_commit", "torch_version"}
        model, _ = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 3, int(cfg["data"]["image_size"]), int(cfg["data"]["image_size"])))
        checkpoint_ok = required.issubset(checkpoint.keys()) and tuple(logits.shape) == (2, NUM_CLASSES)
        checkpoint_detail = (
            f"khóa đủ={required.issubset(checkpoint.keys())}, forward={tuple(logits.shape)}, "
            f"epoch={checkpoint['epoch']}, val_f1={checkpoint['val_macro_f1']:.4f}"
        )
    add("checkpoint_load", checkpoint_ok, checkpoint_detail)

    for key, path, label in (
        ("overfit_batch", PHASE3_OVERFIT_PATH, "overfit 1 batch"),
        ("determinism", PHASE3_DETERMINISM_PATH, "determinism"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        passed = bool(payload.get("passed_loss", payload.get("passed", False))) and bool(
            payload.get("passed_acc", payload.get("passed", False))
        )
        detail = (
            f"loss cuối {payload.get('loss_last', 'NA')}, acc {payload.get('accuracy', 'NA')}"
            if key == "overfit_batch"
            else f"lệch lớn nhất {payload.get('max_abs_diff', 'NA')} (ngưỡng 1e-4)"
        )
        add(key, passed, f"{label}: {detail}")

    curves = run_dir / "curves.png"
    add(
        "curves_png",
        curves.is_file() and curves.stat().st_size > 10_000,
        f"{curves.name} {curves.stat().st_size // 1024 if curves.is_file() else 0} KB",
    )

    summary_rows = collect_summary_rows()
    summary_ok = any(r["run_id"] == name for r in summary_rows)
    add("summary_csv", summary_ok, f"{len(summary_rows)} dòng trong summary.csv, có run {name}: {summary_ok}")

    cfg_aug = json.loads(json.dumps(cfg))
    cfg_aug["augmentation"]["enabled"] = True
    loaders = make_loaders(cfg_aug, seed=int(cfg["experiment"]["seed"]))
    add(
        "aug_chi_train",
        loaders["train"].dataset.transform is not None and loaders["val"].dataset.transform is None,
        "train có transform augmentation, val không có",
    )

    try:
        sample_inputs, sample_targets = next(iter(loaders["val"]))
        model, _ = build_model(cfg)
        spy = _ModeSpy(model).to(device)
        spy.train()  # cố tình để chế độ train: evaluate() phải tự chuyển sang eval
        evaluate(
            spy,
            [(sample_inputs[:2].to(device), sample_targets[:2].to(device))],
            nn.CrossEntropyLoss(),
            device,
        )
        spy.train()  # trả lại trạng thái để model dùng tiếp nếu cần
        guard_ok = bool(spy.training_flags) and not any(spy.training_flags) and not any(spy.grad_flags)
        guard_detail = (
            f"forward của evaluate chạy với training={spy.training_flags}, "
            f"grad_enabled={spy.grad_flags}"
        )
    except RuntimeError as exc:
        guard_ok, guard_detail = False, f"lỗi khi kiểm tra: {exc}"
    add("eval_guard", guard_ok, guard_detail)

    summary = {
        "phase": 3,
        "generated_at": time.strftime("%Y-%m-%d"),
        "passed": sum(1 for c in checks if c["passed"]),
        "total": len(checks),
        "checks": checks,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase3_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("Kết quả: %d/%d PASS → %s", summary["passed"], summary["total"], RESULTS_DIR / "phase3_check.json")
    return checks


def collect_summary_rows() -> list[dict[str, Any]]:
    """Đọc lại summary.csv (nếu có) để kiểm tra run đã được tổng hợp."""
    from .summarize import SUMMARY_PATH, collect_runs

    if SUMMARY_PATH.is_file():
        with SUMMARY_PATH.open(newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    return collect_runs()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Tham số CLI của ``python -m src.train``."""
    parser = argparse.ArgumentParser(description="Huấn luyện SmallCNN trên GTSRB (Giai đoạn 3–5)")
    parser.add_argument("--config", type=str, required=True, help="file YAML thí nghiệm")
    parser.add_argument("--seed", type=int, default=None, help="ghi đè experiment.seed")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="thiết bị chạy"
    )
    parser.add_argument("--overfit-batch", action="store_true", help="test overfit 1 batch")
    parser.add_argument("--determinism-check", action="store_true", help="test tái lập 2 lần cùng seed")
    parser.add_argument("--verify", action="store_true", help="chạy checklist hạ tầng Giai đoạn 3")
    parser.add_argument("--steps", type=int, default=50, help="số vòng cho --overfit-batch")
    parser.add_argument("--epochs", type=int, default=3, help="số epoch cho --determinism-check")
    return parser.parse_args(argv)


def _configure_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point huấn luyện / test hạ tầng."""
    _configure_logging()
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = int(args.seed)
    device = get_device(args.device)

    if args.overfit_batch:
        payload = run_overfit_test(cfg, device, steps=args.steps)
        return 0 if payload["passed_loss"] and payload["passed_acc"] else 1
    if args.determinism_check:
        payload = run_determinism_check(cfg, device, epochs=args.epochs)
        return 0 if payload["passed"] else 1
    if args.verify:
        checks = verify_phase3(args.config, device)
        failed = [c["check"] for c in checks if not c["passed"]]
        if failed:
            LOG.error("Checklist Giai đoạn 3 FAIL: %s", failed)
            return 1
        return 0

    meta = run_training(cfg, device)
    LOG.info("Hoàn tất run %s", meta["run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
