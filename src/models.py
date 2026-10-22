"""Định nghĩa CNN tự xây dựng (SmallCNN A/B/C) và ``build_model``.

Ba cấu hình dùng chung một lớp, khác nhau ở ``widths`` / ``use_bn`` /
``dropout`` / ``gap``:

* CNN-A: 3 khối conv, không BN, không dropout, không GAP (mốc so sánh).
* CNN-B: 3 khối conv + BN + dropout + weight decay (mô hình chính).
* CNN-C: 4 khối conv + BN + GAP → sâu hơn nhưng ít tham số hơn CNN-B.

Model trả **logits** (không softmax) để dùng trực tiếp với ``CrossEntropyLoss``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import torch
import torch.nn as nn

from .config import DEFAULT_SEED, NUM_CLASSES

LOG = logging.getLogger(__name__)


class SmallCNN(nn.Module):
    """CNN nhỏ cho ảnh 64×64: các khối conv 3×3 + BN + ReLU + MaxPool, rồi đầu phân loại."""

    def __init__(
        self,
        widths: tuple[int, ...] | list[int] = (32, 64, 128),
        use_bn: bool = True,
        dropout: float = 0.3,
        gap: bool = False,
        fc_hidden: int = 256,
        num_classes: int = NUM_CLASSES,
        in_size: int = 64,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        widths = tuple(int(w) for w in widths)
        if not widths:
            raise ValueError("widths phải có ít nhất một khối conv")
        if int(in_size) % (2 ** len(widths)) != 0:
            raise ValueError(
                f"in_size={in_size} không chia hết cho 2^{len(widths)} sau {len(widths)} lần MaxPool"
            )
        self.widths = widths
        self.gap = bool(gap)
        self.in_size = int(in_size)

        blocks: list[nn.Module] = []
        previous = int(in_channels)
        for width in widths:
            layers: list[nn.Module] = [
                nn.Conv2d(previous, width, kernel_size=3, padding=1, bias=not use_bn)
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(width))
            layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(2)])
            blocks.append(nn.Sequential(*layers))
            previous = width
        self.features = nn.Sequential(*blocks)

        if self.gap:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(previous, num_classes),
            )
        else:
            flattened = previous * (int(in_size) // 2 ** len(widths)) ** 2
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(flattened, int(fc_hidden)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(int(fc_hidden), num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` → logits ``(B, 43)``."""
        x = self.features(x)
        if self.gap:
            x = x.mean(dim=(2, 3))
        else:
            x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Số tham số huấn luyện được (không tính buffer của BatchNorm)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_model(cfg: Mapping[str, Any]) -> tuple[SmallCNN, int]:
    """Tạo model từ config; trả ``(model, số params)``."""
    model_cfg = cfg["model"]
    model = SmallCNN(
        widths=model_cfg["widths"],
        use_bn=bool(model_cfg["use_bn"]),
        dropout=float(model_cfg["dropout"]),
        gap=bool(model_cfg["gap"]),
        fc_hidden=int(model_cfg.get("fc_hidden", 256)),
        num_classes=int(cfg["data"]["num_classes"]),
        in_size=int(cfg["data"]["image_size"]),
    )
    params = count_parameters(model)
    LOG.info(
        "Model %s: %s params (widths=%s, bn=%s, dropout=%.2f, gap=%s)",
        model_cfg["name"],
        f"{params:,}",
        model.widths,
        model_cfg["use_bn"],
        float(model_cfg["dropout"]),
        model_cfg["gap"],
    )
    return model, params


def smoke_forward(cfg: Mapping[str, Any], batch: int = 2) -> tuple[SmallCNN, torch.Tensor]:
    """Forward với batch giả để kiểm tra shape đầu ra (test nhanh, không cần dữ liệu)."""
    torch.manual_seed(DEFAULT_SEED)
    model, _ = build_model(cfg)
    size = int(cfg["data"]["image_size"])
    dummy = torch.randn(batch, 3, size, size)
    model.eval()
    with torch.no_grad():
        logits = model(dummy)
    expected = (batch, int(cfg["data"]["num_classes"]))
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"Forward ra {tuple(logits.shape)}, kỳ vọng {expected}")
    return model, logits
