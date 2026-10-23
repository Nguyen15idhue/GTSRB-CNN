"""Nạp checkpoint và suy luận top-k cho một ảnh — dùng chung cho demo và OOD.

Nguyên tắc: **không** viết lại tiền xử lý. ``preprocess`` được import từ
``src.data`` (hàm duy nhất dùng cho train/eval/demo).

Cách dùng nhanh::

    from src.predict import Predictor
    p = Predictor("outputs/runs/final_s42")
    p.predict(Image.open("external_test/ext_03_kr_speed30_crop.jpg"))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import CLASS_NAMES_PATH, PROJECT_ROOT, get_device
from .data import preprocess  # noqa: F401 - tái xuất hàm tiền xử lý duy nhất
from .models import build_model

LOG = logging.getLogger(__name__)

THRESHOLD_PATH = PROJECT_ROOT / "outputs" / "results" / "threshold.json"


def load_model(
    run_dir: str | Path, device: torch.device | None = None
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], torch.device]:
    """Nạp ``best.pt`` → ``(model ở eval(), config, class_names, device)``."""
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}. "
            "Chạy Giai đoạn 6 (python -m src.train --config configs/final.yaml) trước."
        )
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, _ = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return model, checkpoint["config"], class_names, device


def load_tau(path: Path = THRESHOLD_PATH) -> float:
    """Đọc ngưỡng unknown τ đã chọn trên val (``should_reject``)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Chưa có {path}. Chạy: python -m src.error_analysis --run outputs/runs/final_s42"
        )
    return float(json.loads(path.read_text(encoding="utf-8"))["tau"])


def should_reject(max_prob: float, tau: float) -> bool:
    """Trả True nếu ``max_prob < tau`` → nên hiển thị "không xác định"."""
    return float(max_prob) < float(tau)


def validate_image_bytes(filename: str, data: bytes, max_mb: float = 20.0) -> Image.Image:
    """Kiểm tra file upload: kích thước, định dạng, ảnh hỏng → raise ValueError thân thiện."""
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(f"Ảnh quá lớn ({len(data) / 1024**2:.1f} MB). Giới hạn {max_mb:.0f} MB.")
    suffix = Path(filename).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png"):
        raise ValueError(f"Định dạng '{suffix or 'không rõ'}' không hỗ trợ. Chỉ nhận JPG/JPEG/PNG.")
    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "File không phải ảnh hợp lệ hoặc đã bị hỏng (kiểm tra lại đuôi file)."
        ) from exc


@torch.no_grad()
def predict_topk(
    image: Any,
    model: torch.nn.Module,
    cfg: Mapping[str, Any],
    class_names: Mapping[str, Mapping[str, str]],
    k: int = 3,
    device: torch.device | None = None,
    tau: float | None = None,
) -> dict[str, Any]:
    """Dự đoán top-k cho một ảnh (PIL/ndarray/đường dẫn) bằng đúng ``preprocess`` của train."""
    device = device or next(model.parameters()).device
    tensor = preprocess(
        image,
        size=int(cfg["data"]["image_size"]),
        mean=list(cfg["data"]["mean"]),
        std=list(cfg["data"]["std"]),
    ).unsqueeze(0).to(device)
    model.eval()
    probabilities = torch.softmax(model(tensor).float(), dim=1)[0].cpu().numpy()
    order = probabilities.argsort()[::-1][:k]
    topk = [
        {
            "class_id": int(class_id),
            "name_vi": class_names[str(int(class_id))]["vi"],
            "name_en": class_names[str(int(class_id))]["en"],
            "prob": float(probabilities[class_id]),
        }
        for class_id in order
    ]
    max_prob = float(probabilities.max())
    return {
        "topk": topk,
        "probabilities": [float(value) for value in probabilities],
        "max_prob": max_prob,
        "margin": float(topk[0]["prob"] - topk[1]["prob"]) if len(topk) > 1 else max_prob,
        "tau": tau,
        "unknown": should_reject(max_prob, tau) if tau is not None else False,
    }


class Predictor:
    """Gói model + config + class_names để gọi ``predict`` nhiều lần (app, OOD)."""

    def __init__(self, run_dir: str | Path, tau: float | None = None) -> None:
        self.model, self.cfg, self.class_names, self.device = load_model(run_dir)
        self.tau = tau
        self.run_dir = Path(run_dir)

    def predict(self, image: Any, k: int = 3, tau: float | None = None) -> dict[str, Any]:
        used_tau = self.tau if tau is None else tau
        return predict_topk(
            image,
            self.model,
            self.cfg,
            self.class_names,
            k=k,
            device=self.device,
            tau=used_tau,
        )

    def predict_many(self, images: Sequence[Any], k: int = 3, tau: float | None = None) -> list[dict[str, Any]]:
        return [self.predict(image, k=k, tau=tau) for image in images]
