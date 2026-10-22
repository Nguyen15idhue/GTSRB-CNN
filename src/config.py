"""Hằng số, đường dẫn, nạp/validate cấu hình YAML và tiện ích môi trường.

Mọi entrypoint (``src.data``, ``src.train``, ``src.baseline``, ``src.evaluate``)
đều lấy hằng số và cấu hình từ module này để tránh hard-code rải rác.

Cấu hình gồm 2 lớp: ``configs/default.yaml`` (nền) rồi file thí nghiệm
(``configs/e*.yaml``) được **deep-merge** lên trên. Vì vậy file thí nghiệm chỉ
cần khai báo phần khác biệt so với nền.

``mean``/``std`` không hard-code: lấy từ ``outputs/results/data_stats.json`` do
``python -m src.data --prepare`` sinh ra (tính trên **train split**).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

LOG = logging.getLogger(__name__)

# Thư mục gốc repo (file này nằm trong src/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dữ liệu (không commit).
DATA_ROOT = PROJECT_ROOT / "data"
GTSRB_ROOT = DATA_ROOT / "gtsrb"
# Cấu trúc trên đĩa sau Giai đoạn 1:
#   data/gtsrb/GTSRB/Training/<class>/<track>_<frame>.ppm  (archive chính thức, 39.209 ảnh)
#   data/gtsrb/GTSRB/Final_Test/Images/<index>.ppm          (torchvision, 12.630 ảnh)
#   data/gtsrb/GT-final_test.csv
TRAIN_IMAGES_DIR = GTSRB_ROOT / "GTSRB" / "Training"
TEST_IMAGES_DIR = GTSRB_ROOT / "GTSRB" / "Final_Test" / "Images"
TEST_GT_CSV = GTSRB_ROOT / "GT-final_test.csv"

# Cache ảnh đã resize (uint8 .npy, không commit — đã bị .gitignore chặn).
CACHE_DIR = DATA_ROOT / "cache"

# Kết quả / hình (được commit nếu nhỏ).
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RUNS_DIR = OUTPUTS_DIR / "runs"

# Nhãn lớp dùng chung cho train, evaluate, confusion matrix và app.
CLASS_NAMES_PATH = PROJECT_ROOT / "configs" / "class_names.json"

# Cấu hình.
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"

# Artifact của Giai đoạn 2 (split + thống kê chuẩn hóa).
STATS_PATH = RESULTS_DIR / "data_stats.json"
SPLIT_FILES = {
    "random": RESULTS_DIR / "split_random.json",
    "group": RESULTS_DIR / "split_group.json",
}

# Hằng số cốt lõi của bài toán.
NUM_CLASSES = 43
IMG_SIZE = 64
DEFAULT_SEED = 42

# Số liệu công bố để đối chiếu, không dùng thay cho số đếm thực tế.
PUBLISHED_TRAIN_COUNT = 39_209
PUBLISHED_TEST_COUNT = 12_630

REQUIRED_SECTIONS = ("experiment", "data", "model", "train", "output")
VALID_MODELS = ("cnn_a", "cnn_b", "cnn_c")
VALID_SPLITS = ("random", "group")
VALID_OPTIMIZERS = ("adamw", "adam", "sgd")
VALID_SCHEDULERS = ("cosine", "none")
VALID_CLASS_WEIGHTS = ("none", "inv_sqrt_freq")
MONITOR = "val_macro_f1"


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Đọc một file YAML, raise lỗi rõ ràng nếu không đọc được."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy file cấu hình: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: nội dung YAML phải là mapping ở cấp cao nhất")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge đệ quy ``override`` lên ``base`` (không sửa dữ liệu đầu vào)."""
    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(
    path: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
    require_stats: bool = True,
) -> dict[str, Any]:
    """Nạp ``default.yaml`` + file thí nghiệm (+ overrides), validate, trả dict.

    Args:
        path: file YAML của thí nghiệm (tuỳ chọn).
        overrides: dict phủ lên sau cùng (ví dụ ``{"experiment": {"seed": 123}}``).
        require_stats: nếu True thì bắt buộc đã có ``data_stats.json`` để lấy mean/std.

    Raises:
        FileNotFoundError: thiếu file cấu hình hoặc thiếu ``data_stats.json``.
        ValueError: cấu hình sai kiểu/giá trị.
    """
    cfg = load_yaml(DEFAULT_CONFIG_PATH)
    if path is not None:
        cfg = deep_merge(cfg, load_yaml(path))
    if overrides:
        cfg = deep_merge(cfg, dict(overrides))

    _fill_stats(cfg, require_stats=require_stats)
    validate_config(cfg, source=str(path) if path else "default.yaml")
    return cfg


def _fill_stats(cfg: dict[str, Any], require_stats: bool) -> None:
    """Điền ``data.mean``/``data.std`` từ file thống kê nếu chưa khai báo."""
    data = cfg.setdefault("data", {})
    if data.get("mean") and data.get("std"):
        return
    if STATS_PATH.is_file():
        stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        data["mean"] = stats["mean"]
        data["std"] = stats["std"]
        stats_meta = {
            "source": str(STATS_PATH.relative_to(PROJECT_ROOT).as_posix()),
            "split": stats.get("split"),
            "num_images": stats.get("num_images"),
        }
        cfg.setdefault("_resolved", {})["stats"] = stats_meta
        return
    if require_stats:
        raise FileNotFoundError(
            "Chưa có mean/std trong cấu hình và chưa có "
            f"{STATS_PATH.relative_to(PROJECT_ROOT).as_posix()}. "
            "Chạy trước: python -m src.data --prepare"
        )
    data["mean"] = [0.0, 0.0, 0.0]
    data["std"] = [1.0, 1.0, 1.0]


def validate_config(cfg: Mapping[str, Any], source: str = "config") -> None:
    """Kiểm tra khóa/kiểu/ngưỡng bắt buộc; raise ValueError với message rõ ràng."""

    def fail(message: str) -> None:
        raise ValueError(f"{source}: {message}")

    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        fail(f"thiếu nhóm khóa bắt buộc: {missing}")

    experiment = cfg["experiment"]
    if not str(experiment.get("id", "")).strip():
        fail("experiment.id không được rỗng")
    if not isinstance(experiment.get("seed", DEFAULT_SEED), int):
        fail("experiment.seed phải là số nguyên")

    data = cfg["data"]
    if int(data.get("num_classes", 0)) != NUM_CLASSES:
        fail(f"data.num_classes phải bằng {NUM_CLASSES}")
    if int(data.get("image_size", 0)) <= 0:
        fail("data.image_size phải > 0")
    if data.get("split") not in VALID_SPLITS:
        fail(f"data.split phải thuộc {VALID_SPLITS}")
    val_ratio = float(data.get("val_ratio", 0.0))
    if not 0.0 < val_ratio < 1.0:
        fail("data.val_ratio phải trong (0, 1)")
    for key in ("mean", "std"):
        values = data.get(key)
        if values is None or len(values) != 3:
            fail(f"data.{key} phải là danh sách 3 số")
        if key == "std" and any(float(v) <= 0 for v in values):
            fail("data.std phải > 0")

    model = cfg["model"]
    if model.get("name") not in VALID_MODELS:
        fail(f"model.name phải thuộc {VALID_MODELS}")
    widths = model.get("widths")
    if not isinstance(widths, list) or not widths or not all(int(w) > 0 for w in widths):
        fail("model.widths phải là danh sách số nguyên dương")
    if float(model.get("dropout", 0.0)) < 0.0 or float(model.get("dropout", 0.0)) >= 1.0:
        fail("model.dropout phải trong [0, 1)")
    if not bool(model.get("gap", False)) and int(model.get("fc_hidden", 0)) <= 0:
        fail("model.fc_hidden phải > 0 khi không dùng GAP")

    train = cfg["train"]
    if train.get("optimizer") not in VALID_OPTIMIZERS:
        fail(f"train.optimizer phải thuộc {VALID_OPTIMIZERS}")
    if train.get("scheduler") not in VALID_SCHEDULERS:
        fail(f"train.scheduler phải thuộc {VALID_SCHEDULERS}")
    if train.get("class_weight") not in VALID_CLASS_WEIGHTS:
        fail(f"train.class_weight phải thuộc {VALID_CLASS_WEIGHTS}")
    if train.get("monitor") != MONITOR:
        fail(f"train.monitor phải là '{MONITOR}' (quy ước chọn checkpoint)")
    for key in ("lr", "batch_size", "max_epochs"):
        if float(train.get(key, 0)) <= 0:
            fail(f"train.{key} phải > 0")
    if float(train.get("weight_decay", 0.0)) < 0:
        fail("train.weight_decay phải >= 0")
    if int(train.get("patience", 0)) < 1:
        fail("train.patience phải >= 1")
    if float(train.get("batch_size", 0)) != int(train["batch_size"]):
        fail("train.batch_size phải là số nguyên")


def run_id(cfg: Mapping[str, Any], seed: int | None = None) -> str:
    """Tên run theo quy ước ``<experiment.id>_s<seed>``."""
    experiment = cfg["experiment"]
    used_seed = experiment.get("seed", DEFAULT_SEED) if seed is None else seed
    return f"{experiment['id']}_s{used_seed}"


# ---------------------------------------------------------------------------
# Seed / device / môi trường
# ---------------------------------------------------------------------------


def set_seed(seed: int = DEFAULT_SEED) -> torch.Generator:  # noqa: F821 - import trễ
    """Seed Python, NumPy, torch, CUDA và trả generator cho DataLoader."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Hàm khởi tạo worker: seed lại NumPy cho từng worker (tất định)."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_device(prefer: str = "auto"):
    """Trả ``torch.device``; ``prefer`` là ``auto``/``cuda``/``cpu``."""
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Yêu cầu --device cuda nhưng torch.cuda.is_available() = False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit(short: bool = True) -> str:
    """Hash commit hiện tại (chuỗi rỗng nếu không lấy được)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def collect_env() -> dict[str, Any]:
    """Thông tin môi trường để ghi ``env.json`` (bằng chứng tái lập)."""
    import torch

    info: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "python_impl": f"{platform.python_implementation()} {sys.maxsize.bit_length() + 1}bit",
        "sys_prefix": sys.prefix,
        "torch": torch.__version__,
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:  # pragma: no cover - chỉ để env.json đầy đủ khi có torchvision
        info["torchvision"] = None
    info["cuda_build"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        info["driver_version"] = out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        info["driver_version"] = None
    info["os"] = platform.platform()
    info["cpu"] = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    info["ram_gb"] = _total_ram_gb()
    info["repo_commit"] = git_commit(short=False)
    return info


def _total_ram_gb() -> float | None:
    """Tổng RAM (GB) không cần psutil: psutil nếu có, ngược lại ctypes trên Windows."""
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / 1024**3, 2)
    return None
