"""Sinh ``configs/v2/*.yaml`` từ ``docs/kehoach/1/MA_TRAN_RUN.csv``.

Chỉ sinh config cho các dòng có ``role != alias`` và chưa phải ``TBD``.
Alias (D4=F15, L3=F15, G0=F07, G5=F15, R1–R4) không sinh file riêng để tránh
chạy trùng; mapping alias ghi trong ``configs/v2/ALIASES.md``.

Cách dùng::

    python scripts/gen_configs_v2.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "kehoach" / "1" / "MA_TRAN_RUN.csv"
OUT_DIR = ROOT / "configs" / "v2"


def _float(value: str, default: float = 0.0) -> float:
    return float(value) if str(value).strip() else default


def _bool_on(value: str) -> bool:
    return str(value).strip().lower() in ("on", "true", "1", "yes")


def build_config(row: dict[str, str]) -> dict:
    """Chuyển một dòng CSV thành cấu hình YAML đầy đủ."""
    config_id = row["config_id"].strip().lower()
    split_raw = row["train_split"].strip()
    split = "fold" if split_raw == "dev_fold" else "screening"
    widths = [int(v) for v in row["widths"].split("/")]
    return {
        "experiment": {
            "id": config_id,
            "protocol": "v2_clean",
            "block": row["block"].strip(),
            "seed": 42,
            "notes": row["purpose"].strip(),
        },
        "data": {
            "split": split,
            "fold": None,
            "image_size": int(row["input_size"]),
            "num_classes": 43,
            "num_workers": 4,
        },
        "model": {
            "name": "smallcnn_4blk" if len(widths) == 4 else "smallcnn_3blk",
            "widths": widths,
            "use_bn": _bool_on(row["bn"]),
            "dropout": _float(row["dropout"]),
            "gap": _bool_on(row["gap"]),
            "fc_hidden": int(row["fc_hidden"]),
        },
        "augmentation": {
            "enabled": _bool_on(row["aug"]),
            "profile": row["aug_profile"].strip() or "none",
        },
        "train": {
            "optimizer": "adamw",
            "lr": float(row["lr"]),
            "weight_decay": _float(row["wd"]),
            "batch_size": int(row["batch_size"]),
            "max_epochs": int(row["epochs"]),
            "patience": 7,
            "scheduler": "cosine",
            "warmup_epochs": int(row["warmup_epochs"] or 0),
            "class_weight": (row["class_weight"].strip() or "none"),
            "label_smoothing": _float(row["label_smoothing"]),
            "amp": True,
            "monitor": "val_macro_f1",
            "deterministic": True,
        },
        "output": {"dir": "outputs/runs"},
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            config_id = row["config_id"].strip()
            role = row["role"].strip()
            priority = row["priority"].strip()
            if priority == "alias" or role == "alias" or config_id.upper().startswith(("X", "FINAL")) or config_id == "B1":
                skipped.append(config_id)
                continue
            cfg = build_config(row)
            path = OUT_DIR / f"{config_id.lower()}.yaml"
            path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            written.append(path.name)
    (OUT_DIR / "ALIASES.md").write_text(
        "# Alias không chạy lại\n\n"
        "- D4 = F15\n- L3 = F15\n- G0 = F07\n- G5 = F15\n"
        "- R1 = F00, R2 = F07, R3 = F15, R4 = D3\n\n"
        "Các run alias dùng lại artifact của config gốc; không sinh YAML riêng.\n",
        encoding="utf-8",
    )
    print(f"Đã sinh {len(written)} config vào {OUT_DIR}: {', '.join(written)}")
    print(f"Bỏ qua (alias/TBD/baseline): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
