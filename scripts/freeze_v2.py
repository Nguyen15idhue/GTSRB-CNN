"""Đóng băng mô hình winner (bước 3.2): sinh ``configs/v2/final.yaml`` và FREEZE.json.

Không mở lockbox. Ghi:
* winner config id, seed, epoch = median best epoch của 9 run OOF
* git commit, config_sha256, split_sha256 (dev_all)
* lý do chọn theo luật 4.5

Cách dùng::

    python scripts/freeze_v2.py --winner f15 --reason "OOF tuong duong, chon it aug/params hon"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import git_commit  # noqa: E402
from src.data_v2 import split_sha256  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "v2"
FREEZE_PATH = ROOT / "outputs" / "results" / "v2" / "FREEZE.json"
RUNS_DIR = ROOT / "outputs" / "runs"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Freeze winner v2")
    parser.add_argument("--winner", required=True, help="config_id chạy OOF, ví dụ x2")
    parser.add_argument("--source", default=None, help="config gốc để clone (mặc định = winner)")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--epoch", type=int, default=0, help="0 = median best epoch của 9 run OOF")
    args = parser.parse_args()

    winner = args.winner.lower()
    source = (args.source or winner).lower()
    cfg = yaml.safe_load((CONFIG_DIR / f"{source}.yaml").read_text(encoding="utf-8"))
    best_epochs = []
    for seed in (42, 123, 2024):
        for fold in range(3):
            meta_path = RUNS_DIR / f"v2_{winner}_s{seed}_f{fold}" / "meta.json"
            if meta_path.is_file():
                best_epochs.append(int(json.loads(meta_path.read_text(encoding="utf-8"))["best_epoch"]))
    if not best_epochs:
        raise SystemExit(f"Không có run OOF nào của {winner}; chạy bước 3.1 trước")
    epoch = args.epoch or int(statistics.median(best_epochs))

    final = json.loads(json.dumps(cfg))
    final["experiment"]["id"] = "final"
    final["experiment"]["notes"] = (
        f"final từ {source} (winner OOF {winner}); epoch cố định {epoch} "
        f"(median {len(best_epochs)} run OOF); {args.reason}"
    )
    final["data"]["split"] = "dev_all"
    final["data"]["fold"] = None
    final["train"]["max_epochs"] = int(epoch)
    final["train"]["patience"] = int(epoch) + 1  # không early stop trước epoch cố định
    final_path = CONFIG_DIR / "final.yaml"
    final_path.write_text(yaml.safe_dump(final, allow_unicode=True, sort_keys=False), encoding="utf-8")

    freeze = {
        "protocol": "v2_clean",
        "winner": winner,
        "source_config": source,
        "reason": args.reason,
        "epoch": int(epoch),
        "epoch_source": "median best epoch của 9 run OOF" if not args.epoch else "chỉ định",
        "best_epochs_oof": best_epochs,
        "final_config": "configs/v2/final.yaml",
        "final_config_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "split_sha256": split_sha256("dev_all"),
        "git_commit": git_commit(short=False),
        "frozen_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "lockbox_opened": False,
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã freeze {winner} tại epoch {epoch} → {final_path}")
    print(f"config_sha256={freeze['final_config_sha256'][:16]} split_sha256={freeze['split_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
