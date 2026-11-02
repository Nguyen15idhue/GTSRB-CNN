"""Kiểm tra checkpoint final trước khi mở lockbox (bước 3.3)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402

OUT = ROOT / "outputs" / "results" / "v2"
RUNS = ROOT / "outputs" / "runs"


def main() -> int:
    freeze = json.loads((OUT / "FREEZE.json").read_text(encoding="utf-8"))
    checks = []
    for seed in (42, 123, 2024):
        run = RUNS / f"v2_final_s{seed}"
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        model, params = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 3, 64, 64))
        with (run / "metrics.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        checks.append({
            "run_id": meta["run_id"],
            "epochs_run": len(rows),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "params": params,
            "forward_shape": list(logits.shape),
            "config_sha256_match": checkpoint["config_sha256"] == freeze["final_config_sha256"],
            "split_sha256_match": meta["split_sha256"] == freeze["split_sha256"],
            "meta_config_sha256_match": meta["config_sha256"] == freeze["final_config_sha256"],
        })
    payload = {
        "freeze": freeze,
        "checks": checks,
        "passed": all(
            c["forward_shape"] == [2, 43]
            and c["config_sha256_match"]
            and c["split_sha256_match"]
            and c["meta_config_sha256_match"]
            and c["checkpoint_epoch"] == freeze["epoch"]
            for c in checks
        ),
    }
    (OUT / "final_ckpt_check.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
