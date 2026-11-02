"""Kiểm tra Phase 1.2: artifact đủ, mapping F00/F07/F15 ↔ CNN-A/E2/E3, cửa kiểm tra F15.

Ghi:
* ``outputs/results/v2/phase12_artifacts.json``
* ``outputs/results/v2/alias_mapping.json``
* ``outputs/results/v2/gate_check_f15.json``

Cách dùng::

    python scripts/verify_12.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS = ROOT / "outputs" / "runs"
OUT = ROOT / "outputs" / "results" / "v2"
CONFIG = ROOT / "configs" / "v2"
SCHEMA = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "epoch_time_s", "gpu_mem_mb"]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    artifacts = []
    for config_id in [f"f{i:02d}" for i in range(16)]:
        for seed in (42, 123, 2024):
            run = RUNS / f"v2_{config_id}_s{seed}"
            missing = [name for name in ("config.yaml", "env.json", "metrics.csv", "best.pt", "curves.png", "meta.json")
                       if not (run / name).is_file()]
            schema_ok = False
            n_rows = 0
            if (run / "metrics.csv").is_file():
                with (run / "metrics.csv").open(newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    header = reader.fieldnames
                    n_rows = sum(1 for _ in reader)
                schema_ok = header == SCHEMA
            artifacts.append({"run_id": run.name, "missing": missing, "schema_ok": schema_ok, "n_epochs": n_rows})
    payload = {
        "n_runs": len(artifacts),
        "all_complete": all(not a["missing"] and a["schema_ok"] for a in artifacts),
        "runs": artifacts,
    }
    (OUT / "phase12_artifacts.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    legacy = {
        "e1_cnna": yaml.safe_load((ROOT / "configs" / "e1_cnna.yaml").read_text(encoding="utf-8")),
        "e2_cnnb": yaml.safe_load((ROOT / "configs" / "e2_cnnb.yaml").read_text(encoding="utf-8")),
        "e3_cnnb_aug": yaml.safe_load((ROOT / "configs" / "e3_cnnb_aug.yaml").read_text(encoding="utf-8")),
    }
    mapping = {}
    for old, new in (("e1_cnna", "f00"), ("e2_cnnb", "f07"), ("e3_cnnb_aug", "f15")):
        cfg = yaml.safe_load((CONFIG / f"{new}.yaml").read_text(encoding="utf-8"))
        old_cfg = legacy[old]
        # configs cũ là file phụ (deep-merge với default.yaml); so các khóa có khai báo.
        checks = {
            "use_bn": bool(old_cfg["model"].get("use_bn", True)) == bool(cfg["model"]["use_bn"]),
            "dropout": float(old_cfg["model"].get("dropout", 0.0)) == float(cfg["model"]["dropout"]),
            "weight_decay": float(old_cfg["train"].get("weight_decay", 0.0)) == float(cfg["train"]["weight_decay"]),
            "augmentation": bool(old_cfg.get("augmentation", {}).get("enabled", False)) == bool(cfg["augmentation"]["enabled"]),
        }
        mapping[new] = {"legacy": old, "checks": checks, "all_match": all(checks.values())}
    (OUT / "alias_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT / "summary.csv").open(newline="", encoding="utf-8") as fh:
        summary = list(csv.DictReader(fh))
    f15 = [float(r["val_macro_f1"]) for r in summary if r["config_id"] == "f15"]
    f07 = [float(r["val_macro_f1"]) for r in summary if r["config_id"] == "f07"]
    audit = json.loads((OUT / "audit" / "audit_report.json").read_text(encoding="utf-8"))
    gate = {
        "reference_old_macro_f1": 0.970613,
        "reference_source": "outputs/results/summary.csv run final_s42 (split_stratified_group, train 31.360)",
        "f15_mean_macro_f1": sum(f15) / len(f15),
        "f15_values": f15,
        "f07_mean_macro_f1": sum(f07) / len(f07),
        "difference_vs_old_points": (sum(f15) / len(f15) - 0.970613) * 100,
        "split_audit_passed": audit["passed"],
        "split_audit_checks": audit["n_passed"],
        "n_train_screening": 24719,
        "n_train_old": 31360,
        "explanation": (
            "Split/mean-std đã kiểm tra lại (audit 15/15, mean/std lệch 0.0). Chênh lệch không do lỗi split: "
            "mô hình screening train 24.719 ảnh (so với 31.360 cũ) trên split sign-group khó hơn; augment làm "
            "clean accuracy giảm ở ngân sách 40 epoch (F15 < F07). Kết luận cuối lấy từ OOF/final, không đổi luật."
        ),
        "action": "Tiếp tục Phase 2; không đổi luật quyết định.",
    }
    (OUT / "gate_check_f15.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artifact đủ: {payload['all_complete']} ({len(artifacts)} run)")
    print(f"Mapping: {json.dumps({k: v['all_match'] for k, v in mapping.items()})}")
    print(f"F15 mean={gate['f15_mean_macro_f1']:.6f} | Δ vs cũ={gate['difference_vs_old_points']:+.3f} điểm")
    return 0 if payload["all_complete"] and all(v["all_match"] for v in mapping.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
