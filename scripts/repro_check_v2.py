"""Kiểm tra tái lập: cùng seed, cùng split, 2 lần chạy ngắn phải cho metric giống nhau.

Không dùng run chính; tạo config tạm 3 epoch trong ``outputs/results/v2/tmp``.
Ghi ``outputs/results/v2/reproducibility.json``.

Cách dùng::

    python scripts/repro_check_v2.py
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

from src.train_v2 import run_training_v2  # noqa: E402

TMP = ROOT / "outputs" / "results" / "v2" / "tmp"
RUNS = ROOT / "outputs" / "runs"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    TMP.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / "configs" / "v2" / "f00.yaml").read_text(encoding="utf-8"))
    paths = []
    for tag in ("a", "b"):
        cfg = json.loads(json.dumps(base))
        cfg["experiment"]["id"] = f"repro{tag}"
        cfg["experiment"]["notes"] = "kiểm tra tái lập 3 epoch"
        cfg["train"]["max_epochs"] = 3
        path = TMP / f"repro_{tag}.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths.append(path)

    metas = []
    for path in paths:
        for run_dir in RUNS.glob(f"v2_repro{path.stem[-1]}_s42"):
            import shutil

            shutil.rmtree(run_dir)
        metas.append(run_training_v2(path, seed=42))

    rows = []
    for meta in metas:
        with (RUNS / meta["run_id"] / "metrics.csv").open(newline="", encoding="utf-8") as fh:
            rows.append(list(csv.DictReader(fh)))
    diffs = {
        key: max(abs(float(a[key]) - float(b[key])) for a, b in zip(rows[0], rows[1]))
        for key in ("train_loss", "val_loss", "val_acc", "val_macro_f1")
    }
    payload = {
        "seed": 42,
        "epochs": 3,
        "runs": [m["run_id"] for m in metas],
        "max_abs_diff": diffs,
        "tolerance": 1e-6,
        "passed": all(v <= 1e-6 for v in diffs.values()),
        "note": "Cùng seed/split/config; khác experiment.id không ảnh hưởng quá trình học.",
    }
    out = ROOT / "outputs" / "results" / "v2" / "reproducibility.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
