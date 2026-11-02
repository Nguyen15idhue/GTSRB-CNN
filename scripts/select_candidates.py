"""Chọn top 2 + control từ kết quả screening (bước 2.4), TRƯỚC khi chạy OOF.

Luật (đăng ký trước):
1. Sắp theo mean macro-F1 (3 seed).
2. Nhóm tương đương = các config trong 0,3 điểm % so với tốt nhất.
3. Trong nhóm, chọn top 2 theo tie-break: ít params → latency thấp → ít aug →
   ít epoch → đơn giản hơn.
4. Thêm 1 control: F07 (nếu F07 đã trong top 2 thì F00).

Ghi lý do vào ``outputs/results/v2/candidates.json`` và sinh ``configs/v2/x1..x3.yaml``
(split = fold). Cập nhật X1–X3 trong ``MA_TRAN_RUN.csv``.

Cách dùng::

    python scripts/select_candidates.py
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

CONFIG_DIR = ROOT / "configs" / "v2"
SUMMARY = ROOT / "outputs" / "results" / "v2" / "summary.csv"
OUT = ROOT / "outputs" / "results" / "v2" / "candidates.json"
MATRIX = ROOT / "docs" / "kehoach" / "1" / "MA_TRAN_RUN.csv"


def _rank_key(row: dict) -> tuple:
    return (
        -float(row["macro_f1_mean"]),
        int(row["params"]),
        float(row["infer_ms_per_img"]),
        1 if row["aug_profile"] not in ("none", "") else 0,
        int(round(float(row["best_epoch_mean"]))),
    )


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    with SUMMARY.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    screening = [r for r in rows if r["split"] == "screening" and r["config_id"] not in ("b1",)]
    by_config: dict[str, list[dict]] = {}
    for row in screening:
        by_config.setdefault(row["config_id"], []).append(row)
    table = []
    for config_id, runs in by_config.items():
        import numpy as np

        values = np.array([float(r["val_macro_f1"]) for r in runs])
        table.append({
            "config_id": config_id,
            "n_seeds": len(runs),
            "macro_f1_mean": float(values.mean()),
            "macro_f1_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "params": int(runs[0]["params"]),
            "infer_ms_per_img": float(runs[0]["infer_ms_per_img"]),
            "aug_profile": runs[0]["aug_profile"],
            "best_epoch_mean": float(np.mean([int(r["best_epoch"]) for r in runs])),
        })
    if not table:
        raise SystemExit("summary.csv chưa có run screening — chạy Phase 1–2 trước")
    table.sort(key=_rank_key)
    best = table[0]["macro_f1_mean"]
    equivalent = [r for r in table if best - r["macro_f1_mean"] <= 0.003]
    forced = False
    top2 = equivalent[:2]
    if len(top2) < 2:
        # Không có 2 cấu hình trong biên tương đương: lấy tiếp theo thứ hạng,
        # ghi rõ là "forced" để OOF vẫn có 2 ứng viên theo thiết kế xác nhận.
        for row in table:
            if row["config_id"] not in {r["config_id"] for r in top2}:
                top2.append(row)
                forced = True
            if len(top2) == 2:
                break
    top_ids = {r["config_id"] for r in top2}
    control = "f07" if "f07" not in top_ids else "f00"
    control_row = next((r for r in table if r["config_id"] == control), None)
    if control_row is None:
        raise SystemExit(f"Không có control {control} trong screening")

    chosen = top2 + [control_row]
    payload = {
        "rule": "mean macro-F1 3 seed; nhóm trong 0.3 điểm % so với tốt nhất; tie-break params/latency/aug/epoch; control F07 (hoặc F00)",
        "best_config": table[0]["config_id"],
        "best_macro_f1": best,
        "equivalent_group": [r["config_id"] for r in equivalent],
        "top2": [r["config_id"] for r in top2],
        "forced_second_candidate": forced,
        "control": control,
        "rationale": {
            r["config_id"]: (
                f"f1={r['macro_f1_mean']:.6f} (sd {r['macro_f1_sd']:.6f}), "
                f"params={r['params']}, latency={r['infer_ms_per_img']:.4f} ms, "
                f"aug={r['aug_profile']}, best_epoch={r['best_epoch_mean']:.1f}"
            )
            for r in chosen
        },
        "ranked": table,
        "note": "Chọn trước khi chạy OOF; không đổi sau khi xem OOF.",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Sinh config x1..x3 (giữ nguyên hyperparam, chỉ đổi split sang fold).
    for index, row in enumerate(chosen, start=1):
        source = row["config_id"]
        cfg = yaml.safe_load((CONFIG_DIR / f"{source}.yaml").read_text(encoding="utf-8"))
        cfg["experiment"]["id"] = f"x{index}"
        cfg["experiment"]["block"] = "X_confirmation"
        cfg["experiment"]["notes"] = f"OOF cho {source}: {payload['rationale'][source]}"
        cfg["experiment"]["source_config"] = source
        cfg["data"]["split"] = "fold"
        cfg["data"]["fold"] = 0
        (CONFIG_DIR / f"x{index}.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    # Cập nhật MA_TRAN_RUN.csv (X1–X3).
    with MATRIX.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        matrix = list(reader)
    for index, row in enumerate(chosen, start=1):
        source = row["config_id"]
        src_cfg = yaml.safe_load((CONFIG_DIR / f"{source}.yaml").read_text(encoding="utf-8"))
        for line in matrix:
            if line["config_id"] == f"X{index}":
                line["model"] = src_cfg["model"]["name"]
                line["widths"] = "/".join(str(w) for w in src_cfg["model"]["widths"])
                line["bn"] = "on" if src_cfg["model"]["use_bn"] else "off"
                line["dropout"] = str(src_cfg["model"]["dropout"])
                line["wd"] = str(src_cfg["train"]["weight_decay"])
                line["aug"] = "on" if src_cfg["augmentation"]["enabled"] else "off"
                line["aug_profile"] = src_cfg["augmentation"]["profile"]
                line["gap"] = "on" if src_cfg["model"]["gap"] else "off"
                line["fc_hidden"] = str(src_cfg["model"]["fc_hidden"])
                line["input_size"] = str(src_cfg["data"]["image_size"])
                line["batch_size"] = str(src_cfg["train"]["batch_size"])
                line["lr"] = str(src_cfg["train"]["lr"])
                line["purpose"] = f"OOF cho {source}: {payload['rationale'][source]}"
    with MATRIX.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix)

    print(f"Top2: {payload['top2']} | control: {control}")
    print(f"Đã ghi {OUT} và configs/v2/x1..x3.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
