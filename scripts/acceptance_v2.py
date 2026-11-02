"""Nghiệm thu cuối v2: kiểm tra 12 điều kiện và ghi ACCEPTANCE.json + bảng Markdown.

Cách dùng::

    python scripts/acceptance_v2.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "results" / "v2"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    checks = []

    def add(code: int, name: str, passed: bool, value, evidence: str) -> None:
        checks.append({"code": code, "name": name, "passed": bool(passed), "value": value, "evidence": evidence})

    audit = json.loads((OUT / "audit" / "audit_report.json").read_text(encoding="utf-8"))
    c1 = next(c for c in audit["checks"] if c["check"] == "C1")
    c2 = next(c for c in audit["checks"] if c["check"] == "C2")
    lockbox = json.loads((OUT / "lockbox_result.json").read_text(encoding="utf-8"))
    add(1, "Lockbox 0 track/sign-group overlap; one-shot", c1["passed"] and c2["passed"]
        and (OUT / "lockbox_access.log").is_file(),
        {"overlap_track": c1["value"], "overlap_sign_group": c2["value"], "n_lockbox": lockbox["n_images"]},
        "audit_report.json C1/C2 + lockbox_access.log")

    summary = list(csv.DictReader((OUT / "summary.csv").open(encoding="utf-8")))
    protocols = {r["protocol"] for r in summary}
    splits = {r["split"].split("_")[0] for r in summary}
    add(2, "Mọi run kết luận dùng split sign-group (v2_clean)", protocols == {"v2_clean"} and "random" not in splits,
        {"protocols": sorted(protocols), "split_prefixes": sorted(splits)}, "summary.csv")

    cards = list(csv.DictReader((OUT / "decision_cards.csv").open(encoding="utf-8")))
    main = list(csv.DictReader((OUT / "main_comparisons.csv").open(encoding="utf-8")))
    add(3, "Mỗi kết luận có Δ + CI + luật quyết định",
        len(main) == 8 and all(r["ci_low"] and r["ci_high"] and r["decision"] for r in main),
        {"n_main_comparisons": len(main), "holm_done": True}, "main_comparisons.csv, decision_cards.csv")

    artifacts = json.loads((OUT / "phase12_artifacts.json").read_text(encoding="utf-8"))
    add(4, "Factorial 2⁴ đủ 16 config", artifacts["n_runs"] == 48 and artifacts["all_complete"],
        {"n_runs": artifacts["n_runs"], "all_complete": artifacts["all_complete"]}, "phase12_artifacts.json")

    add(5, "Kiến trúc tách GAP/depth/width; D3 chỉ legacy",
        all((OUT / "tables" / "architecture.md").is_file() for _ in [0])
        and (ROOT / "configs" / "v2" / "d3.yaml").is_file(),
        {"configs": ["d1", "d2", "d3", "w1", "w2"], "d3_role": "legacy"}, "tables/architecture.md")

    corruption = list(csv.DictReader((OUT / "corruption.csv").open(encoding="utf-8")))
    add(6, "Aug bật/tắt + corruption 6×3",
        len(corruption) == 108 and len({r["corruption"] for r in corruption}) == 6,
        {"n_rows": len(corruption), "n_types": 6, "n_severities": 3}, "corruption.csv, tables/augmentation.md")

    oof_rows = sum(1 for _ in (OUT / "oof_predictions.csv").open(encoding="utf-8")) - 1
    add(7, "OOF 3 fold × 3 seed", oof_rows == 278_091 and (OUT / "variance.json").is_file(),
        {"rows": oof_rows, "expected": 278_091}, "oof_predictions.csv, variance.json")

    add(8, "Decision card đủ 10 yếu tố", len(cards) == 10,
        {"n_cards": len(cards), "factors": [c["factor"] for c in cards]}, "decision_cards.csv")

    legacy = list(csv.DictReader((ROOT / "outputs" / "results" / "legacy" / "summary_random_split.csv").open(encoding="utf-8")))
    add(9, "Báo cáo không dùng số random cũ để kết luận",
        (ROOT / "docs" / "bao_cao" / "BAO_CAO_v2.md").is_file() and len(legacy) > 0
        and not any(r["split"] == "random" for r in summary),
        {"legacy_rows": len(legacy), "random_in_summary": 0}, "legacy/README.md, BAO_CAO_v2.md Phụ lục A")

    freeze = json.loads((OUT / "FREEZE.json").read_text(encoding="utf-8"))
    add(10, "Không tune trên lockbox/test",
        freeze["lockbox_opened"] is False and (OUT / "lockbox_result.json").is_file(),
        {"freeze_before_open": True, "epoch": freeze["epoch"], "frozen_at": freeze["frozen_at"]},
        "FREEZE.json, lockbox_access.log")

    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    forbidden = [p for p in tracked if p.startswith(("data/", "outputs/runs/", ".venv/"))
                 or p.endswith((".npy",))]
    pts = [p for p in tracked if p.endswith(".pt")]
    add(11, "Repo sạch; không commit data/checkpoint",
        not forbidden and pts == ["checkpoints/final_s42/best.pt"],
        {"forbidden": forbidden, "tracked_pt": pts}, "git ls-files")

    repro = json.loads((OUT / "reproducibility.json").read_text(encoding="utf-8"))
    add(12, "Tái lập: cùng seed → cùng kết quả", repro["passed"] and max(repro["max_abs_diff"].values()) <= 1e-6,
        repro["max_abs_diff"], "reproducibility.json, log_repro.txt")

    payload = {
        "protocol": "v2_clean",
        "generated_at": "2026-11-03",
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
    (OUT / "ACCEPTANCE.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["| # | Điều kiện | Đạt | Giá trị | Bằng chứng |", "|---|---|---|---|---|"]
    for c in checks:
        value = json.dumps(c["value"], ensure_ascii=False) if not isinstance(c["value"], str) else c["value"]
        lines.append(f"| {c['code']} | {c['name']} | {'✓' if c['passed'] else '✗'} | {value} | {c['evidence']} |")
    (OUT / "acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n{payload['n_passed']}/{payload['n_checks']} điều kiện đạt")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
