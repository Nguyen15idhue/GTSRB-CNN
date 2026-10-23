"""Golden test 20 ảnh: đo xác suất để so giữa 2 môi trường (AGENTS §10).

Sinh (trong môi trường A):

    python tests/golden_test.py dump --out outputs/results/golden_venv.json

So sánh (trong môi trường B hoặc sau khi có 2 file):

    python tests/golden_test.py dump --out outputs/results/golden_venv_test.json
    python tests/golden_test.py compare outputs/results/golden_venv.json \
        outputs/results/golden_venv_test.json

Đạt nếu sai khác xác suất lớn nhất < 1e-3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))

from src.data import load_cache, load_split  # noqa: E402
from src.predict import Predictor  # noqa: E402


def dump(out_path: Path, run_dir: Path, n: int = 20) -> dict:
    cache = load_cache(64)
    split = load_split("stratified_group")
    indices = sorted(split["val"])[:n]
    predictor = Predictor(run_dir, tau=None)
    entries = []
    for index in indices:
        tensor_image = cache["train"]["x"][index]
        prediction = predictor.predict(tensor_image, k=3)
        entries.append(
            {
                "index": int(index),
                "path": cache["train"]["paths"][index],
                "pred_id": prediction["topk"][0]["class_id"],
                "probs": [float(p) for p in prediction["probabilities"]],
            }
        )
    checkpoint = run_dir / "best.pt"
    payload = {
        "run_dir": run_dir.as_posix(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "torch_version": __import__("torch").__version__,
        "device": str(predictor.device),
        "n_images": len(entries),
        "entries": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {out_path} ({n} ảnh, torch {payload['torch_version']}, {payload['device']})")
    return payload


def compare(left_path: Path, right_path: Path, threshold: float = 1e-3) -> int:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    if left["n_images"] != right["n_images"]:
        print("FAIL: số ảnh khác nhau")
        return 1
    if left["checkpoint_sha256"] != right["checkpoint_sha256"]:
        print("FAIL: checkpoint sha256 khác nhau")
        return 1
    diffs = []
    pred_mismatch = 0
    for a, b in zip(left["entries"], right["entries"]):
        if a["path"] != b["path"]:
            print("FAIL: thứ tự ảnh khác nhau")
            return 1
        pa, pb = np.asarray(a["probs"]), np.asarray(b["probs"])
        diffs.append(float(np.abs(pa - pb).max()))
        pred_mismatch += int(a["pred_id"] != b["pred_id"])
    max_diff = max(diffs)
    passed = max_diff < threshold and pred_mismatch == 0
    payload = {
        "left": left_path.as_posix(),
        "right": right_path.as_posix(),
        "n_images": left["n_images"],
        "max_probability_difference": max_diff,
        "threshold": threshold,
        "prediction_mismatches": pred_mismatch,
        "passed": passed,
    }
    (ROOT / "outputs/results/golden_compare.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{'PASS' if passed else 'FAIL'}: lệch xác suất lớn nhất {max_diff:.3e} (ngưỡng {threshold:g}), "
          f"lệch nhãn {pred_mismatch}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden test 20 ảnh (dump/compare)")
    sub = parser.add_subparsers(dest="command", required=True)
    dump_cmd = sub.add_parser("dump")
    dump_cmd.add_argument("--out", type=Path, required=True)
    dump_cmd.add_argument("--run", type=Path, default=ROOT / "outputs/runs/final_s42")
    dump_cmd.add_argument("--n", type=int, default=20)
    compare_cmd = sub.add_parser("compare")
    compare_cmd.add_argument("left", type=Path)
    compare_cmd.add_argument("right", type=Path)
    args = parser.parse_args()

    if args.command == "dump":
        dump(args.out, args.run, args.n)
        return 0
    return compare(args.left, args.right)


if __name__ == "__main__":
    raise SystemExit(main())
