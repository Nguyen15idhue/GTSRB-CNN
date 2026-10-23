"""Đóng gói bộ demo đầy đủ (A+): model + báo cáo + ảnh mẫu, không cần data/train.

Chạy::

    python tests/package_demo.py

Ra:
    outputs/GTSRB_demo_bundle/          # thư mục copy sang máy mới
    outputs/GTSRB_demo_bundle.zip       # file nén (kèm báo cáo + ảnh demo)
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT = ROOT / "outputs" / "GTSRB_demo_bundle"
ZIP = ROOT / "outputs" / "GTSRB_demo_bundle.zip"

# File bắt buộc để app chạy
REQUIRED = {
    "app.py": "app.py",
    ".streamlit/config.toml": ".streamlit/config.toml",
    "requirements-demo.txt": "requirements-demo.txt",
    "README.md": "README.md",
    "outputs/results/threshold.json": "outputs/results/threshold.json",
    "outputs/results/test_report.json": "outputs/results/test_report.json",
    "outputs/results/multiseed.json": "outputs/results/multiseed.json",
    "outputs/runs/final_s42/best.pt": "outputs/runs/final_s42/best.pt",
    "configs/class_names.json": "configs/class_names.json",
}

# Báo cáo + tài liệu
EXTRAS = {
    "docs/bao_cao/BAO_CAO.docx": "docs/bao_cao/BAO_CAO.docx",
    "docs/bao_cao/BAO_CAO.md": "docs/bao_cao/BAO_CAO.md",
    "docs/DEMO_DATA.md": "docs/DEMO_DATA.md",
    "docs/HUONG_DAN_CHAY_MAY_MOI.md": "docs/HUONG_DAN_CHAY_MAY_MOI.md",
}

# Ảnh để app chọn sẵn (đúng đường dẫn SAMPLES trong app.py)
SAMPLE_FILES = {
    "external_test/ext_01_gtsrb_test_00000.png": "external_test/ext_01_gtsrb_test_00000.png",
    "external_test/ext_03_kr_speed30_crop.jpg": "external_test/ext_03_kr_speed30_crop.jpg",
    "external_test/ext_09_road_hills.jpg": "external_test/ext_09_road_hills.jpg",
    "data/gtsrb/GTSRB/Final_Test/Images/00034.ppm": "data/gtsrb/GTSRB/Final_Test/Images/00034.ppm",
    "data/gtsrb/GTSRB/Final_Test/Images/00093.ppm": "data/gtsrb/GTSRB/Final_Test/Images/00093.ppm",
    "external_test/manifest.csv": "external_test/manifest.csv",
}

DEMO_IMAGES = ROOT / "outputs" / "demo_images"

GUIDE = """HƯỚNG DẪN DEMO TRÊN MÁY MỚI (A+)
==================================
Không cần train. Không cần dataset. Không cần GPU.

Yêu cầu: Python 3.12 (64-bit).

Bước 1 - tạo môi trường tại thư mục này:
    py -3.12 -m venv .venv
    Windows:   .\\.venv\\Scripts\\Activate.ps1
    Linux/Mac: source .venv/bin/activate

Bước 2 - cài thư viện (torch CPU là đủ, ~1 GB):
    python -m pip install --upgrade pip
    pip install -r requirements-demo.txt

Bước 3 - chạy:
    streamlit run app.py
    (lần đầu hỏi Email: nhấn Enter; mở http://localhost:8501)

Trong app:
- Tab "Dự đoán": upload ảnh hoặc chọn 5 ảnh mẫu có sẵn.
- Tab "Về mô hình": chỉ số test + giải thích.

Dữ liệu kèm theo:
    outputs/demo_images/*.png       16 ảnh GTSRB (8 val + 8 test) + demo_manifest.csv
    external_test/                  3 ảnh mẫu của app + manifest
    data/gtsrb/.../00034.ppm        2 ảnh test dùng làm mẫu (50 km/h, STOP)
    docs/bao_cao/BAO_CAO.docx       báo cáo đầy đủ (6 chương, 19 bảng, 21 hình)

Kỳ vọng:
    Sidebar: Test accuracy 98.56 %, macro-F1 0.9799, top-3 99.64 %, τ = 0.9462.
    Ảnh biển Đức (16, 50 km/h, STOP) -> đúng; ảnh đường không biển -> "Không xác định".
    Biển Việt Nam -> "Không xác định" hoặc sai (mô hình chỉ biết 43 lớp biển ĐỨC).

Lưu ý:
- Không chạy src.train / src.data / evaluate trong bundle (thiếu dataset).
- Muốn dùng checkpoint khác: đặt GTSRB_RUN_DIR trỏ thư mục có best.pt.
"""


def copy_file(src: str, dst: str) -> None:
    target = OUT / dst
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / src, target)


def main() -> int:
    missing = [src for src in {**REQUIRED, **EXTRAS, **SAMPLE_FILES} if not (ROOT / src).is_file()]
    if missing:
        print("Thiếu file, không đóng gói được:")
        for name in missing:
            print("  -", name)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    for folder in (".streamlit", "configs", "external_test", "docs/bao_cao",
                   "data/gtsrb/GTSRB/Final_Test/Images", "outputs/results",
                   "outputs/runs/final_s42"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    shutil.copytree(ROOT / "src", OUT / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for src in list(REQUIRED) + list(EXTRAS) + list(SAMPLE_FILES):
        copy_file(src, {**REQUIRED, **EXTRAS, **SAMPLE_FILES}[src])

    if DEMO_IMAGES.is_dir():
        shutil.copytree(
            DEMO_IMAGES,
            OUT / "outputs/demo_images",
            ignore=shutil.ignore_patterns("*.ppm"),
        )

    (OUT / "HUONG_DAN.txt").write_text(GUIDE, encoding="utf-8", newline="\n")

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUT).as_posix())

    size_mb = ZIP.stat().st_size / 1024**2
    print(f"Đã tạo {OUT.relative_to(ROOT)}")
    print(f"Đã nén  {ZIP.relative_to(ROOT)} ({size_mb:.1f} MB)")
    print("Nội dung: model + báo cáo DOCX + 16 ảnh demo + 5 ảnh mẫu + hướng dẫn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
