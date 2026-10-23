"""Demo Streamlit: tải ảnh biển báo, xem top-3 và cảnh báo "không xác định".

Chạy::

    streamlit run app.py

- Tiền xử lý dùng đúng ``src.data.preprocess`` (không viết lại).
- Model: ``outputs/runs/final_s42/best.pt`` (có thể đổi bằng biến môi trường
  ``GTSRB_RUN_DIR`` để test).
- Ngưỡng τ: ``outputs/results/threshold.json`` (chọn trên val, không chỉnh theo demo).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from src.config import PROJECT_ROOT
from src.predict import Predictor, load_tau, validate_image_bytes

_DEFAULT_RUNS = (
    PROJECT_ROOT / "outputs" / "runs" / "final_s42",
    PROJECT_ROOT / "checkpoints" / "final_s42",
)
RUN_DIR = (
    Path(os.environ["GTSRB_RUN_DIR"])
    if os.environ.get("GTSRB_RUN_DIR")
    else next((path for path in _DEFAULT_RUNS if (path / "best.pt").is_file()), _DEFAULT_RUNS[0])
)
THRESHOLD_PATH = PROJECT_ROOT / "outputs" / "results" / "threshold.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "results" / "test_report.json"
MULTISEED_PATH = PROJECT_ROOT / "outputs" / "results" / "multiseed.json"
FOOTER = (
    "Demo học thuật trên GTSRB (biển báo Đức). **Không dùng cho quyết định giao thông thực tế** "
    "— mô hình có thể sai, đặc biệt với ảnh ngoài phân phối."
)
SAMPLES = {
    "Ảnh test GTSRB — biển 16 (cấm xe >3,5 t)": "external_test/ext_01_gtsrb_test_00000.png",
    "Ảnh test GTSRB — biển 50 km/h": "data/gtsrb/GTSRB/Final_Test/Images/00034.ppm",
    "Ảnh test GTSRB — biển STOP": "data/gtsrb/GTSRB/Final_Test/Images/00093.ppm",
    "Biển Hàn Quốc 30 km/h (đã crop)": "external_test/ext_03_kr_speed30_crop.jpg",
    "Cảnh đường không có biển": "external_test/ext_09_road_hills.jpg",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@st.cache_resource(show_spinner=False)
def get_predictor(run_dir: str, tau: float) -> Predictor:
    """Nạp model một lần cho cả phiên."""
    return Predictor(run_dir, tau=tau)


def sidebar(run_dir: Path, tau: float) -> None:
    report = read_json(REPORT_PATH)
    multiseed = read_json(MULTISEED_PATH)
    with st.sidebar:
        st.header("Thông tin mô hình")
        st.caption(f"Checkpoint: `{run_dir.name}`")
        if report:
            st.metric("Test accuracy", f"{report['accuracy'] * 100:.2f} %")
            st.metric("Test macro-F1", f"{report['macro_f1_43']:.4f}")
            st.metric("Top-3 accuracy", f"{report['top3_accuracy'] * 100:.2f} %")
            st.caption(f"sha256 `{report['checkpoint_sha256'][:16]}…`")
        if multiseed:
            st.caption(
                f"Multi-seed val: {multiseed['val_acc']['mean'] * 100:.2f} % ± "
                f"{multiseed['val_acc']['std'] * 100:.2f} điểm %"
            )
        st.divider()
        st.markdown(f"**Ngưỡng từ chối τ = {tau:.4f}**")
        st.caption("Chọn trên val: giữ ~95 % ảnh trong phân phối. Đây là ngưỡng theo confidence, "
                   "không phải bộ phát hiện OOD.")
        st.divider()
        st.caption("Ảnh nên crop quanh biển. Ảnh cảnh rộng có biển nhỏ thường bị từ chối hoặc sai.")
        st.caption(
            "Ảnh demo có sẵn: `outputs/demo_images/` — 16 ảnh PNG kèm nhãn trong "
            "`demo_manifest.csv` (tạo bằng `python tests/export_demo_images.py`)."
        )


def render_prediction(prediction: dict[str, Any], tau: float) -> None:
    top = prediction["topk"][0]
    if prediction["unknown"]:
        st.warning(
            f"**Không xác định** — độ tin cậy cao nhất {prediction['max_prob']:.3f} "
            f"< τ = {tau:.3f}. Ảnh có thể là biển **ngoài 43 lớp GTSRB** (ví dụ biển Việt Nam), "
            "hoặc quá mờ/tối, hoặc crop chưa sát biển."
        )
        st.caption(f"Lớp gần nhất nếu buộc phải chọn: **{top['name_vi']}** ({top['prob'] * 100:.1f} %).")
    else:
        st.success(f"Dự đoán: **{top['name_vi']}** ({top['prob'] * 100:.1f} %)")

    st.markdown("**Top-3 lớp khả năng cao nhất**")
    for rank, item in enumerate(prediction["topk"], start=1):
        st.markdown(f"**{rank}. {item['name_vi']}** — {item['prob'] * 100:.1f} %")
        st.progress(float(item["prob"]))
    st.caption(
        f"Margin top-1 − top-2: {prediction['margin']:.3f} · τ = {tau:.3f} · "
        f"lớp dự đoán: {top['class_id']}"
    )


def tab_predict(predictor: Predictor, tau: float) -> None:
    uploaded = st.file_uploader(
        "Tải ảnh biển báo (JPG/JPEG/PNG, ≤ 20 MB)",
        type=["jpg", "jpeg", "png"],
        help="Crop sát quanh biển, tránh watermark/chữ/QR. Ảnh lớn hơn 20 MB sẽ bị chặn.",
    )
    available = {key: value for key, value in SAMPLES.items() if (PROJECT_ROOT / value).is_file()}
    sample_name = st.selectbox("Hoặc chọn ảnh mẫu có sẵn", ["(không dùng ảnh mẫu)"] + list(available))
    image: Image.Image | None = None
    source = ""

    if uploaded is not None:
        try:
            image = validate_image_bytes(uploaded.name, uploaded.getvalue())
            source = uploaded.name
        except ValueError as exc:
            st.error(f"Không đọc được ảnh: {exc}")
            return
    elif sample_name in available:
        path = PROJECT_ROOT / available[sample_name]
        if path.is_file():
            image = Image.open(path).convert("RGB")
            source = path.name
        else:
            st.error(f"Thiếu ảnh mẫu: {path}")

    if image is None:
        st.info("Chưa có ảnh. Hãy tải lên hoặc chọn một ảnh mẫu để xem kết quả.")
        return

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.image(image, caption=f"Ảnh đầu vào: {source}", width="stretch")
    with right:
        prediction = predictor.predict(image, k=3, tau=tau)
        render_prediction(prediction, tau)


def tab_about(tau: float) -> None:
    report = read_json(REPORT_PATH)
    st.subheader("Mô hình và cách đánh giá")
    st.markdown(
        "- **Kiến trúc:** CNN-B tự xây dựng (3 khối conv 32/64/128 + BN + dropout 0,3), "
        "2.201.931 tham số, đầu vào 64×64 RGB.\n"
        "- **Huấn luyện:** split `stratified_group` (track train/val rời nhau, đủ 43 lớp), "
        "3 seed 42/123/2024, chọn checkpoint theo val macro-F1.\n"
        "- **Test:** đánh giá **đúng một lần** sau khi đóng băng `configs/final.yaml`."
    )
    if report:
        st.dataframe(
            {
                "Chỉ số test": ["Accuracy", "Macro-F1 (43 lớp)", "Top-3 accuracy", "Số ảnh"],
                "Giá trị": [
                    f"{report['accuracy'] * 100:.2f} %",
                    f"{report['macro_f1_43']:.4f}",
                    f"{report['top3_accuracy'] * 100:.2f} %",
                    f"{report['n_samples']:,}".replace(",", "."),
                ],
            },
            hide_index=True,
            width="stretch",
        )
    st.info(
        f"Ngưỡng τ = **{tau:.4f}** là cơ chế từ chối theo confidence. Trong thử nghiệm OOD, "
        "1/4 ảnh không có biển vẫn bị nhận nhầm với confidence cao — không nên xem τ là bộ lọc OOD."
    )
    with st.expander("Vì sao biển Việt Nam không phân loại được?"):
        st.markdown(
            "GTSRB chỉ có **43 lớp biển Đức** (giới hạn tốc độ, cấm vượt, ưu tiên, cảnh báo…). "
            "Biển Việt Nam như 'cấm dừng/đỗ', 'cấm rẽ trái' hay biển chữ không nằm trong 43 lớp này, "
            "nên không có nhãn đúng để dự đoán — kể cả khi nhìn giống biển Đức. "
            "Khi đó app nên trả 'không xác định'; nếu vẫn đoán, đó là **domain shift** đã nêu ở Chương 5 "
            "của báo cáo. Muốn nhận diện biển Việt Nam cần dữ liệu và mô hình riêng, không phải bài này."
        )
    with st.expander("43 lớp và hạn chế"):
        st.markdown(
            "Mô hình chỉ biết 43 lớp biển Đức. Biển nước khác, ảnh quá mờ/tối, biển quá nhỏ hoặc "
            "nhiều biển trong khung đều có thể cho kết quả sai. Xem Chương 5 của báo cáo "
            "(`docs/bao_cao/BAO_CAO.docx`) để biết 12 mẫu sai và nguyên nhân định lượng."
        )


def main() -> None:
    st.set_page_config(page_title="GTSRB CNN — demo", page_icon="🚸", layout="wide")
    st.title("Phân loại biển báo giao thông Đức (GTSRB)")
    st.caption("CNN tự xây dựng · 43 lớp · ảnh 64×64 · demo học thuật")
    st.info(
        "Mô hình chỉ biết **43 lớp biển ĐỨC (GTSRB)**. Biển Việt Nam hoặc thiết kế khác "
        "**không có lớp tương ứng** nên sẽ bị trả 'không xác định' hoặc đoán sai — "
        "đây là giới hạn đã biết, không phải lỗi app. Hãy thử ảnh mẫu GTSRB bên dưới."
    )

    if not (RUN_DIR / "best.pt").is_file():
        st.error(
            f"Không tìm thấy checkpoint: `{RUN_DIR / 'best.pt'}`. "
            "Chạy `python -m src.train --config configs/final.yaml` trước khi mở demo."
        )
        st.stop()
    if not THRESHOLD_PATH.is_file():
        st.error(
            "Chưa có ngưỡng τ (`outputs/results/threshold.json`). "
            "Chạy `python -m src.error_analysis --run outputs/runs/final_s42` trước."
        )
        st.stop()

    tau = load_tau(THRESHOLD_PATH)
    try:
        predictor = get_predictor(str(RUN_DIR), tau)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    sidebar(RUN_DIR, tau)
    predict_tab, about_tab = st.tabs(["Dự đoán", "Về mô hình"])
    with predict_tab:
        tab_predict(predictor, tau)
    with about_tab:
        tab_about(tau)

    st.divider()
    st.caption(FOOTER)


main()
