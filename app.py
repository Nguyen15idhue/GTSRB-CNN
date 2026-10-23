"""Demo Streamlit: tải ảnh lên, hiển thị top-3 và cảnh báo "không xác định".

Chạy::

    streamlit run app.py

Toàn bộ tiền xử lý dùng ``src.data.preprocess`` (không viết lại). Model lấy từ
checkpoint cuối ``outputs/runs/final_s42/best.pt``; ngưỡng τ lấy từ
``outputs/results/threshold.json`` (chọn trên val, không chọn trên demo).
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import FIGURES_DIR, PROJECT_ROOT
from src.predict import Predictor, load_tau, validate_image_bytes

FINAL_RUN = PROJECT_ROOT / "outputs" / "runs" / "final_s42"
THRESHOLD_PATH = PROJECT_ROOT / "outputs" / "results" / "threshold.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "results" / "test_report.json"
FOOTER = (
    "Demo học thuật trên GTSRB (biển báo Đức). **Không dùng cho quyết định giao thông thực tế** "
    "— mô hình có thể sai, đặc biệt với ảnh ngoài phân phối."
)


@st.cache_resource(show_spinner=False)
def get_predictor(tau: float) -> Predictor:
    """Nạp model một lần cho cả phiên (cache_resource)."""
    return Predictor(FINAL_RUN, tau=tau)


def show_header() -> None:
    st.title("Phân loại biển báo giao thông Đức (GTSRB)")
    st.caption("CNN tự xây dựng (CNN-B) · 43 lớp · ảnh đầu vào 64×64 · demo học thuật")


def show_metrics() -> None:
    """Hiển thị số liệu test đã đánh giá (đọc từ test_report.json, không hard-code)."""
    if not REPORT_PATH.is_file():
        return
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    columns = st.columns(3)
    columns[0].metric("Test accuracy", f"{report['accuracy'] * 100:.2f} %")
    columns[1].metric("Test macro-F1", f"{report['macro_f1_43']:.4f}")
    columns[2].metric("Top-3 accuracy", f"{report['top3_accuracy'] * 100:.2f} %")
    st.caption(f"Checkpoint: `{report['run_id']}` · sha256 `{report['checkpoint_sha256'][:16]}…`")


def show_prediction(prediction: dict, tau: float) -> None:
    """Hiển thị top-3 (xác suất giảm dần) và cảnh báo nếu dưới ngưỡng τ."""
    topk = prediction["topk"]
    if prediction["unknown"]:
        st.warning(
            f"**Không xác định** — độ tin cậy cao nhất {prediction['max_prob']:.3f} "
            f"< ngưỡng τ = {tau:.3f}. Ảnh có thể không chứa biển báo GTSRB hoặc quá mờ/tối."
        )
    else:
        st.success(f"Dự đoán: **{topk[0]['name_vi']}** ({topk[0]['prob'] * 100:.1f} %)")
    st.write("**Top-3:**")
    for rank, item in enumerate(topk, start=1):
        st.progress(min(1.0, item["prob"]), text=f"{rank}. {item['name_vi']} — {item['prob'] * 100:.1f} %")
    st.caption(f"Margin top-1 − top-2: {prediction['margin']:.3f} · τ = {tau:.3f}")


def main() -> None:
    st.set_page_config(page_title="GTSRB CNN demo", page_icon="🚸", layout="centered")
    show_header()
    show_metrics()

    if not FINAL_RUN.joinpath("best.pt").is_file():
        st.error(
            "Không tìm thấy checkpoint: `outputs/runs/final_s42/best.pt`. "
            "Chạy `python -m src.train --config configs/final.yaml` trước khi mở demo."
        )
        st.stop()
    if not THRESHOLD_PATH.is_file():
        st.error(
            "Chưa có ngưỡng τ (`outputs/results/threshold.json`). "
            "Chạy `python -m src.error_analysis --run outputs/runs/final_s42` trước."
        )
        st.stop()

    tau = load_tau()
    try:
        predictor = get_predictor(tau)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    uploaded = st.file_uploader("Tải ảnh biển báo (JPG/JPEG/PNG, ≤ 20 MB)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Chưa có ảnh. Hãy tải lên một ảnh biển báo đã crop để xem kết quả.")
    else:
        try:
            image = validate_image_bytes(uploaded.name, uploaded.getvalue())
        except ValueError as exc:
            st.error(f"Không đọc được ảnh: {exc}")
            st.stop()
        st.image(image, caption=f"Ảnh tải lên: {uploaded.name}", use_container_width=True)
        prediction = predictor.predict(image, k=3, tau=tau)
        show_prediction(prediction, tau)

    with st.expander("Cách demo và giới hạn"):
        st.markdown(
            "- Ảnh nên **crop quanh biển**; ảnh cảnh rộng có biển nhỏ thường bị từ chối hoặc sai.\n"
            "- Model chỉ biết **43 lớp biển Đức**; biển nước khác có thể bị gán nhãn sai.\n"
            "- Ngưỡng τ chọn trên **val**, không chỉnh theo ảnh demo.\n"
            "- Ảnh minh chứng và số liệu: `outputs/figures/`, `outputs/results/`."
        )
        sample = FIGURES_DIR / "07_error_samples.png"
        if sample.is_file():
            st.image(str(sample), caption="Ví dụ các mẫu model dự đoán sai trên test (Giai đoạn 7)")

    st.divider()
    st.caption(FOOTER)


main()
