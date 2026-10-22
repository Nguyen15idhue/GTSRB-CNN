# GTSRB-CNN

Bài tập lớn Học sâu (12INF028): phân loại **43 lớp biển báo giao thông Đức (GTSRB)** bằng CNN tự xây dựng.

> **Trạng thái:** Giai đoạn 0 đã xong (môi trường + cấu trúc repo). Chưa có kết quả thực nghiệm.
> Mọi số liệu trong báo cáo sẽ được sinh từ lần chạy thật trong `outputs/results/`.

## Yêu cầu

- Windows 10/11, Python 3.12 64-bit
- GPU NVIDIA RTX 5070 (Blackwell, capability 12.0)
- PyTorch bản **cu128** trở lên

## Cài đặt

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Kiểm tra môi trường:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

Kỳ vọng: `2.11.0+cu128 True (12, 0)`.

## Cấu trúc

```
src/            # config, data, models, baseline, train, evaluate, error_analysis,
                # predict, summarize
configs/        # YAML cho từng thí nghiệm + class_names.json
outputs/runs/   # mỗi run: config.yaml, env.json, metrics.csv, best.pt, curves.png
outputs/results/# summary.csv, test_final.json, env.json
external_test/  # ảnh tự chụp đã ẩn danh + manifest.csv
app.py          # demo Streamlit
```

## Tài liệu

- Kế hoạch chủ: `docs/kehoach/KE_HOACH_TONG_QUAN_GTSRB_CNN.md`
- Các bước thực hiện: `docs/kehoach/Cac_buoc_thuc_hien_GTSRB_CNN.md`
- Demo trên máy yếu: `docs/kehoach/Phan_tich_demo_tren_may_yeu.md`
- Quy tắc làm việc: `AGENTS.md`

## Ghi chú

- Không commit `data/`, `.venv/`, `outputs/runs/`, `*.pt`, `*.npy`.
- Tập test chỉ được đánh giá một lần với `configs/final.yaml` đã đóng băng.
- Mô hình chỉ phục vụ mục đích học tập, không dùng cho quyết định giao thông thực tế.
