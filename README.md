# GTSRB-CNN — Phân loại 43 lớp biển báo giao thông Đức

Bài tập lớn Học sâu (12INF028). CNN **tự xây dựng** (không pretrained) cho GTSRB 64×64,
kèm baseline, ablation, đánh giá test **một lần**, phân tích lỗi và demo Streamlit.

## 1. Kết quả chính (số thật, lấy từ `outputs/results/`)

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Mô hình cuối | **E3 — CNN-B** (3 khối conv 32/64/128 + BN + dropout 0,3 + WD 1e-4 + aug), 2.201.931 params | `configs/final.yaml` |
| Split | `split_stratified_group`: **31.371 train / 7.830 val** (track rời nhau, đủ 43 lớp), test **12.630** | `outputs/results/split_stratified_group.json` |
| Val multi-seed (42/123/2024) | accuracy **0,980843 ± 0,003052** · macro-F1 **0,965392 ± 0,004689** | `multiseed.json` |
| **Test (đánh giá đúng 1 lần)** | accuracy **0,985590** · macro-F1 **0,979894** · top-3 **0,996437** | `test_report.json`, `test_final.json` |
| Baseline | majority 0,0574 / 0,0025 · HOG+LinearSVC 0,9657 / 0,9740 (val) | `summary.csv` |
| Phân tích lỗi | 1 lớp F1 < 0,90 (lớp 27 = 0,846, support 60); top cặp nhầm 27→11 (15 ảnh) | `error_report.json` |
| Demo | 5 kịch bản; τ = 0,9462 chọn trên val; model-only 3,02 ms, end-to-end Python 5,78 ms | `threshold.json`, `latency_effect.json` |

> **Cảnh báo đọc số:** val **random** (dùng ở Giai đoạn 5) bị **rò rỉ frame**: 95,22 % ảnh val
> có frame train cách ±1 trong cùng track → val random đạt ~99,9 % và **không dùng để xếp hạng
> mô hình**. Mọi kết luận cuối dùng **val stratified-group** và **test**. Chi tiết: mục 5.8 của
> `docs/kehoach/Cac_buoc_thuc_hien_GTSRB_CNN.md` và `outputs/results/leak_delta.json`.

## 2. Yêu cầu hệ thống

- Windows 10/11 + PowerShell, Python **3.12 64-bit**.
- GPU NVIDIA (đã kiểm trên RTX 5070, Blackwell capability 12.0) cho train/demo; eval CPU vẫn chạy.
- RAM ≥ 16 GB (khuyến nghị 32 GB), dung lượng ≥ 8 GB cho dữ liệu + cache + checkpoint.
- Không bắt buộc GPU để chạy **test suite** (`pytest`) và đọc artifact.

## 3. Cài đặt

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

# BẮT BUỘC: cài torch/torchvision từ index cu128 trước (bản PyPI mặc định là CPU)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# Sau đó cài phần còn lại (pip coi torch 2.11.0+cu128 là thoả mãn ==2.11.0)
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

Kỳ vọng: `2.11.0+cu128 True (12, 0)`. Danh sách phiên bản đầy đủ: `requirements.txt`
(và `requirements-lock.txt` là bản `pip freeze` tại máy chạy thật). Chỉ cần suy luận/demo:
`pip install -r requirements-demo.txt` (vẫn phải cài torch cu128 trước).

## 4. Dữ liệu

```powershell
python -m src.data --download --report   # tải GTSRB, xác minh, EDA
python -m src.data --prepare             # cache .npy 64×64 + split + mean/std + hình augmentation
python -m src.data --verify              # checklist Giai đoạn 2 (12 mục)
```

- Train **39.209** ảnh (43 lớp) và test **12.630** ảnh; sau khi loại 8 ảnh train trùng MD5 với
  test còn **39.201** ảnh dùng để chia split (`usable_for_train` trong `train_meta.csv`).
- Cache `data/cache/*.npy` (uint8, ~610 MB) và `data/` **không** được commit.
- Split lưu sẵn: `split_random.json`, `split_group.json`, `split_stratified_group.json`
  (không sinh lại mỗi run).
- `mean/std` lấy từ `outputs/results/data_stats.json` (tính trên train của split random), dùng
  chung cho train/eval/demo để chỉ có **một** pipeline tiền xử lý.

## 5. Train, đánh giá, demo

```powershell
# Tái lập mô hình cuối (3 seed, chạy tuần tự)
python -m src.train --config configs/final.yaml --seed 42
python -m src.train --config configs/final.yaml --seed 123
python -m src.train --config configs/final.yaml --seed 2024
python -m src.summarize --multiseed

# Đánh giá validation (an toàn, chạy nhiều lần)
python -m src.evaluate --run outputs/runs/final_s42 --split val

# Đánh giá test: CHỈ MỘT LẦN, sau khi configs/final.yaml đã đóng băng
python -m src.evaluate --run outputs/runs/final_s42 --split test --confirm-freeze

# Tổng hợp / phân tích lỗi / demo
python -m src.summarize --figure
python -m src.error_analysis --run outputs/runs/final_s42
streamlit run app.py
```

- **Test đã được đánh giá đúng 1 lần** trong repo này (`test_runs.log` có 1 dòng). Chạy lại sẽ
  bị từ chối trước khi ghi file; nếu buộc phải chạy lại do bug, dùng
  `--allow-rerun --reason "..."` → sinh `test_report_v2.json`, **không ghi đè** bản gốc.
- `outputs/runs/` bị `.gitignore`; bản sao `predictions.csv` của lần test duy nhất nằm ở
  `outputs/results/predictions_test_final.csv` (kèm sha256 trong
  `predictions_test_final.sha256.json`).

## 6. Cấu trúc

```
src/            config, data, models, baseline, train, evaluate,
                error_analysis, predict, summarize
configs/        default/e*/final YAML + class_names.json
tests/          pytest cho split, metric, checkpoint, upload, artifact
outputs/figures/  01…10: EDA, augmentation, so sánh, lỗi, OOD, pHash, báo cáo
outputs/results/  summary.csv, multiseed.json, test_report.json, test_final.json,
                  predictions_test_final.csv, threshold.json, error_report.json,
                  leak_delta.json, phash_*.json, phase*_check.json
external_test/  10 ảnh CC0/PD + 1 ảnh GTSRB tham chiếu + manifest.csv
docs/kehoach/   kế hoạch + nhật ký 13 giai đoạn
docs/bao_cao/   báo cáo Markdown + DOCX
app.py          demo Streamlit
```

## 7. Tái lập kết quả

- Seed cố định 42/123/2024; config đã đóng băng; split lưu JSON; `env.json` mỗi run ghi phiên bản.
- Kiểm nhanh: `python -m pytest tests -q` (11 test, ~3 s).
- Kiểm tái lập eval: chạy `src.evaluate --split val` trên checkpoint có sẵn, so với
  `multiseed.json` — phải khớp (sai khác < 0,2 điểm %).
- Clean-room (môi trường mới): xem `outputs/results/cleanroom.json` và mục 9.6 của
  `docs/kehoach/Cac_buoc_thuc_hien_GTSRB_CNN.md`.
- **Không** tái tạo test nếu không thực sự cần: test chỉ có giá trị khi chỉ dùng một lần.

## 8. Hạn chế và cảnh báo sử dụng

- Mô hình chỉ biết **43 lớp biển Đức**. Ảnh biển nước khác (Mỹ/Hàn/Philippines) là **domain
  shift**: có ca đúng nhãn tương ứng, có ca bị gán nhãn Đức với confidence cao.
- Ngưỡng τ = 0,9462 là **cơ chế từ chối theo confidence**, **không phải** bộ phát hiện OOD:
  vẫn có 1/4 ảnh không biển bị nhận nhầm với confidence ~1,00.
- Val random bị rò rỉ frame (95,22 % có frame ±1); chỉ dùng val **stratified-group** để kết luận.
- Audit pHash: ngoài 8 cặp MD5 đã loại, không phát hiện cặp gần trùng được xác nhận; rủi ro còn
  lại ước lượng ≤ 0,1 điểm % accuracy (`outputs/results/phash_highres.json`).
- Chưa có screenshot UI/ video demo; chưa làm slide. Báo cáo nằm ở `docs/bao_cao/`.
- **Không dùng cho quyết định giao thông thực tế.** Đây là demo học thuật.

## 9. Tài liệu

- Kế hoạch chủ: `docs/kehoach/KE_HOACH_TONG_QUAN_GTSRB_CNN.md`
- Các bước thực hiện + nhật ký: `docs/kehoach/Cac_buoc_thuc_hien_GTSRB_CNN.md`
- Báo cáo: `docs/bao_cao/BAO_CAO.md` (và `.docx` nếu đã xuất)
- Quy tắc làm việc: `AGENTS.md`
