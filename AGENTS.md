# AGENTS.md

Hướng dẫn cho AI agent (và người) khi làm việc trong repo này.
Đọc file này **trước khi** sửa code, chạy thí nghiệm hoặc viết báo cáo.

| Thông tin | Giá trị |
|---|---|
| Dự án | Bài tập lớn Học sâu 12INF028 — Phân loại biển báo GTSRB bằng CNN |
| Ngôn ngữ tài liệu | Tiếng Việt |
| Ngôn ngữ code | Tên biến/hàm bằng tiếng Anh; comment/docstring tiếng Việt được phép |
| Cập nhật lần cuối | 23/10/2026 |
| Trạng thái repo | **Giai đoạn 10 hoàn tất (23/10/2026)**: GĐ1–10 xong. GĐ6 chốt E3 trên `split_stratified_group`, multi-seed, **test 1 lần: acc 0,985590 · macro-F1 0,979894** (xác minh 14/14). GĐ9: clean-room `.venv_test` (CUDA (12,0), val khớp tuyệt đối, golden test 0,000e+00, pytest 11). GĐ10: `docs/bao_cao/BAO_CAO.{md,docx}` (6 chương, 19 bảng, 20 hình). **GĐ11–12 chưa làm** (slide/video/đóng gói/nộp) theo yêu cầu. |

---

## 1. Bối cảnh và mục tiêu

Xây dựng pipeline học sâu hoàn chỉnh cho bài toán phân loại **43 lớp biển báo giao thông Đức (GTSRB)**: dữ liệu → baseline → CNN tự xây dựng → ablation → đánh giá → phân tích lỗi → demo → báo cáo.

Điểm quan trọng nhất: **mọi con số trong báo cáo phải sinh ra từ lần chạy thật**, không có số liệu minh họa hay "điền sau".

---

## 2. Thông tin cốt lõi (không tự ý thay đổi)

| Hằng số | Giá trị | Nơi định nghĩa |
|---|---|---|
| `NUM_CLASSES` | **43** | `src/config.py`, `configs/*.yaml` |
| `IMG_SIZE` | **64** | `src/config.py` |
| `SEED` mặc định | **42** | `configs/default.yaml` |
| Multi-seed | 42, 123, 2024 | `configs/final.yaml` |
| Train / Val / Test | 31.360 / 7.841 / **12.630** | `outputs/results/split_*.json` |
| Split | stratified 80/20 + biến thể group theo track | `src/data.py` |
| Checkpoint chọn | theo **val macro-F1** | `src/train.py` |
| Metric chính | Accuracy, **Macro-F1**, Confusion matrix | `src/evaluate.py` |
| Ngưỡng unknown | chọn trên **val**, không chọn trên test/demo | `src/predict.py` |

Thay đổi bất kỳ hằng số nào ở trên phải: ghi lý do, cập nhật tài liệu trong `docs/kehoach/`, và chạy lại các thí nghiệm bị ảnh hưởng.

> **Lý do Train/Val = 31.360 / 7.841 (cập nhật 22/10/2026).** GTSRB có **8 ảnh train trùng MD5** với 8 ảnh test (đều nhãn 14; 8 ảnh train này thuộc **track `00023` của tập train**, còn file test như `02291.ppm` **không có metadata track**). Đã **loại 8 ảnh đó khỏi tập huấn luyện** để test độc lập hoàn toàn: `39.209 − 8 = 39.201` ảnh dùng chia split, 80/20 stratified (seed 42) → **31.360 / 7.841**. File dữ liệu vẫn giữ nguyên trên đĩa (39.209) để số đếm khớp công bố; danh sách loại trừ ở `outputs/results/excluded_train_duplicates.csv` và cột `usable_for_train` trong `train_meta.csv`.

---

## 3. Môi trường

| Thành phần | Giá trị |
|---|---|
| OS | Windows 10/11, PowerShell |
| Python | 3.12, 64-bit, trong `.venv` |
| PyTorch | bản **cu128** trở lên (RTX 5070 = Blackwell, capability 12.0) |
| GPU | NVIDIA RTX 5070 12 GB |
| CPU / RAM | i9-14900KF / 64 GB |

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Quy tắc môi trường**
- Không cài `torch` từ PyPI mặc định (dễ ra bản CPU).
- Mọi lệnh chạy trong `.venv` đã activate.
- Thêm thư viện mới → cập nhật `requirements.txt` trong cùng commit.
- Không đổi phiên bản Python giữa chừng nếu chưa chạy lại smoke test.

---

## 4. Cấu trúc repo

```
.
├── AGENTS.md                  # file này
├── docs/kehoach/              # kế hoạch + tài liệu thi hành (nguồn sự thật)
├── data/                      # KHÔNG commit
├── src/
│   ├── config.py              # đọc/validate YAML, hằng số
│   ├── data.py                # load, split, cache, preprocess, dataloader
│   ├── models.py              # SmallCNN (A/B/C), build_model
│   ├── baseline.py            # majority, HOG + LinearSVC
│   ├── train.py               # train_one
│   ├── evaluate.py            # metric, predictions.csv, confusion matrix
│   ├── error_analysis.py      # bucket lỗi, top cặp nhầm, lưới ảnh
│   ├── predict.py             # load_model, predict_topk
│   ├── preprocess.py          # (nếu tách) chỉ tiền xử lý, không import datasets
│   └── summarize.py           # sinh summary.csv
├── configs/
│   ├── default.yaml
│   ├── e*.yaml                # từng thí nghiệm
│   ├── final.yaml             # config đã đóng băng
│   └── class_names.json       # 43 lớp: en, vi, shape, color
├── notebooks/                 # EDA, hình, phân tích (không chứa logic train)
├── outputs/
│   ├── runs/<exp>_s<seed>/    # config, metrics.csv, best.pt, curves.png
│   ├── figures/
│   └── results/               # summary.csv, test_final.json, data_report.json
├── external_test/             # ảnh tự chụp đã ẩn danh + manifest.csv
├── app.py                     # Streamlit demo
├── requirements.txt
├── requirements-demo.txt      # chỉ thư viện cho suy luận
└── README.md
```

**Quy tắc cấu trúc**
- Logic train/eval/demo **không** nằm trong notebook; notebook chỉ gọi hàm từ `src/`.
- `app.py` import `preprocess` từ `src/`, tuyệt đối không viết lại.
- Không tạo file rời ở root ngoài các file đã liệt kê.

---

## 5. Hợp đồng dữ liệu

- Nguồn: `torchvision.datasets.GTSRB`, `root="data"`, split `train`/`test`.
- Ảnh gốc kích thước thay đổi → luôn `convert("RGB")` rồi resize.
- `mean`/`std` tính **trên train** sau khi chia; lưu vào config, không hard-code.
- Cache: `uint8` shape `(N, 64, 64, 3)` trong `.npy`, không commit. **`X_train.npy` là pool 39.201 ảnh** dùng để chia split (chưa phải tập train 31.360); danh sách train/val thật nằm trong `split_random.json`.
- Augmentation **chỉ** cho train: affine nhẹ, color jitter nhẹ, perspective nhẹ.
- **Cấm** `HorizontalFlip`, `VerticalFlip`, xoay 90/180 (đổi nghĩa biển).
- Split lưu ra JSON (`split_random.json`, `split_group.json`, `split_stratified_group.json`) và tái sử dụng; không sinh lại mỗi run. `split_stratified_group` là split chính của Giai đoạn 6 (track rời nhau **và** đủ 43 lớp ở val: 31.371/7.830).
- Track ID (nếu trích được) dùng cho `split_group` và phân tích rò rỉ.

---

## 6. Hợp đồng mô hình và checkpoint

- Mô hình **tự xây dựng** (`SmallCNN`); không dùng pretrained cho mô hình chính.
- Forward: input `(B, 3, 64, 64)` → output `(B, 43)` **logits**, không softmax trong model.
- Checkpoint `best.pt` là dict:

```python
{
    "state_dict": model.state_dict(),
    "config": {...},              # kiến trúc + image_size + mean/std
    "class_names": {...},         # mapping 43 lớp
    "epoch": int,
    "val_macro_f1": float,
    "git_commit": str,
    "torch_version": str,
}
```

- Load: `torch.load(path, map_location="cpu", weights_only=True)` rồi `load_state_dict`.
- Không pickle cả object model (rủi ro lệch phiên bản).
- Không đổi kiến trúc mà không đổi `experiment.id`.

---

## 7. Quy ước thí nghiệm

**Tên run:** `<exp_id>_<model>_<feature>_s<seed>` — ví dụ `e3_cnnb_aug_s42`.

**Mỗi run phải có:**

```
outputs/runs/<run_id>/
├── config.yaml
├── env.json
├── metrics.csv
├── best.pt
└── curves.png
```

**Schema `metrics.csv` (đúng thứ tự cột):**

```
epoch,lr,train_loss,train_acc,val_loss,val_acc,val_macro_f1,epoch_time_s,gpu_mem_mb
```

**Quy tắc**
- Ghi 1 dòng/epoch, flush ngay; không ghi đè run cũ.
- Early stopping patience 7 theo `val_macro_f1`; tối đa 40 epoch.
- Mỗi run chỉ thay **một nhóm yếu tố** so với run gốc; ghi rõ trong `notes`.
- Cùng seed, cùng split, cùng early stopping khi so sánh.
- Chênh lệch < 0,3 điểm % → coi là tương đương, không tuyên bố "tốt hơn".
- Số liệu tổng hợp chỉ lấy từ `outputs/results/summary.csv`.

---

## 8. Quy tắc code

- **Config-driven**: không hard-code hyperparameter trong vòng train; đọc từ YAML.
- **Device**: dùng `get_device()`; không gọi `.cuda()` trực tiếp. Hỗ trợ `--device cpu`.
- **Eval**: luôn `model.eval()` + `torch.no_grad()`; không quên tắt dropout/BN train mode.
- **AMP**: chỉ bật khi `device.type == "cuda"`.
- **Windows**: `num_workers > 0` phải có `if __name__ == "__main__":`.
- **Đường dẫn**: `pathlib.Path`, tương đối từ root repo; không hard-code `C:\`.
- **Seed**: gọi `set_seed()` ở đầu mọi entrypoint; seed cả DataLoader generator.
- **Log**: dùng `logging`, không `print` trong thư viện `src/`.
- **Lỗi**: raise exception có message rõ; không `except: pass`.
- **Kiểu dữ liệu**: type hint cho hàm public.
- **Độ dài hàm**: giữ < ~80 dòng; tách hàm khi vượt.

---

## 9. Quy tắc đánh giá và số liệu

- Test chỉ được đánh giá **một lần** sau khi `configs/final.yaml` được đóng băng.
- Lệnh test phải có cờ `--confirm-freeze` và ghi log.
- Không dùng test để chọn mô hình, LR, epoch, ngưỡng.
- Accuracy không đủ; luôn báo cáo kèm **macro-F1** và per-class.
- **Macro-F1 luôn tính trên đủ 43 lớp** (`labels=range(43)`, `zero_division=0`). Nếu split thiếu lớp phải báo kèm `macro_f1_present_classes`, support từng lớp và trần lý thuyết (`số lớp có mặt / 43`) — không so sánh trực tiếp F1/43 giữa hai split khác support.
- Khi so sánh nhóm (tertile, split…), báo cáo **cả effect size** (r², chênh điểm % kèm CI95), không chỉ p-value; n nhỏ (OOD 3–6 ảnh) phải ghi CI và không kết luận tỷ lệ chính xác.
- Thời gian suy luận phải ghi rõ phạm vi đo (model-only vs end-to-end trong Python vs thời gian UI); không gọi model-only là "thời gian phản hồi app".
- Audit trùng/gần trùng: MD5 chỉ bắt trùng tuyệt đối; dùng pHash/MAD/NCC để **lọc ứng viên**, kiểm chứng ở ảnh gốc và xem ảnh trước khi kết luận — **không tự động xóa** ảnh theo ngưỡng hash.
- Confusion matrix chuẩn hóa theo hàng; ghi rõ nếu không chuẩn hóa.
- Mọi bảng kết quả trong báo cáo/slide phải khớp `summary.csv`, `test_final.json`, `multiseed.json`.
- Nếu chạy lại test do bug: tạo file `test_final_v2.json` và ghi lý do, không ghi đè.

---

## 10. Quy tắc demo và dữ liệu ngoài

- Demo dùng đúng checkpoint cuối và đúng `preprocess` của train.
- Có ngưỡng "không xác định" chọn trên val.
- Ảnh tự chụp: crop quanh biển; che mặt người và biển số xe; xóa EXIF.
- `external_test/manifest.csv` ghi nguồn, ngày, mô tả, đã ẩn danh.
- Không tuyên bố mô hình dùng được cho quyết định giao thông thực tế.
- Nếu demo trên máy khác: chạy golden test 20 ảnh, sai khác xác suất < 1e-3.

---

## 11. Git

- Branch chính: `main`.
- Commit message theo dạng: `feat:`, `fix:`, `docs:`, `exp:`, `chore:` + mô tả ngắn.
- Ví dụ: `feat: add GTSRB dataloader with cache`, `exp: run e3 with augmentation`.
- **Chỉ commit/push khi được yêu cầu rõ ràng.**
- Không commit: `data/`, `.venv/`, `outputs/runs/`, `*.pt`, `*.npy`, `.env`.
- Không force push, không rewrite lịch sử đã push.
- Trước khi commit: chạy smoke test liên quan.

---

## 12. Tài liệu

Nguồn sự thật nằm trong `docs/kehoach/`:

| File | Vai trò |
|---|---|
| `KE_HOACH_TONG_QUAN_GTSRB_CNN.md` | Kế hoạch chủ: phạm vi, thiết kế, thí nghiệm, tiêu chí |
| `Cac_buoc_thuc_hien_GTSRB_CNN.md` | 13 giai đoạn thi hành + checklist test |
| `Phan_tich_demo_tren_may_yeu.md` | Đóng gói và demo trên máy cấu hình thấp |
| `*.docx` | Bản kế hoạch ban đầu, chỉ để tham khảo |

**Quy tắc**
- Thay đổi phạm vi/kỹ thuật → cập nhật file kế hoạch tương ứng trong cùng commit.
- Không sửa `.docx` cũ; ghi thay đổi vào file `.md`.
- Mọi tài liệu mới đặt trong `docs/kehoach/` hoặc `docs/`.

---

## 13. Những điều cấm

1. **Không bịa số liệu** hoặc đưa số dự kiến vào báo cáo như kết quả thật.
2. **Không tune trên test** dưới mọi hình thức.
3. **Không lật ảnh** biển báo theo chiều ngang/dọc.
4. **Không commit dữ liệu, checkpoint, venv** hoặc thông tin cá nhân.
5. **Không viết lại preprocessing** trong app/notebook.
6. **Không đổi mapping nhãn** mà không cập nhật `class_names.json` và test lại.
7. **Không thêm dependency** mà không cập nhật `requirements.txt`.
8. **Không xóa/sửa** công việc có sẵn của người khác khi chưa kiểm tra.
9. **Không dùng ảnh** có người/biển số rõ nét trong báo cáo, slide, repo.
10. **Không tuyên bố** mô hình đủ an toàn cho hệ thống giao thông thực tế.

---

## 14. Lệnh chuẩn

```powershell
# Kiểm tra môi trường
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"

# Dữ liệu
python -m src.data --download --report

# Train
python -m src.train --config configs/e3_cnnb_aug.yaml --seed 42

# Đánh giá
python -m src.evaluate --run outputs/runs/e3_cnnb_aug_s42 --split val
python -m src.evaluate --run outputs/runs/final_s42 --split test --confirm-freeze

# Tổng hợp + demo
python -m src.summarize
streamlit run app.py

# Test giao diện (Playwright, cần cài chromium)
python tests/e2e_app.py
```

---

## 15. Definition of Done

Một task chỉ được coi là xong khi:

- [ ] Code chạy được trong `.venv`, không lỗi import.
- [ ] Có smoke test hoặc test tương ứng đã chạy.
- [ ] Không hard-code đường dẫn tuyệt đối.
- [ ] Config/hằng số được đưa vào YAML, không nằm rải rác.
- [ ] Kết quả (nếu có) được ghi vào `outputs/` đúng quy ước.
- [ ] Tài liệu liên quan được cập nhật.
- [ ] `git status` không có file rác ngoài ý muốn.

---

## 16. Quy trình làm việc cho agent

1. Đọc `AGENTS.md` + file kế hoạch liên quan.
2. Kiểm tra trạng thái repo (`git status`, file hiện có) trước khi sửa.
3. Thực hiện thay đổi **nhỏ nhất** đáp ứng yêu cầu.
4. Chạy test/smoke test.
5. Cập nhật tài liệu nếu hành vi thay đổi.
6. Báo cáo: đã làm gì, file nào, test nào đã chạy, còn rủi ro gì.
7. Không commit/push nếu chưa được yêu cầu.

---

## 17. Bẫy đã biết

| Bẫy | Hậu quả | Cách tránh |
|---|---|---|
| Quên `model.eval()` | Val không ổn định, kết quả sai | Luôn eval + no_grad |
| Preprocess lệch giữa train và demo | Demo sai dù test 99% | Một hàm `preprocess` duy nhất |
| Chia val ngẫu nhiên khi ảnh gần trùng | Val bị thổi phồng | So sánh với split theo track |
| Cài PyTorch bản CPU/CUDA cũ | Không dùng được RTX 5070 | Index cu128, kiểm tra capability (12,0) |
| `num_workers>0` trên Windows | Treo/ crash | `if __name__ == "__main__":` |
| Ghi log cuối run | Mất dữ liệu khi gián đoạn | Flush từng epoch |
| Chọn checkpoint theo accuracy | Bỏ qua lớp hiếm | Theo val macro-F1 |
| Dùng test nhiều lần | Test không còn độc lập | Đóng băng config, cờ xác nhận |
| Copy `.venv` sang máy khác | Lỗi đường dẫn/Python | Cài lại từ requirements |
| Ảnh demo có người/biển số | Vi phạm yêu cầu môn | Crop/blur trước khi lưu |

---

## 18. Liên kết

- Kế hoạch chủ: `docs/kehoach/KE_HOACH_TONG_QUAN_GTSRB_CNN.md`
- Các bước thực hiện: `docs/kehoach/Cac_buoc_thuc_hien_GTSRB_CNN.md`
- Demo máy yếu: `docs/kehoach/Phan_tich_demo_tren_may_yeu.md`
- Repo: https://github.com/Nguyen15idhue/GTSRB-CNN
