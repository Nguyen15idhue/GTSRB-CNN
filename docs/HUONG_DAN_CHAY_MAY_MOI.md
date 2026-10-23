# HƯỚNG DẪN CHẠY TRÊN MÁY MỚI (CÓ GPU)

> **Chỉ muốn demo nhanh?** Xem bản 1 trang: **`docs/DEMO_NHANH.md`** (5 phút, không train, không data).

Tài liệu này dành cho người **chưa từng chạy repo**: làm theo đúng thứ tự, copy–paste lệnh,
khoảng **30–45 phút** là có demo + kết quả. Không cần biết trước về PyTorch.

> **Đã kiểm chứng trên:** Windows 11 · Python 3.12 · RTX 5070 12 GB · driver 610.88.
> Linux chạy tương tự (xem mục 13). macOS **không có CUDA** → chỉ demo/CPU (mục 13).

---

## 0. Bạn muốn làm gì?

| Mục tiêu | Cần làm | Thời gian |
|---|---|---|
| Chỉ **đọc kết quả** | mở `docs/bao_cao/BAO_CAO.docx` | 0 phút |
| **Chỉ demo, không train** | giải nén bundle + cài `requirements-demo.txt` + `streamlit run app.py` | **~5 phút** |
| **Xem demo từ mã nguồn** | cài đặt + data + **train 1 seed** + `streamlit run app.py` | ~15 phút |
| **Tái lập đầy đủ** | cài đặt + data + 3 seed + đánh giá + báo cáo | ~30–45 phút |

> Repo **không chứa checkpoint** (`best.pt` bị gitignore). Vì vậy máy mới muốn demo phải
> **train lại** `configs/final.yaml` (hoặc được cấp sẵn `best.pt`, xem mục 0.1 và 9).

---

## 0.1. `git clone` xong có gì, còn thiếu gì?

`git clone` chỉ mang **mã nguồn + báo cáo + kết quả test đã lưu**, không mang môi trường/dữ liệu/checkpoint.

| Có sẵn trong repo | Thiếu — phải tự thêm |
|---|---|
| `src/`, `app.py`, `configs/` (kể cả `final.yaml`), `tests/` | `.venv` + thư viện Python |
| `.streamlit/config.toml`, `requirements.txt`, `requirements-demo.txt` | `data/` GTSRB (1,4 GB) |
| `docs/bao_cao/BAO_CAO.docx` (mở đọc ngay) | `outputs/runs/*/best.pt` (checkpoint, 8,8 MB) |
| `outputs/results/`: `test_report.json`, `test_final.json`, `summary.csv`, `multiseed.json`, `threshold.json`, `predictions_test_final.csv` | `outputs/demo_images/` (tuỳ chọn) |
| `external_test/` 11 ảnh + manifest, `outputs/figures/` | Playwright Chromium (tuỳ chọn, nếu test UI) |

> Bị gitignore: `data/`, `outputs/runs/`, `outputs/demo_images/`, `outputs/GTSRB_demo_bundle.zip`, `.venv*`.
> Dung lượng clone: ~68 MB (kể cả `.git`); ~29 MB nội dung.

### Ba mức cài đặt

**A. Chỉ đọc báo cáo — không cần cài gì:** mở `docs/bao_cao/BAO_CAO.docx`.

**B. Chỉ demo (không train, không data) — ~5 phút.** Cần Python 3.12 + **copy `best.pt`** (từ máy cũ
hoặc từ `GTSRB_demo_bundle.zip`; không có trong git):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt          # torch CPU, ~1 GB, không cần GPU
# đặt checkpoint tại: outputs\runs\final_s42\best.pt
# (hoặc nơi khác + set GTSRB_RUN_DIR trỏ tới thư mục có best.pt)
streamlit run app.py
```

**C. Chạy đầy đủ (train/eval) — ~30–45 phút.** Cần Python 3.12, Git, **driver NVIDIA**
(không cần cài CUDA Toolkit), ~8 GB trống:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
python -m src.data --download --report
python -m src.data --prepare --verify
python -m src.train --config configs/final.yaml --seed 42
streamlit run app.py
```

### Không cần cài

- **CUDA Toolkit** riêng: torch cu128 đã kèm runtime.
- **Visual Studio / Build Tools**: các gói đều có wheel.
- **GPU cho demo**: torch CPU đủ (model 1 ảnh ~1,2 ms).
- **Dữ liệu tải thủ công**: chỉ khi mạng chặn, xem mục 5.

### Hai điều dễ vấp sau khi clone

1. **Test bị từ chối**: repo đã có `outputs/results/test_report.json` (test chỉ chạy 1 lần).
   Muốn tái lập: sao lưu 3 file `test_report.json`, `test_final.json`, `test_runs.log` rồi xoá,
   sau đó chạy **đúng 1 lần** `python -m src.evaluate --run outputs/runs/final_s42 --split test --confirm-freeze`
   (chi tiết mục 9).
2. **Streamlit lần đầu có thể hỏi Email**: thông tin này lưu theo user, không nằm trong repo → nhấn **Enter**.

### Tóm tắt một dòng

- **Clone + đọc báo cáo:** không cần cài gì.
- **Clone + demo:** Python + `requirements-demo.txt` + **copy `best.pt`**.
- **Clone + train/eval:** Python + torch cu128 + requirements + `--download --prepare`.

> Muốn “clone xong chạy Streamlit luôn” mà không copy checkpoint: dùng `GTSRB_demo_bundle.zip`
> (mục 8.1) hoặc đính kèm `best.pt` vào GitHub Release — **không** commit checkpoint vào git.

---

## 1. Máy cần gì

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| GPU | NVIDIA có CUDA (driver mới) | RTX 5070 / 3060 12 GB trở lên |
| Driver NVIDIA | hỗ trợ CUDA 12.8 (Windows: ≥ 570.x) | bản mới nhất |
| RAM | 16 GB | 32 GB |
| Ổ đĩa trống | 8 GB | 12 GB (venv ~5 GB + data ~1,4 GB + outputs) |
| Python | **3.12 64-bit** | đúng 3.12 |
| Git | có | có |
| Internet | để tải PyTorch + GTSRB | mạng ổn định |

Kiểm tra nhanh trong PowerShell:

```powershell
nvidia-smi                    # thấy GPU + Driver Version
py -3.12 --version            # Python 3.12.x
git --version
```

Nếu chưa có Python: tải **Python 3.12 (64-bit)** từ python.org (nhớ tick *Add to PATH*) hoặc
`winget install Python.Python.3.12`. Chưa có Git: `winget install Git.Git`.
Nếu driver cũ: cập nhật bằng GeForce Experience / trang NVIDIA rồi khởi động lại.

---

## 2. Lấy mã nguồn

```powershell
# Cách 1: clone
git clone https://github.com/Nguyen15idhue/GTSRB-CNN.git
cd GTSRB-CNN

# Cách 2: đã có sẵn thư mục dự án thì chỉ cần mở PowerShell tại thư mục đó
cd C:\0\1.Laptrinh\BTLHocSau
```

Mọi lệnh sau đây chạy **tại thư mục gốc repo** (nơi có `README.md`, `src/`, `app.py`).

> Clone **không** kèm `data/`, `outputs/runs/` (checkpoint), `outputs/demo_images/` hay `.venv`.
> Xem bảng “có sẵn / còn thiếu” ở **mục 0.1** để biết cần thêm gì.

---

## 3. Tạo môi trường ảo

```powershell
Set-ExecutionPolicy -Scope Process Bypass
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Dấu hiệu đúng: đầu dòng lệnh có `(.venv)`.

> Nếu PowerShell báo `running scripts is disabled`: chạy lại `Set-ExecutionPolicy -Scope Process Bypass`
> rồi activate lại.

---

## 4. Cài PyTorch + thư viện

**Bắt buộc cài PyTorch từ index CUDA trước** (bản PyPI mặc định là CPU, train rất chậm):

```powershell
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

- Tải khoảng **2,7 GB** cho torch; tổng venv ~5 GB.
- `requirements.txt` có ghi `torch==2.11.0`; vì torch `2.11.0+cu128` đã cài nên pip **không cài lại**
  bản CPU. **Đừng** chạy `pip install -r requirements.txt` trước lệnh torch.

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

Kết quả đúng dạng:

```
2.11.0+cu128 True (12, 0)      # RTX 5070; GPU khác sẽ in capability khác, miễn là True
```

- `False` → đang là bản CPU hoặc driver lỗi: cài lại đúng lệnh torch ở trên, kiểm tra `nvidia-smi`,
  khởi động lại máy.
- `no kernel image is available` → wheel CUDA cũ: gỡ `torch/torchvision` rồi cài lại cu128.

---

## 5. Tải dữ liệu GTSRB

```powershell
python -m src.data --download --report
```

Việc này tải **test 12.630 ảnh** (qua torchvision) và **train 39.209 ảnh** (archive chính thức),
sau đó đếm lại, kiểm tra ảnh lỗi/trùng lặp/ROI/track và vẽ 3 hình EDA.
Thời gian tuỳ mạng (thường 1–5 phút). Kết quả đúng:

```
counts: train=39209, test=12630, train_pool_after_exclusion=39201, excluded_duplicates=8
```

Sau đó tạo cache 64×64 + chia tập + mean/std:

```powershell
python -m src.data --prepare
python -m src.data --verify
```

- `--prepare`: cache ~610 MB (13–20 giây), sinh 3 split + `data_stats.json` + hình augmentation.
- `--verify`: checklist dữ liệu, kỳ vọng **12/12 PASS**.

### Nếu bước tải lỗi (trang GTSRB không truy cập được)

1. Test: cho torchvision tự tải lại (`GTSRB(split="test", download=True)`).
2. Train: tải thủ công `GTSRB_Final_Training_Images.zip` từ mirror
   `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/`
   (MD5 `f33fd80ac59bff73c82d25ab499e03a3`), giải nén sao cho có:
   ```
   data/gtsrb/GTSRB/Training/00000/*.ppm … 00042/*.ppm     (39.209 ảnh)
   data/gtsrb/GTSRB/Final_Test/Images/*.ppm                 (12.630 ảnh)
   data/gtsrb/GT-final_test.csv
   ```
3. Chạy lại `python -m src.data --report` rồi `--prepare --verify`.

### Bản đồ dữ liệu sau khi xong

```
data/gtsrb/GTSRB/Training/<lớp>/<track>_<frame>.ppm   39.209 ảnh
data/gtsrb/GTSRB/Final_Test/Images/*.ppm              12.630 ảnh
data/gtsrb/GT-final_test.csv                          nhãn test
data/cache/*.npy                                      cache 64×64 (~610 MB)
outputs/results/split_random.json                     split GĐ5 (31.360/7.841)
outputs/results/split_group.json                      split so sánh
outputs/results/split_stratified_group.json           split chốt (31.371/7.830)
```

> Ảnh **val không nằm trong thư mục riêng**; val là danh sách index trong
> `split_stratified_group.json`, trỏ tới ảnh trong `Training/`.
> Chi tiết cách lấy ảnh demo: `docs/DEMO_DATA.md`.

---

## 6. Kiểm tra môi trường (30 giây)

```powershell
python -m pytest tests -q
```

Kỳ vọng: `11 passed`. Nếu lỗi import → kiểm tra đang ở thư mục gốc repo và đã activate `.venv`.

Tuỳ chọn — kiểm tra hạ tầng train:

```powershell
python -m src.train --config configs/smoke.yaml --overfit-batch      # loss < 0,05, acc 100%
python -m src.train --config configs/smoke.yaml --determinism-check  # 2 lần cùng seed lệch ~0
```

---

## 7. Train mô hình cuối (bắt buộc để demo/evaluate)

```powershell
python -m src.train --config configs/final.yaml --seed 42
python -m src.train --config configs/final.yaml --seed 123
python -m src.train --config configs/final.yaml --seed 2024
python -m src.summarize --multiseed
```

- Mỗi seed ~6–7 phút trên RTX 5070 (GPU yếu hơn có thể 15–30 phút/seed).
- Mỗi run tạo `outputs/runs/final_s<seed>/`: `config.yaml`, `env.json`, `metrics.csv`,
  `best.pt`, `curves.png`, `meta.json`.
- `multiseed.json` sinh mean ± std. Kỳ vọng (máy tương đương):
  - accuracy **0,9808 ± 0,0031**
  - macro-F1 **0,9654 ± 0,0047**

> Chạy lại cùng seed sẽ báo `FileExistsError` (đúng quy ước “không ghi đè run”).
> Muốn chạy lại: đổi `experiment.id` trong config hoặc dùng seed khác.

**Nếu chỉ cần demo nhanh:** chạy **1 seed 42** là đủ (`final_s42/best.pt`).

### Tuỳ chọn: tái lập toàn bộ ablation/baseline

```powershell
foreach ($c in 'e1_cnna','e2_cnnb','e3_cnnb_aug','e4_cnnc_gap_aug','e5a_lr3e4','e5b_lr3e3','e7_classweight') {
  python -m src.train --config "configs/$c.yaml"
}
python -m src.baseline --config configs/default.yaml      # HOG+SVC, ~33 phút CPU
python -m src.summarize --figure
```

---

## 8. Chạy demo

```powershell
streamlit run app.py
```

- Lần đầu Streamlit hỏi **Email** → **nhấn Enter** (bỏ trống).
- Sau đó màn hình in: `Local URL: http://localhost:8501` → mở link.
- Dừng: `Ctrl+C`.

Ảnh để thử:

| Cách | Ở đâu |
|---|---|
| Chọn sẵn trong app | ô “Hoặc chọn ảnh mẫu có sẵn” (5 ảnh) |
| Bộ PNG xuất sẵn | `outputs/demo_images/*.png` (16 ảnh, có `demo_manifest.csv`) |
| Xuất thêm từ GTSRB | `python tests/export_demo_images.py --split val --n 8 --out outputs/demo_images` |
| Ảnh ngoài (OOD) | `external_test/` (11 ảnh, có manifest) |

> Model **chỉ biết 43 lớp biển ĐỨC**. Biển Việt Nam sẽ bị trả “không xác định” hoặc đoán sai —
> đó là giới hạn, không phải lỗi app. Xem `docs/DEMO_DATA.md`.

Tuỳ chọn — test giao diện tự động (Playwright):

```powershell
python -m playwright install chromium     # ~150 MB, chỉ lần đầu
python tests/e2e_app.py                   # 9 kịch bản, ~40 giây, kèm screenshot
```

---

### 8.1. Chỉ demo trên máy mới — KHÔNG cần train, KHÔNG cần data

Bundle đã đóng gói sẵn model + báo cáo + ảnh mẫu (**A+**). Tạo bundle trên máy đang có model:

```powershell
python tests/package_demo.py
# → outputs/GTSRB_demo_bundle.zip  (~15 MB)
```

Mang file zip sang máy mới, giải nén, rồi:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt      # torch CPU, ~1 GB, KHÔNG cần GPU
streamlit run app.py                      # mở http://localhost:8501
```

Bundle gồm: `app.py`, `src/`, `best.pt` (8,8 MB), `threshold.json`, `test_report.json`,
`multiseed.json`, `configs/class_names.json`, **`docs/bao_cao/BAO_CAO.docx`**,
**16 ảnh demo** (`outputs/demo_images/*.png` + `demo_manifest.csv`),
5 ảnh mẫu của app (`external_test/` + 2 ảnh test `.ppm`), `HUONG_DAN.txt`.

Kỳ vọng (đã kiểm trong venv CPU sạch, không data):

| Mục | Kết quả |
|---|---|
| Sidebar | accuracy 98,56 % · macro-F1 0,9799 · top-3 99,64 % · τ 0,9462 |
| Mẫu biển 16 / 50 km/h / STOP | đúng lớp 16 / 2 / 14, conf ~1,00 |
| Biển Hàn 30 (crop) | đúng lớp 1, conf 0,95 |
| Ảnh đường không biển | “Không xác định” |
| Biển Việt Nam | “Không xác định” hoặc sai (giới hạn 43 lớp Đức) |
| Báo cáo | mở `docs/bao_cao/BAO_CAO.docx` |
| Tốc độ (CPU) | model ~1,2 ms/ảnh; upload → kết quả thường < 1 giây |

**Không làm được trong bundle:** `src.data`, `src.train`, `evaluate --split val/test`
(thiếu dataset/cache). Nếu cần tái lập số liệu, dùng mã nguồn đầy đủ (mục 5–9).

## 9. Đánh giá test — CHỈ MỘT LẦN

> Repo này **đã đánh giá test đúng 1 lần** và lưu kết quả trong `outputs/results/`.
> Lệnh test sẽ **bị từ chối** nếu report đã tồn tại. Đây là chủ ý (test phải độc lập).

Nếu bạn muốn **tái lập từ đầu** trên máy mới (chấp nhận test dùng lại 1 lần):

```powershell
# 1) Sao lưu kết quả cũ (không xoá)
Copy-Item outputs/results/test_report.json outputs/results/test_report.backup.json
Copy-Item outputs/results/test_final.json  outputs/results/test_final.backup.json
Copy-Item outputs/results/test_runs.log    outputs/results/test_runs.backup.log

# 2) Xoá report/log hiện tại để lần chạy mới được ghi
Remove-Item outputs/results/test_report.json, outputs/results/test_final.json, outputs/results/test_runs.log

# 3) Chạy đúng 1 lần
python -m src.evaluate --run outputs/runs/final_s42 --split test --confirm-freeze
```

Kỳ vọng (máy tương đương): accuracy **0,9856**, macro-F1 **0,9799**, top-3 **0,9964** trên
12.630 ảnh. Kết quả ghi vào `test_report.json`, `test_final.json`, `test_runs.log`,
`outputs/runs/final_s42/eval_test/`.

- **Không** sửa config/mô hình sau khi đã xem test.
- Nếu buộc phải chạy lại do bug: `--allow-rerun --reason "..."` → sinh bản `_v2`, không ghi đè.
- Không có checkpoint: lệnh sẽ báo thiếu `best.pt` → làm mục 7 trước.

---

## 10. Xem kết quả và báo cáo

| Muốn xem | Mở file |
|---|---|
| Kết quả test (acc, F1, per-class, confusion) | `outputs/results/test_report.json` |
| Bảng tổng hợp các run | `outputs/results/summary.csv` |
| Đa seed | `outputs/results/multiseed.json` |
| Dự đoán từng ảnh test | `outputs/results/predictions_test_final.csv` |
| Hình cho báo cáo | `outputs/figures/` (01…11) |
| **Báo cáo** | `docs/bao_cao/BAO_CAO.docx` (mở bằng Word) |

Sinh lại báo cáo sau khi có kết quả mới:

```powershell
python docs/bao_cao/build_report.py
python docs/bao_cao/md_to_docx.py
```

---

## 11. Thời gian tham khảo (RTX 5070)

| Việc | Thời gian |
|---|---|
| Cài torch + requirements | 5–15 phút (tuỳ mạng) |
| Tải data + EDA | 1–5 phút |
| `--prepare` + `--verify` | ~20–30 giây |
| pytest | ~3 giây |
| Train 1 seed | ~6,5 phút |
| Train 3 seed | ~20 phút |
| Evaluate val | ~20 giây |
| Evaluate test (1 lần) | ~1 phút |
| Error analysis + OOD | ~2 phút |
| E2E UI (Playwright) | ~40 giây |
| Toàn bộ E1–E7 + baseline | ~1 giờ |

---

## 12. Lỗi thường gặp

| Lỗi / triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `torch.cuda.is_available()` = False | cài torch bản CPU | cài lại torch từ index cu128, kiểm tra `nvidia-smi`, khởi động lại |
| `no kernel image is available` | wheel CUDA quá cũ | gỡ torch/torchvision, cài lại cu128 |
| `ModuleNotFoundError: No module named 'src'` | chạy sai thư mục | `cd` về thư mục gốc repo (có `src/`) |
| `ModuleNotFoundError: streamlit/pandas…` | chưa activate venv | `.\.venv\Scripts\Activate.ps1` rồi cài lại requirements |
| `FileNotFoundError: data_stats.json` | chưa prepare | `python -m src.data --prepare` |
| `FileNotFoundError: ... best.pt` | chưa train final | làm mục 7 (hoặc đặt checkpoint đúng chỗ) |
| `FileExistsError: Run đã tồn tại` | chạy lại cùng seed | đổi seed hoặc `experiment.id` (không ghi đè run cũ) |
| `streamlit` hỏi Email | onboarding lần đầu | nhấn Enter; hoặc tạo `~/.streamlit/credentials.toml` với `[general] email=""` |
| `Port 8501 is already in use` | cổng bận | `streamlit run app.py --server.port 8502` |
| Test bị từ chối (`đã tồn tại`) | test đã chạy 1 lần | xem mục 9 (sao lưu rồi mới chạy lại) |
| DataLoader treo/crash trên Windows | `num_workers` | giảm `data.num_workers: 0` trong config hoặc chạy lại |
| `CUDA out of memory` | batch quá lớn | giảm `train.batch_size` trong config (ghi rõ khi báo cáo) |
| Tải GTSRB lỗi DNS | trang chính thức không truy cập | làm fallback thủ công ở mục 5 |
| Playwright: `Executable doesn't exist` | chưa cài Chromium | `python -m playwright install chromium` |

---

## 13. Linux / macOS

**Linux (NVIDIA):** các bước giống hệt, chỉ khác lệnh kích hoạt và cài Python:

```bash
sudo apt install python3.12 python3.12-venv git
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m src.data --download --report
python -m src.data --prepare --verify
python -m src.train --config configs/final.yaml --seed 42
streamlit run app.py
```

**macOS:** không có CUDA → chỉ chạy CPU (hoặc MPS không được hỗ trợ trong code).
Demo/báo cáo chạy được nếu có checkpoint, nhưng **train rất chậm**:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0      # bản CPU
pip install -r requirements.txt
python -m pytest tests -q
streamlit run app.py
```

---

## 14. Checklist “đã chạy xong”

- [ ] `python` là 3.12 trong `.venv`
- [ ] `torch.cuda.is_available()` = True
- [ ] `data_report.json` có `39209 / 12630 / 39201 / 8`
- [ ] `python -m src.data --verify` → 12/12 PASS
- [ ] `python -m pytest tests -q` → 11 passed
- [ ] `outputs/runs/final_s42/best.pt` tồn tại
- [ ] (nếu dùng bundle) `streamlit run app.py` mở được và sidebar hiện 98,56 %
- [ ] `streamlit run app.py` mở được `http://localhost:8501`
- [ ] `outputs/results/test_report.json` có accuracy **0,9856**, macro-F1 **0,9799**
- [ ] Mở được `docs/bao_cao/BAO_CAO.docx`

---

## 15. Liên kết

- Lấy ảnh demo, vị trí val, xuất thêm ảnh: `docs/DEMO_DATA.md`
- Quy trình 13 giai đoạn + nhật ký: `docs/kehoach/Cac_buoc_thuc_hien_GTSRB_CNN.md`
- Kế hoạch chủ: `docs/kehoach/KE_HOACH_TONG_QUAN_GTSRB_CNN.md`
- Quy tắc làm việc: `AGENTS.md`
- Báo cáo: `docs/bao_cao/BAO_CAO.docx`
