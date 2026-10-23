# CHẠY NHANH SAU KHI CLONE (không train)

Mục tiêu: máy mới → `git clone` → cài thư viện → mở Streamlit demo.
**Không train, không tải GTSRB, không cần GPU.** Khoảng **5 phút**.

> Khác với `docs/DEMO_NHANH.md`: bản này dùng **git clone** repo, nên bạn cần
> **copy thêm 1 file checkpoint** (`best.pt`, 8,8 MB) từ máy cũ / file bundle / GitHub Release.
> Repo không chứa checkpoint vì lý do dung lượng.

---

## 0. Chuẩn bị (2 thứ)

1. **Python 3.12 64-bit** — kiểm tra: `py -3.12 --version`
2. **File `best.pt`** của mô hình cuối (8,8 MB), lấy từ:
   - máy đang chạy được: `outputs/runs/final_s42/best.pt`, hoặc
   - file `GTSRB_demo_bundle.zip` (giải nén lấy `outputs/runs/final_s42/best.pt`), hoặc
   - GitHub Release (nếu có).

---

## 1. Clone repo

```powershell
git clone https://github.com/Nguyen15idhue/GTSRB-CNN.git
cd GTSRB-CNN
```

---

## 2. Cài thư viện (torch CPU, ~1 GB)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt
```

- Không cần cài CUDA. Không cần `requirements.txt` (bản đó dành cho train).
- Nếu PowerShell chặn script: `Set-ExecutionPolicy -Scope Process Bypass` rồi activate lại.

---

## 3. Đặt checkpoint vào đúng chỗ

```powershell
New-Item -ItemType Directory -Force outputs\runs\final_s42 | Out-Null
Copy-Item "D:\duong\dan\best.pt" outputs\runs\final_s42\best.pt
```

Muốn để checkpoint ở nơi khác (không copy vào repo):

```powershell
$env:GTSRB_RUN_DIR = "D:\checkpoints\final_s42"   # thư mục có best.pt
```

`threshold.json`, `test_report.json`, `class_names.json` đã có sẵn trong repo — không cần copy.

---

## 4. Chạy

```powershell
streamlit run app.py
```

- Lần đầu hỏi **Email** → nhấn **Enter**.
- Mở **http://localhost:8501**. Dừng: `Ctrl+C`.
- Không nhận lệnh `streamlit`: dùng `python -m streamlit run app.py`.

---

## 5. Demo trong 2 phút

| Chọn ảnh | Kết quả mong đợi |
|---|---|
| “Ảnh test GTSRB — biển 16” | Cấm xe trên 3,5 tấn |
| “biển 50 km/h” / “STOP” | Giới hạn tốc độ 50 km/h / Dừng lại |
| “Biển Hàn Quốc 30 km/h” | Giới hạn tốc độ 30 km/h |
| “Cảnh đường không có biển” | **Không xác định** |
| Upload `outputs/demo_images/*.png` | top-3 + xác suất (nếu có thư mục này) |

Sidebar: accuracy **98,56 %**, macro-F1 **0,9799**, τ = 0,9462.
Biển Việt Nam → “Không xác định” hoặc sai: model chỉ biết 43 lớp biển ĐỨC.

---

## 6. Nếu không có `best.pt`

Chọn 1 trong 2:

- **Lấy từ bundle:** xin file `GTSRB_demo_bundle.zip`, giải nén, chép `best.pt` như mục 3 (nhanh nhất).
- **Tự train:** xem `docs/HUONG_DAN_CHAY_MAY_MOI.md` (cần GPU + data, ~15 phút cho 1 seed).

---

## 7. Checklist

- [ ] `py -3.12 --version` chạy
- [ ] `pip install -r requirements-demo.txt` xong, không lỗi
- [ ] `outputs\runs\final_s42\best.pt` tồn tại (hoặc đã set `GTSRB_RUN_DIR`)
- [ ] `streamlit run app.py` mở được `http://localhost:8501`
- [ ] chọn “biển 16” → ra “Cấm xe trên 3,5 tấn”
