# LẤY DỮ LIỆU ẢNH ĐỂ DEMO — GTSRB

Tài liệu thao tác: tập nào có bao nhiêu ảnh, ảnh nằm ở đâu, và cách xuất thêm ảnh
để demo mà **không** làm sai split hay rò rỉ dữ liệu.

> Quy tắc bất di bất dịch: **không dùng ảnh demo để chỉnh mô hình / ngưỡng τ**.
> Test chỉ được đánh giá **một lần** (đã dùng); val dùng để chọn mô hình.

---

## 1. Số ảnh từng tập

| Tập | Số ảnh | Nguồn vật lý |
|---|---|---|
| GTSRB train gốc trên đĩa | **39.209** | `data/gtsrb/GTSRB/Training/<lớp>/*.ppm` |
| Train **dùng được** (đã loại 8 ảnh trùng MD5 với test) | **39.201** | cùng thư mục; cột `usable_for_train` trong `train_meta.csv` |
| **Test chính thức** | **12.630** | `data/gtsrb/GTSRB/Final_Test/Images/*.ppm` + `data/gtsrb/GT-final_test.csv` |
| **Val — split `stratified_group` (mô hình cuối)** | **7.830** | **index** trên 39.201 ảnh train (không có thư mục riêng) |
| **Train — split `stratified_group`** | **31.371** | index trên 39.201 ảnh train |

Các split tham khảo:

| Split | Train | Val | Track train/val | Lớp ở val |
|---|---|---|---|---|
| `random` (dùng ở GĐ5, bị rò rỉ frame) | 31.360 | 7.841 | 1.307 / 1.306 (chồng) | 43/43 |
| `group` (GroupShuffleSplit) | 31.341 | 7.860 | 1.045 / 262 (rời) | 39/43 — thiếu 0, 19, 34, 39 |
| **`stratified_group` (chốt GĐ6)** | **31.371** | **7.830** | **1.046 / 261 (rời)** | **43/43**, lớp ít nhất 30 ảnh |

`31.371 + 7.830 = 39.201`. Test 12.630 tách hoàn toàn.

---

## 2. Ảnh val nằm ở đâu

**Val không phải thư mục riêng.** Val là danh sách **index**; ảnh gốc vẫn nằm trong
`data/gtsrb/GTSRB/Training/<lớp>/<track>_<frame>.ppm`.

Ba file quyết định mapping:

| File | Nội dung |
|---|---|
| `outputs/results/split_stratified_group.json` | `"train"`: 31.371 index, `"val"`: 7.830 index |
| `data/cache/paths_train.txt` | **39.201 dòng**; dòng thứ `i` = đường dẫn của ảnh index `i` |
| `data/cache/X_train.npy`, `y_train.npy` | ảnh uint8 64×64 và nhãn, cùng thứ tự với `paths_train.txt` |

Ví dụ 5 ảnh val đầu:

```
index 0 → data/gtsrb/GTSRB/Training/00000/00000_00000.ppm
index 1 → data/gtsrb/GTSRB/Training/00000/00000_00001.ppm
index 2 → data/gtsrb/GTSRB/Training/00000/00000_00002.ppm
```

Lấy toàn bộ 7.830 đường dẫn val:

```python
import json
from pathlib import Path

split = json.load(open("outputs/results/split_stratified_group.json", encoding="utf-8"))
paths = Path("data/cache/paths_train.txt").read_text(encoding="utf-8").splitlines()
val_paths = [paths[i] for i in split["val"]]
Path("outputs/results/val_paths.txt").write_text("\n".join(val_paths), encoding="utf-8")
print(len(val_paths))
```

> **Cẩn thận:** `outputs/results/train_meta.csv` có **39.209 dòng** (gồm 8 ảnh bị loại).
> Nếu dùng file này phải lọc `usable_for_train == 1` **trước**, rồi mới lấy dòng thứ `i`;
> dùng `paths_train.txt` (39.201 dòng) sẽ tránh lệch 8.

---

## 3. Xuất ảnh demo (PNG) từ GTSRB

App chỉ nhận JPG/JPEG/PNG, còn GTSRB là `.ppm` → dùng script có sẵn:

```powershell
# 8 ảnh val (mỗi lớp 1 ảnh)
python tests/export_demo_images.py --split val  --n 8 --out outputs/demo_images

# 8 ảnh test
python tests/export_demo_images.py --split test --n 8 --out outputs/demo_images
```

Kết quả (đang có sẵn):

```
outputs/demo_images/
├── val_00.png … val_07.png      # 8 ảnh val (lớp 0..7)
├── test_01.png, test_11.png, test_12.png, test_16.png,
│   test_18.png, test_25.png, test_33.png, test_38.png
└── demo_manifest.csv            # file, split, label, class_vi, source
```

- `--n` là số ảnh tối đa, **mỗi lớp tối đa 1 ảnh** (đa dạng lớp).
- Ảnh giữ nguyên độ phân giải gốc, chỉ convert sang PNG.
- `outputs/demo_images/` **bị gitignore** — không commit ảnh dữ liệu.
- Muốn xuất toàn bộ val (không khuyến nghị, 7.830 file): sửa script hoặc lặp theo `val_paths`.

---

## 4. Năm nguồn ảnh, dùng cho tình huống nào

| Nguồn | Đường dẫn | SL | Dùng khi |
|---|---|---|---|
| Ảnh mẫu trong app | ô “Hoặc chọn ảnh mẫu có sẵn” | 5 | demo nhanh, không cần file |
| Bộ xuất sẵn | `outputs/demo_images/*.png` | 16 | upload trực tiếp, có nhãn đối chiếu |
| Validation sạch | `split_stratified_group.json` → `"val"` | 7.830 | demo “chưa học” (nhưng đã dùng chọn mô hình) |
| Test chính thức | `data/gtsrb/GTSRB/Final_Test/Images/` | 12.630 | demo ảnh held-out; đã dùng 1 lần để báo cáo |
| Ngoài phân phối | `external_test/` + `manifest.csv` | 11 | demo “không xác định”: 4 đường không biển, 4 biển chữ, 2 biển Mỹ/Hàn |

**5 ảnh mẫu trong app:** `ext_01` (test, lớp 16), `00034` (test, 50 km/h),
`00093` (test, STOP), `ext_03` (biển Hàn 30 — OOD), `ext_09` (đường — OOD).

---

## 5. Lấy dữ liệu từ đầu (máy mới)

```powershell
.\.venv\Scripts\Activate.ps1

# Tải + xác minh + EDA (39.209 train, 12.630 test)
python -m src.data --download --report

# Cache 64×64 + 3 split + mean/std + hình augmentation
python -m src.data --prepare

# Checklist dữ liệu
python -m src.data --verify
```

Chi tiết nguồn:

- **Test:** `torchvision.datasets.GTSRB(split="test", download=True)`.
- **Train:** archive chính thức `GTSRB_Final_Training_Images.zip` (39.209 ảnh) —
  **không** dùng bản train 26.640 ảnh của torchvision.
- Trang chính thức có thể không truy cập được: cách tải thực tế + MD5 ghi trong
  `outputs/results/data_report.json`.

Bản đồ thư mục sau khi tải:

```
data/gtsrb/GTSRB/Training/00000…00042/*.ppm
data/gtsrb/GTSRB/Final_Test/Images/*.ppm
data/gtsrb/GT-final_test.csv
data/cache/{X_train,X_test,y_train,y_test,roi_train,roi_test}.npy
data/cache/paths_{train,test}.txt
data/cache/cache_meta.json
outputs/results/split_random.json
outputs/results/split_group.json
outputs/results/split_stratified_group.json
outputs/results/train_meta.csv
outputs/results/excluded_train_duplicates.csv
```

---

## 6. Thêm ảnh ngoài dataset (OOD) để demo

1. Chỉ dùng ảnh **CC0 / Public domain** hoặc ảnh tự chụp đã ẩn danh.
2. Crop quanh biển; **che mặt người / biển số**; xoá EXIF.
3. Thêm dòng vào `external_test/manifest.csv` với các cột:
   `filename, group, source, license, artist, date, description, anonymized, expected_label, expected_vi, scenario`
   - `group`: `ood_sign_speed` | `ood_sign_text` | `no_sign` | `reference_gtsrb_test`
   - `license`: ghi đúng (CC0/PD; ảnh GTSRB ghi rõ “nghiên cứu, không phải CC0”)
4. Chạy lại đánh giá OOD + cập nhật ngưỡng (τ vẫn **chỉ chọn trên val**):

```powershell
python -m src.error_analysis --run outputs/runs/final_s42
```

5. **Không** chỉnh τ theo ảnh OOD vừa thêm; đó là dữ liệu demo, không phải val.

---

## 7. Kiểm tra số đếm

```powershell
python -m src.data --verify
python -c "import json;print(json.load(open('outputs/results/data_report.json',encoding='utf-8'))['counts'])"
python -c "import json;d=json.load(open('outputs/results/split_stratified_group.json',encoding='utf-8'));print(len(d['train']),len(d['val']))"
```

Kỳ vọng: `39209 / 39201 / 12630` và `31371 7830`.

---

## 8. Lưu ý để không làm hỏng kết quả

- **Không** di chuyển/đổi tên/xoá file trong `Training/` hay `Final_Test/`:
  split lưu bằng **index**, đổi thứ tự là sai mapping.
- **Không** copy ảnh val/test vào train hoặc ngược lại.
- **Không** chia lại split mỗi lần chạy: dùng JSON đã lưu.
- **Không** dùng test để chọn mô hình/LR/epoch/τ. Test đã đánh giá **1 lần**
  (`test_runs.log` 1 dòng); chạy lại bị từ chối.
- Val sạch track nhưng **đã dùng** để chọn checkpoint/multi-seed/τ → khi demo
  nói “chưa dùng để train”, không nói “dữ liệu hoàn toàn mới”.
- Ảnh trong `data/` và `outputs/demo_images/` **không commit**.

---

## 9. Liên kết

- Quy trình đầy đủ: `docs/kehoach/Cac_buoc_thuc_hien_GTSRB_CNN.md`
- Báo cáo: `docs/bao_cao/BAO_CAO.docx`
- Quy tắc làm việc: `AGENTS.md`
- Script xuất ảnh: `tests/export_demo_images.py`
- Ảnh OOD hiện có: `external_test/` + `external_test/manifest.csv`
