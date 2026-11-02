# Kế hoạch thực nghiệm v2 — GTSRB CNN

Thư mục này chứa **kế hoạch đầy đủ** để tái thiết kế thực nghiệm, trả lời 4 phản biện trong buổi bảo vệ:

1. Bài test chưa đủ để chốt mô hình.
2. Số liệu còn rò rỉ (leak) do dùng `split_random`.
3. Kết quả chỉ được báo cáo, chưa được giải thích cơ chế.
4. Chưa rõ từng cấu hình (BN, dropout, WD, augmentation, depth, GAP, width, LR…) có thực sự cần hay không.

> **Trạng thái:** kế hoạch, **chưa thực thi**. Tài liệu không chứa kết quả thật; mọi bảng kết quả là khung chờ số. Không dùng số cũ trên `split_random` để kết luận.

---

## 1. File trong thư mục

| File | Vai trò | Khi nào đọc |
|---|---|---|
| `KE_HOACH_V2_DAY_DU.md` | **Tài liệu chính**: bối cảnh, split, pre-registration, ma trận thí nghiệm, thống kê, cách giải thích, kế hoạch triển khai, mẫu báo cáo | Đọc toàn bộ trước khi làm |
| `MA_TRAN_RUN.csv` | Danh sách config để chạy (56 dòng: baseline, factorial, kiến trúc, hyperparam, augmentation, xác nhận, final) | Dùng để sinh `configs/v2/*.yaml` |
| `CHECKLIST_TRIEN_KHAI.md` | Checklist từng phase, tiêu chí pass, ký xác nhận | Dùng hằng ngày khi triển khai |
| `MAU_KET_QUA_GIA_DINH.md` | **Mẫu kết quả giả định** — cho thấy sau khi chạy sẽ có bảng/hình/thông tin gì và mỗi thông tin trả lời vấn đề nào. **Toàn bộ số là giả định, không dùng trong báo cáo** | Đọc để hình dung đầu ra, không chép số |
| `README.md` | Trang này: tóm tắt và chỉ mục | Đọc đầu tiên |

---

## 2. Tóm tắt trong 1 phút

- **Chia lại từ đầu** 39.201 ảnh thành 3 tầng:
  - **Lockbox** 20% track: **7.830 ảnh / 261 track**, test một lần.
  - **Dev** 80% track: **31.371 ảnh / 1.046 track**.
  - Trong dev: **screening** 25.131/6.240 và **3-fold** 20.541–21.360/10.011–10.830.
- **Mô hình cuối vẫn train trên 31.371 ảnh** — bằng đúng cỡ train cũ. Chỉ mô hình screening/3-fold mới train ít hơn.
- **Mọi run dùng `stratified_group`**, không dùng `random`. Official test 12.630 ảnh chỉ là benchmark phụ (raw + clean 12.499).
- **Factorial 2⁴** cho BN × dropout × WD × augmentation; tách riêng kiến trúc (depth/GAP/width), hyperparam và từng thành phần augmentation.
- **Chọn mô hình bằng OOF 3-fold × 3 seed**, không bằng một val 7.830 ảnh; kết luận kèm Δ + CI cluster bootstrap theo track.
- **Test một lần** trên lockbox sau khi freeze; không tune trên test.
- **Mỗi yếu tố có decision card**: mức đã thử, kết quả, Δ, CI, cơ chế, kết luận giữ/bỏ.

---

## 3. Con số khóa

| Hạng mục | Giá trị |
|---|---:|
| Pool sau loại 8 ảnh trùng MD5 | 39.201 |
| Lockbox (20% track) | 7.830 ảnh / 261 track |
| Dev | 31.371 ảnh / 1.046 track |
| Screening train / val | 25.131 / 6.240 |
| 3-fold train / val mỗi fold | 20.541–21.360 / 10.011–10.830 |
| Final train | 31.371 |
| Official test | 12.630 (clean 12.499) |
| Số run đầy đủ | ~174 |
| Số run tối thiểu | ~71 |
| Thời gian GPU đầy đủ | ~15–18 giờ |

> Số ảnh lockbox/screening/3-fold có thể lệch vài chục ảnh tùy seed. **Số chính xác lấy từ `outputs/splits/v2/manifest.json`**, không chép từ bảng này vào báo cáo.

---

## 4. Vì sao thiết kế này trả lời được 4 phản biện

| Phản biện | Cách trả lời | Bằng chứng |
|---|---|---|
| 1. Bài test chưa đủ | Pre-registration + factorial + tách kiến trúc + OOF + lockbox + CI + luật quyết định | Mục 4, 6, 7; `summary.csv`, `decision_cards.csv` |
| 2. Còn leak | Split 3 tầng theo track/sign-group; lockbox 0 overlap; mean/std theo split; official test chỉ phụ | Mục 3; `manifest.json`; audit 131 ảnh |
| 3. Không giải thích | Decision card + learning curve + per-class Δ + Grad-CAM + calibration + corruption | Mục 8; hình 5.x |
| 4. Không rõ cấu hình cần | Bật/tắt từng yếu tố; factorial; aug components; báo cả "không đủ bằng chứng" | Mục 6; bảng main effect |

---

## 5. Thứ tự đọc / làm

1. Đọc `KE_HOACH_V2_DAY_DU.md` mục 0–4.
2. Chốt và commit tài liệu (pre-registration) **trước khi chạy**.
3. Làm Phase 0 trong `CHECKLIST_TRIEN_KHAI.md`: chia split, audit, test.
4. Chạy Phase 1–3 theo `MA_TRAN_RUN.csv`.
5. Làm Phase 4: thống kê, hình, viết báo cáo.
6. (Tùy chọn) xem `MAU_KET_QUA_GIA_DINH.md` để hình dung bảng/hình đầu ra — **không chép số giả định**.

---

## 6. Những điều không được làm

- Không dùng `split_random` cho bất kỳ kết luận nào.
- Không tune trên lockbox/official test.
- Không so 10 epoch với 40 epoch.
- Không đổi 3–4 yếu tố trong một run rồi quy công cho một yếu tố.
- Không báo "tốt hơn" khi CI chứa 0.
- Không đưa số cũ 99,9% (random val) vào kết luận.
- Không tuyên bố mô hình đủ an toàn cho hệ thống giao thông thực tế.
- Không commit `data/`, checkpoint, `.venv`.

---

## 7. Liên kết

- Kế hoạch bản 0: `../0/KE_HOACH_TONG_QUAN_GTSRB_CNN.md`
- Báo cáo hiện tại: `../../bao_cao/BAO_CAO.md`
- Kết quả cũ: `../../../outputs/results/summary.csv`
- Split hiện tại: `../../../outputs/results/split_stratified_group.json`
- Audit leak: `../../../outputs/results/leak_delta.json`
