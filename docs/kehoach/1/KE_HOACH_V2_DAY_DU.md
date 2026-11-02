# KẾ HOẠCH THỰC NGHIỆM V2 — GTSRB CNN

## Tái thiết kế đánh giá để trả lời 4 phản biện

| Thông tin | Giá trị |
|---|---|
| Dự án | Bài tập lớn Học sâu 12INF028 — Phân loại 43 lớp biển báo GTSRB bằng CNN tự xây dựng |
| Phiên bản tài liệu | **2.0 — kế hoạch đầy đủ** |
| Ngày lập | 02/11/2026 |
| Trạng thái | **Kế hoạch, chưa thực thi.** Tài liệu không chứa kết quả thật; mọi ô kết quả là khung chờ số |
| Thay thế / bổ sung | Bổ sung cho `docs/kehoach/0/KE_HOACH_TONG_QUAN_GTSRB_CNN.md`; **không sửa** bản 0 |
| Người thực hiện | 1 người, máy RTX 5070 12 GB, i9-14900KF, 64 GB RAM |
| Ngôn ngữ | Tiếng Việt |
| Tài liệu liên quan | `AGENTS.md`, `docs/bao_cao/BAO_CAO.md`, `outputs/results/summary.csv`, `outputs/results/split_stratified_group.json`, `outputs/results/leak_delta.json` |
| Mục đích | Chốt một giao thức thực nghiệm đủ chặt để (1) chọn mô hình có căn cứ, (2) không còn rò rỉ, (3) giải thích được cơ chế, (4) kết luận rõ từng cấu hình có cần hay không |

> **Cam kết xuyên suốt:** mọi con số trong báo cáo/slide phải sinh ra từ lần chạy thật, lấy từ `outputs/results/v2/`. Không đưa số dự kiến, số "minh họa", số cũ đã leak vào kết luận. Các bảng trong tài liệu này chỉ là **thiết kế**, không phải kết quả.

---

## Mục lục

0. [Tóm tắt điều hành](#0-tóm-tắt-điều-hành)
1. [Bối cảnh và chẩn đoán 4 phản biện](#1-bối-cảnh-và-chẩn-đoán-4-phản-biện)
2. [Nguyên tắc thiết kế](#2-nguyên-tắc-thiết-kế)
3. [Dữ liệu và chia split v2](#3-dữ-liệu-và-chia-split-v2)
4. [Đăng ký trước (pre-registration)](#4-đăng-ký-trước-pre-registration)
5. [Giao thức huấn luyện cố định](#5-giao-thức-huấn-luyện-cố-định)
6. [Ma trận thí nghiệm](#6-ma-trận-thí-nghiệm)
7. [Phân tích thống kê](#7-phân-tích-thống-kê)
8. [Giải thích cơ chế](#8-giải-thích-cơ-chế)
9. [Đối chiếu 4 vấn đề và kịch bản phản biện](#9-đối-chiếu-4-vấn-đề-và-kịch-bản-phản-biện)
10. [Kế hoạch triển khai](#10-kế-hoạch-triển-khai)
11. [Artifact và quy ước](#11-artifact-và-quy-ước)
12. [Thay đổi báo cáo](#12-thay-đổi-báo-cáo)
13. [Rủi ro và đối phó](#13-rủi-ro-và-đối-phó)
14. [Checklist nghiệm thu](#14-checklist-nghiệm-thu)
15. [Phụ lục](#15-phụ-lục)

---

## 0. Tóm tắt điều hành

### 0.1. Bốn phản biện và bốn giải pháp

| # | Phản biện | Giải pháp cốt lõi | Bằng chứng sẽ có |
|---|---|---|---|
| 1 | Bài test chưa đủ để chốt mô hình | Pre-registration + factorial 2⁴ + tách kiến trúc + 3-fold OOF + lockbox một lần + CI | Bảng main effect, bảng kiến trúc, bảng OOF/lockbox, decision card |
| 2 | Số liệu còn leak | Chia lại 3 tầng theo track/sign-group; lockbox 20%; mean/std theo split; official test chỉ là phụ | Split manifest có hash, audit 131 ảnh gần trùng, test raw/clean |
| 3 | Chỉ báo số, không giải thích | Decision card + learning curve + per-class Δ + Grad-CAM + calibration + corruption | Chương 5 mới, 6 nhóm hình |
| 4 | Không rõ cấu hình nào cần | Bật/tắt từng yếu tố, factorial, aug thành phần, báo cáo cả "không có tác dụng" | Bảng 2⁴, bảng aug, kết luận giữ/bỏ từng cấu hình |

### 0.2. Ba thay đổi lớn so với bản 0

1. **Split 3 tầng thay vì train/val + test công khai:** lockbox 20% track (7.830 ảnh) là test chính; dev 80% (31.371 ảnh) dùng cho screening và 3-fold; official test 12.630 ảnh chỉ là benchmark phụ.
2. **Chọn mô hình bằng OOF theo track, không bằng một val 7.830 ảnh:** 3 fold × 3 seed, mỗi ảnh dev được dự đoán đúng một lần; CI bootstrap theo track.
3. **Mỗi kết luận phải có Δ + CI + cơ chế:** không còn câu "E3 tốt hơn E4" mà không biết vì sao; mọi yếu tố được bật/tắt riêng hoặc chạy factorial.

### 0.3. Con số khóa (dự kiến từ pool hiện tại)

| Hạng mục | Giá trị | Ghi chú |
|---|---:|---|
| Pool huấn luyện | **39.201** | 39.209 − 8 ảnh trùng MD5 với test |
| Lockbox test | **7.830 ảnh / 261 track** | 20% track, stratified theo lớp, one-shot |
| Dev | **31.371 ảnh / 1.046 track** | 80% track còn lại |
| Screening train / val | **25.131 / 6.240** | 80/20 track trong dev |
| 3-fold train / val mỗi fold | **20.541–21.360 / 10.011–10.830** | 2/3 – 1/3 track trong dev |
| Final train | **31.371 ảnh** | Toàn bộ dev; **bằng đúng cỡ train cũ** |
| Official test | **12.630 ảnh** | Phụ; clean 12.499 sau khi bỏ 131 ảnh gần trùng |
| Số run bản gọn | **~112** | ~10–11 h GPU |
| Số run tối thiểu nếu gấp | **~58** | Screening 1 seed; kết luận từ OOF/lockbox |

> Số ảnh của lockbox/screening/3-fold có thể lệch vài chục ảnh tùy seed và cách chọn track; **số chính xác phải ghi từ `outputs/splits/v2/manifest.json`**, không chép từ bảng này vào báo cáo.

### 0.4. Đọc gì trước

- Nếu có **10 phút**: đọc mục 0, 3.3, 6.2, 8.8.
- Nếu chuẩn bị **bảo vệ**: đọc mục 9.2 (kịch bản phản biện) và 8.8 (khung trả lời "vì sao CNN X").
- Nếu **triển khai**: đọc mục 3, 6, 10, 11 và file `CHECKLIST_TRIEN_KHAI.md`.
- Nếu muốn **xem trước đầu ra**: đọc `MAU_KET_QUA_GIA_DINH.md` — toàn bộ số là giả định, không chép vào báo cáo.

---

## 1. Bối cảnh và chẩn đoán 4 phản biện

### 1.1. Bốn phản biện, bản chất kỹ thuật

| # | Phản biện | Bản chất | Nếu không sửa thì bị bắt lỗi ở đâu |
|---|---|---|---|
| 1 | "Bài test chưa đủ để chốt mô hình" | Thiếu kiểm soát biến, thiếu lặp lại, thiếu CI, không có tiêu chí chọn trước | Không trả lời được "vì sao CNN-B"; không phân biệt được nhiễu seed với cải thiện thật |
| 2 | "Số liệu vẫn leak" | E1–E7 chạy trên `split_random`; val random có frame cùng track với train | Mọi so sánh ablation bị thổi phồng; kết luận "aug tốt" có thể là do leak |
| 3 | "Chỉ báo số, không giải thích" | Thiếu phân tích cơ chế: overfit/underfit, per-class, calibration, robustness | Không chứng minh được hiểu mô hình; chỉ là "chạy được" |
| 4 | "Không rõ cấu hình nào cần" | E1→E2 đổi 3 thứ; E3→E4 đổi 4 thứ; E5 chỉ 10 epoch; E7 một seed | Không thể nói BN/dropout/WD/aug có cần hay không; kết luận LR không hợp lệ |

### 1.2. Bằng chứng chẩn đoán đã có (số cũ — chỉ dùng để hiểu vấn đề)

| Hạng mục | Số | Nguồn | Ý nghĩa |
|---|---:|---|---|
| Pool sau loại trùng MD5 | 39.201 | `data/cache/cache_meta.json` | 39.209 − 8 |
| Official test | 12.630 | `cache_meta.json` | Không có track ID |
| E3 random val — accuracy / macro-F1 | 0,999362 / 0,999471 | `summary.csv` | Quá cao, không đáng tin |
| Final stratified_group s42 — accuracy / macro-F1 | 0,983525 / 0,970613 | `summary.csv` | Mốc sạch hiện có |
| Final multi-seed — accuracy / macro-F1 | 0,980843 ± 0,003052 / 0,965392 ± 0,004689 | `multiseed.json` | Dao động seed ~0,47 điểm F1 |
| E4 clean s42 — accuracy / macro-F1 | 0,976373 / 0,954453 | `configs/final.yaml` (comment) | E3 hơn E4 1,62 điểm F1, nhưng 1 seed |
| Val random có frame train cách ±1 | 7.466 / 7.841 = **95,22%** | `leak_delta.json` | Leak nặng |
| 1-NN trên val random / val group | 90,5% / 31,0% | `leak_delta.json` | Val random bị "giải sẵn" |
| Track train/val random trùng | 1.306 track | `leak_delta.json` | Gần như trùng hoàn toàn |
| Track train/val stratified_group trùng | 0 | `split_stratified_group.json` | Sạch track |
| Ảnh test gần trùng train (MAD < 5) | **131 / 12.630 = 1,04%** | Audit read-only | Ứng với 24 track train |
| Ảnh test gần trùng train (MAD < 10) | 409 / 12.630 = 3,24% | Audit read-only | Mức rộng hơn |
| Trong 131 ảnh gần trùng, dự đoán đúng | 130 / 131 = 99,24% | Audit read-only | Không phải nguồn thổi phồng lớn |
| Bỏ 131 ảnh — accuracy | 98,559% → 98,552% (**−0,007 điểm**) | Audit read-only | Ảnh hưởng nhỏ, nhưng test vẫn không độc lập |
| Class 10 trong 131 ảnh gần trùng | 77 ảnh | Audit read-only | Tập trung ở một lớp lớn |
| Số track tối thiểu mỗi lớp | 7 (lớp 0, 19, 37) | `paths_train.txt` | Đủ để 3-fold: mỗi fold 2 track |

**Kết luận chẩn đoán:**

1. Val random không dùng được cho bất kỳ kết luận nào. Số 99,9% là do leak, không phải do mô hình giỏi.
2. Val stratified_group hiện tại đã sạch track (0 overlap, 43/43 lớp), nhưng **chỉ dùng để chọn mô hình**, chưa phải test one-shot; và mới có 3 seed cho một cấu hình.
3. Official test **không track-disjoint**. 131 ảnh gần trùng chỉ làm lệch 0,007 điểm, nên không thể nói "test bị thổi phồng nặng"; nhưng không thể gọi là độc lập. Cần lockbox nội bộ.
4. Bảng ablation cũ có confound: E1→E2 đổi BN + dropout + WD; E3→E4 đổi depth + width + dropout + GAP. Không thể quy công cho một yếu tố.
5. E5a/E5b chỉ 10 epoch, không hội tụ; kết luận LR không hợp lệ. E7 một seed, không CI.

### 1.3. Mục tiêu v2

| Mã | Mục tiêu | Tiêu chí đo |
|---|---|---|
| M1 | Test độc lập thật | Lockbox 0 track/sign-group overlap; one-shot; hash |
| M2 | Chọn mô hình có căn cứ | OOF macro-F1/43 + CI; decision rule đăng ký trước |
| M3 | Tách được từng yếu tố | Factorial 2⁴ + architecture decomposition + aug components |
| M4 | Giải thích được cơ chế | Decision card + 6 phân tích (curve, per-class, Grad-CAM, calibration, corruption, error) |
| M5 | Kết luận rõ cấu hình nào cần | Mỗi yếu tố có bảng bật/tắt, Δ, CI, kết luận giữ/bỏ |
| M6 | Tái lập được | Config + seed + split hash + git commit + môi trường |

---

## 2. Nguyên tắc thiết kế

| Mã | Nguyên tắc | Cách thực hiện | Điều cấm |
|---|---|---|---|
| P1 | Split theo track/sign-group | Lockbox/dev/screening/3-fold đều group theo track; sign-group nếu làm được | Không dùng random split cho kết luận |
| P2 | Đăng ký trước | Chốt câu hỏi, endpoint, luật quyết định trước khi chạy | Không đổi luật sau khi xem kết quả |
| P3 | Tách yếu tố | Factorial cho 4 yếu tố chính; one-factor cho phần còn lại | Không đổi nhiều thứ trong một run rồi quy công |
| P4 | CI + effect size | Cluster bootstrap theo track; Δ kèm 95% CI | Không chỉ báo p-value hoặc chỉ báo điểm |
| P5 | Test một lần | Lockbox chỉ mở sau freeze | Không tune trên lockbox/test |
| P6 | Kết quả âm hợp lệ | Báo "không đủ bằng chứng" nếu CI chứa 0 | Không giữ cấu hình vì "báo cáo đẹp" |
| P7 | Tái lập | Seed, config hash, split hash, git commit, env | Không chạy tay không log |
| P8 | Công bằng | Cùng split, cùng budget, cùng early stopping khi so sánh | Không so 10 epoch với 40 epoch |
| P9 | Minh bạch test | Báo cả official test raw và clean, ghi rõ không độc lập | Không gọi official test là "test độc lập" |
| P10 | Dữ liệu quyết định | Nếu CNN-A/C thắng, chọn theo dữ liệu | Không ép chọn CNN-B vì báo cáo cũ |

---

## 3. Dữ liệu và chia split v2

### 3.1. Nguồn và pool

- Nguồn: `torchvision.datasets.GTSRB`, root `data/`, split `train`/`test`.
- Train gốc: **39.209** ảnh, 43 lớp.
- Loại **8 ảnh trùng MD5** với test (đều lớp 14, thuộc track `00023` của train; file test như `02291.ppm` không có metadata track). Còn **39.201** ảnh dùng để chia.
- File dữ liệu vẫn giữ 39.209 trên đĩa để số đếm khớp công bố; danh sách loại trừ ở `outputs/results/excluded_train_duplicates.csv`, cột `usable_for_train` trong `train_meta.csv`.
- Test: **12.630** ảnh, 43 lớp, không có track ID.

### 3.2. Định nghĩa track và sign-group

**Track** (theo GTSRB):

- Tên file train: `<track>_<frame>.ppm`, ví dụ `00023_00012.ppm`.
- Track chỉ duy nhất trong một lớp.
- Khóa nhóm: `track_key = f"{label:05d}_{track}"`.
- Mỗi track là một đoạn video ngắn (thường 30 frame, min 22, median 30) của **cùng một biển**; các frame gần như liên tiếp.

**Sign-group** (mức chặt hơn, khuyến nghị):

- Hai track khác nhau có thể là **cùng một biển vật lý** được quay ở hai đoạn/lần khác nhau.
- Xây graph: nối hai track nếu có ít nhất một cặp frame gần trùng (MAD < 5 và NCC > 0,95, hoặc pHash khoảng cách nhỏ).
- Dùng union-find để gộp thành **sign-group**; kiểm tra nhãn nhất quán trong mỗi group.
- Chia split theo sign-group thay vì track.
- **Không tự động xóa** ảnh theo hash; chỉ dùng để gộp nhóm/kiểm toán. Ảnh nghi ngờ phải xem thủ công trước khi kết luận.
- Nếu sign-group làm số nhóm của một lớp < 6, ghi rõ và xử lý theo mục 3.5.

**Official test:** không có track ID, nên chỉ có thể audit gần trùng ở mức ảnh; không thể chia lại theo track. Vì vậy official test chỉ là benchmark phụ.

### 3.3. Cấu trúc split 3 tầng

```
39.201 ảnh pool
│
├── LOCKBOX TEST        7.830 ảnh / 261 track   (20% track, stratified 43 lớp, seed L)
│   └── Chỉ mở 1 lần sau freeze; không dùng để chọn gì
│
└── DEV                31.371 ảnh / 1.046 track
    ├── SCREENING      25.131 train / 6.240 val  (80/20 track trong dev, seed S)
    │   └── Dùng cho factorial, kiến trúc, hyperparam, augmentation
    │
    └── 3-FOLD         20.541–21.360 train / 10.011–10.830 val mỗi fold
        └── Dùng cho top 2 + control; OOF 31.371 ảnh
```

| Tập | Quy tắc | Track | Ảnh dự kiến | Vai trò |
|---|---|---:|---:|---|
| Lockbox | 20% track mỗi lớp, stratified, seed 2024 | 261 | 7.830 | Test one-shot |
| Dev | Phần còn lại | 1.046 | 31.371 | Train/val |
| Screening train | 80% track của dev | 838 | 25.131 | Train để so sánh nhanh |
| Screening val | 20% track của dev | 208 | 6.240 | Val để so sánh nhanh |
| 3-fold train | 2/3 track của dev mỗi fold | ~697 | 20.541–21.360 | Train OOF |
| 3-fold val | 1/3 track của dev mỗi fold | ~349 | 10.011–10.830 | Val OOF |
| Final train | Toàn bộ dev | 1.046 | 31.371 | Train mô hình cuối |
| Official test | GTSRB test | – | 12.630 | Benchmark phụ |

**Điểm quan trọng:** mô hình cuối train trên **31.371 ảnh**, đúng bằng cỡ train của mô hình cũ. Chỉ các mô hình screening/3-fold mới train ít hơn. Vì vậy nếu số lockbox thấp hơn 98,56% cũ, nguyên nhân chính là **test sạch/khó hơn**, không phải mô hình học ít dữ liệu hơn.

### 3.4. Thuật toán chia

**Bước 1 — Lockbox:**

```
for each class c in 0..42:
    tracks_c = sorted(unique track của lớp c)
    k_c = min(max(1, round(0.2 * len(tracks_c))), len(tracks_c) - 1)
    lock_c = rng.choice(tracks_c, size=k_c, replace=False)
lockbox = union(lock_c)
dev = all - lockbox
```

**Bước 2 — Screening 80/20 trong dev:**

```
for each class c:
    d_c = sorted(track của lớp c trong dev)
    k2_c = min(max(1, round(0.2 * len(d_c))), len(d_c) - 1)
    val_c = rng.choice(d_c, size=k2_c, replace=False)
screening_val = union(val_c)
screening_train = dev - screening_val
```

**Bước 3 — 3-fold trong dev:**

```
for each class c:
    d_c = track của lớp c trong dev, sắp xếp giảm dần theo số frame
    gán round-robin vào fold 0,1,2 (cân bằng số frame)
    mỗi fold phải có >= 1 track của lớp c
```

**Ràng buộc bắt buộc:**

- `lockbox ∩ dev = ∅` theo track và theo sign-group.
- `screening_train ∩ screening_val = ∅` theo track.
- Mỗi cặp fold: train ∩ val = ∅ theo track.
- Mỗi tập val/lockbox có đủ **43/43 lớp**.
- Mỗi lớp có **≥ 1 track** trong lockbox và **≥ 1 track** trong dev.
- Nếu sign-group: mỗi lớp có **≥ 2 sign-group**; nếu không, gộp lớp đó ở mức track và ghi rõ.
- Seed: lockbox 2024; screening 42; fold 42 (có thể dùng cùng seed, vì tập dữ liệu khác nhau).

**Kiểm tra khả thi trên dữ liệu hiện tại:**

- Số track/lớp nhỏ nhất: **7** (lớp 0, 19, 37).
- Lockbox lấy 1 track/lớp này → dev còn 6 track → 3-fold mỗi fold 2 track. **Khả thi, đủ 43 lớp.**
- Nếu sign-group gộp còn 5 nhóm/lớp, một fold có thể chỉ 1 nhóm; vẫn đủ lớp nhưng phương sai cao. Ghi rõ trong manifest.

### 3.5. Kiểm tra bắt buộc sau khi chia

| # | Kiểm tra | Trường/hàm | Tiêu chí đạt |
|---|---|---|---|
| C1 | Track overlap lockbox/dev | `overlap_track` | = 0 |
| C2 | Sign-group overlap | `overlap_sign_group` | = 0 (nếu bật) |
| C3 | Đủ lớp | `classes_lockbox`, `classes_dev` | = 43 / 43 |
| C4 | Support lớp hiếm | `min_class_lockbox` | ≥ 1 track, ≥ 22 ảnh |
| C5 | Track overlap screening | `overlap_track` | = 0 |
| C6 | Track overlap từng fold | `overlap_track` | = 0 |
| C7 | Mỗi fold đủ lớp | `classes_fold` | = 43 |
| C8 | Không có test trong train | `test_in_train` | = 0 |
| C9 | Hash split | `sha256` | Ghi vào manifest + config |
| C10 | mean/std theo split | `mean`, `std` | Tính trên train của split, không dùng toàn cục |
| C11 | MD5 trùng trong train | `md5_duplicates` | = 0 (8 ảnh đã loại) |
| C12 | Ảnh gần trùng trong lockbox | audit MAD < 5 | = 0 so với train |

**Đầu ra:** `outputs/splits/v2/manifest.json` chứa:

```json
{
  "pool_size": 39201,
  "lockbox": {"n": 7830, "n_tracks": 261, "seed": 2024, "sha256": "..."},
  "dev": {"n": 31371, "n_tracks": 1046},
  "screening": {"train": 25131, "val": 6240, "seed": 42},
  "folds": [{"fold": 0, "train": 20541, "val": 10830}, ...],
  "checks": {"overlap_track": 0, "overlap_sign_group": 0, "classes_lockbox": 43, ...},
  "mean_std": {"lockbox": {"mean": [...], "std": [...]}, ...}
}
```

### 3.6. Vì sao chọn lockbox 20%

| Lockbox | Lockbox test | Dev | Screening train | 3-fold train | Final train | Nhận xét |
|---:|---:|---:|---:|---:|---:|---|
| 15% | 5.880 | 33.321 | 26.657 | 22.214 | 33.321 | Test nhỏ, CI rộng; final train nhiều hơn |
| **20%** | **7.830** | **31.371** | **25.131** | **20.541–21.360** | **31.371** | **Cân bằng; final train bằng cũ** |
| 25% | 9.800 | 29.401 | 23.521 | 19.601 | 29.401 | Test lớn hơn; final train giảm 6% |
| 30% | 11.760 | 27.441 | 21.953 | 18.294 | 27.441 | Test gần official; final train giảm 12,5% |

**Chọn 20%** vì:

- Mô hình cuối vẫn có **31.371 ảnh**, không bị yếu đi vì thiếu dữ liệu.
- Lockbox 7.830 đủ để ước lượng macro-F1/43 với CI chấp nhận được (dự kiến ±0,5–0,8 điểm; sẽ đo).
- 30% làm final train giảm còn 27.441, có thể tạo ra "thua" giả do thiếu dữ liệu, không phải do kiến trúc.

### 3.7. Official test: raw và clean

- Official test **12.630** ảnh; **131 ảnh (1,04%)** gần trùng ảnh train (MAD < 5), ứng với **24 track train**; **409 ảnh (3,24%)** ở MAD < 10.
- Trong 131 ảnh, model đúng **130/131 (99,24%)**; bỏ 131 ảnh, accuracy **98,559% → 98,552%** (−0,007 điểm). Class 10 chiếm **77/131**.
- Kết luận: official test **không bị thổi phồng đáng kể bởi near-duplicate**, nhưng **không track-disjoint**. Vì vậy:
  - **Raw:** 12.630 ảnh — báo cáo để so literature.
  - **Clean:** 12.499 ảnh — báo cáo như kiểm tra độ nhạy.
  - **Không** dùng official test để chọn mô hình/epoch/threshold.
- Tùy chọn (không bắt buộc): retrain loại 24 track train bị nghi trùng, test trên 12.630; đây là **sensitivity analysis**, không phải kết quả chính. Nếu làm, chỉ 1 run, ghi rõ.

---

## 4. Đăng ký trước (pre-registration)

> Mục này phải được **chốt và commit** trước khi chạy run đầu tiên. Nếu phải sửa, tạo `PRE_REGISTRATION_V2_v2.md` và ghi lý do; không sửa im lặng.

### 4.1. Câu hỏi nghiên cứu

| Mã | Câu hỏi | Cách trả lời |
|---|---|---|
| RQ1 | Kiến trúc nào cho macro-F1/43 tốt nhất trên track-disjoint? | So sánh CNN-A/B/C + depth/width/GAP trên OOF |
| RQ2 | BN, dropout, WD, aug có thực sự cần? | Factorial 2⁴, main effect + interaction |
| RQ3 | Aug thành phần nào có tác dụng? | 9 profile aug + corruption benchmark |
| RQ4 | LR, class weight, input size, batch size ảnh hưởng thế nào? | One-factor, 40 epoch, 3 seed |
| RQ5 | Cơ chế nào giải thích kết quả? | Learning curve, per-class, Grad-CAM, calibration, corruption |

### 4.2. Giả thuyết (có hướng, có ngưỡng)

| Mã | Giả thuyết | Ngưỡng "ủng hộ" | Nếu bị bác bỏ |
|---|---|---|---|
| H1 | CNN-B (BN+DO+WD) > CNN-A | Δ macro-F1 ≥ 0,3 điểm, CI > 0 | Báo cáo; có thể BN không cần ở quy mô này |
| H2 | Aug không tăng clean accuracy đáng kể nhưng tăng corruption accuracy | Clean Δ < 0,3; corruption Δ ≥ 0,5, CI > 0 | Nếu aug tăng cả hai → giữ; nếu không tăng cả hai → bỏ |
| H3 | CNN-C (4 khối + GAP) ≤ CNN-B | Δ ≤ 0 hoặc CI chứa 0 | Nếu C thắng → chọn C; giải thích depth/GAP |
| H4 | 3 khối ≥ 4 khối | Δ 3-4 ≥ 0 | Báo cáo; chọn theo dữ liệu |
| H5 | LR 1e-3 ≥ 1e-4 (cách 10 lần) | Δ ≥ 0,3 | Nếu 1e-4 tốt hơn → ghi nhận |
| H6 | Class weight tăng recall lớp hiếm và macro-F1 | Δ macro-F1 ≥ 0,3, CI > 0 | Nếu giảm → bỏ; báo trade-off |
| H7 | Input 96 không hơn 64 quá 0,3 | Δ 96-64 < 0,3 | Nếu 96 thắng rõ → cân nhắc latency |
| H8 | Width 48/96/192 không đáng so với 32/64/128 | Δ < 0,3 so với 32/64/128 | Nếu thắng rõ → chọn theo dữ liệu |

> Đây là **giả thuyết**, không phải kết luận. Kết quả trái giả thuyết vẫn là kết quả tốt nếu quy trình đúng.
>
> **Bản gọn** chỉ so LR **1e-4 vs 1e-3** (cách 10 lần); không khảo sát các mức LR trung gian/warmup, dropout 0,2/0,5, WD 1e-3, label smoothing, batch size, input 48. Chỉ làm exploratory nếu còn thời gian; không dùng để kết luận chính.

### 4.3. Endpoint

| Loại | Metric | Cách tính | Ghi chú |
|---|---|---|---|
| **Chính** | Macro-F1/43 | `f1_score(labels=range(43), average="macro", zero_division=0)` | Trên OOF 31.371 ảnh; xác nhận bằng lockbox |
| Phụ | Accuracy | Top-1 | Báo cáo cùng CI |
| Phụ | Per-class F1, precision, recall, support | `classification_report` | Bắt buộc cho lớp hiếm |
| Phụ | Top-3 accuracy | Logits | Demo |
| Phụ | ECE, Brier | 15 bin | Calibration |
| Phụ | Corruption accuracy | 6 loại × 3 mức | Robustness |
| Phụ | Params, latency | `count_parameters`, đo model-only | Chọn tie-break |
| Phụ | Train time, GPU mem | `metrics.csv` | Minh bạch chi phí |

### 4.4. So sánh chính và hiệu chỉnh đa so sánh

**8 so sánh chính (family):**

1. BN on vs off (trung bình qua các mức còn lại).
2. Dropout 0,3 vs 0.
3. WD 1e-4 vs 0.
4. Aug on vs off.
5. GAP on vs off (3 khối).
6. Depth 4 vs 3.
7. Width lớn (48/96/192) vs 32/64/128.
8. LR 1e-3 vs 1e-4 (cách 10 lần).

**Hiệu chỉnh:** Holm–Bonferroni cho 8 so sánh chính. Các so sánh còn lại (class weight, input, aug components) là **exploratory**, ghi rõ "chưa hiệu chỉnh" và không dùng để tuyên bố thắng/thua tuyệt đối.

### 4.5. Luật quyết định (chốt trước)

Cho Δ = macro-F1(config) − macro-F1(reference), CI 95% cluster bootstrap theo track:

| Tình huống | Kết luận | Hành động |
|---|---|---|
| Δ ≥ +0,3 và CI_low > 0 | **Tốt hơn** | Giữ cấu hình |
| CI nằm hoàn toàn trong [−0,3, +0,3] | **Tương đương** | Chọn ít params/nhanh/đơn giản hơn |
| CI chứa 0 nhưng vượt ra ngoài ±0,3 | **Chưa kết luận** | Tăng fold/seed hoặc báo cáo là inconclusive |
| Δ ≤ −0,3 và CI_high < 0 | **Kém hơn** | Bỏ cấu hình |
| Aug clean-neutral, corruption Δ ≥ +0,5, CI > 0 | **Giữ có điều kiện** | Giữ nếu mục tiêu demo/robustness; ghi rõ đánh đổi |
| Aug không tăng clean, không tăng corruption | **Bỏ** | Giảm chi phí train/demo |
| Class weight tăng macro-F1 nhưng giảm accuracy > 0,5 | **Trade-off** | Chọn theo macro-F1 (endpoint chính), báo accuracy |
| GAP/4 khối tương đương | **Chọn GAP/3 khối** | Ít params hơn, ít overfit hơn |

**Tie-break khi tương đương:** (1) ít params hơn; (2) latency thấp hơn; (3) ít aug hơn; (4) ít epoch hơn; (5) đơn giản hơn để giải thích.

### 4.6. Luật dừng và tính hợp lệ của run

- Tối đa **40 epoch**; early stopping **patience 7** theo val macro-F1.
- Run **không hợp lệ** nếu: loss NaN/Inf; > 10% epoch NaN; không có checkpoint; val thiếu lớp mà không ghi rõ; config/split hash không khớp.
- Run **hợp lệ nhưng thất bại** (ví dụ NaN hoặc phân kỳ) vẫn phải báo cáo, không xóa.
- Không dùng test/lockbox để early stop.
- Không chạy lại run đã có nếu không có lý do; nếu chạy lại do bug, tạo `_v2` và ghi log.

### 4.7. Freeze

Trước khi mở lockbox:

1. Chốt config winner + seed + epoch.
2. Commit code + config + split manifest.
3. Ghi `git_commit`, `config_sha256`, `split_sha256`.
4. Mở lockbox **một lần**; ghi log thời gian, người chạy, lệnh.
5. Nếu phát hiện bug sau khi mở: tạo `lockbox_result_v2.json`, không ghi đè; ghi rõ lý do.

---

## 5. Giao thức huấn luyện cố định

### 5.1. Config gốc (dùng cho mọi run, trừ khi đang test chính yếu tố đó)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Optimizer | AdamW | |
| Learning rate | 1e-3 | L1 test (1e-4, cách 10 lần) |
| Weight decay | 1e-4 | F test (0 vs 1e-4) |
| Batch size | 128 | Không khảo sát |
| Epochs | 40 | **Không so 10 epoch** |
| Scheduler | Cosine annealing | Không warmup |
| Warmup | 0 | Không khảo sát |
| AMP | Bật khi CUDA | |
| Early stopping | Patience 7, theo val macro-F1 | |
| Label smoothing | 0 | Không khảo sát |
| Class weight | None | CW1 test (inv_sqrt) |
| Input size | 64×64 RGB | I2 test (96) |
| Mean/std | Tính trên train của từng split | Không dùng toàn cục |
| Dropout | 0,3 | F test (0 vs 0,3) |
| BN | Bật | F test |
| Aug | Bật (affine + color + perspective) | G test |
| Seed | 42, 123, 2024 | Screening/confirmation |
| `num_workers` | 4 | Windows: `if __name__ == "__main__":` |
| Deterministic | Bật nếu không làm chậm quá 20% | Ghi rõ |

### 5.2. Kiến trúc

| Model | Khối | Widths | BN | Dropout | GAP | FC hidden | Params |
|---|---|---|---:|---:|---|---:|---:|
| CNN-A | 3 | 32/64/128 | Không | 0 | Không | 256 | **2.201.707** |
| CNN-B | 3 | 32/64/128 | Có | 0,3 | Không | 256 | **2.201.931** |
| CNN-C | 4 | 32/64/128/256 | Có | 0,4 | Có | – | **399.947** |

- Forward: `(B, 3, 64, 64)` → logits `(B, 43)`, **không softmax trong model**.
- Khi BN bật: conv `bias=False`; khi BN tắt: conv `bias=True`.
- Không dùng pretrained cho mô hình chính.
- Không đổi kiến trúc mà không đổi `config_id`.

### 5.3. Augmentation được phép

| Profile | Thành phần | Tham số |
|---|---|---|
| `none` | – | – |
| `affine` | RandomAffine | degrees ±10, translate 0,1, scale 0,9–1,1, shear 5 |
| `color` | ColorJitter | brightness 0,2, contrast 0,2, saturation 0,2, hue 0,02 |
| `perspective` | RandomPerspective | distortion 0,1, p=0,3 |
| `all` | affine + color + perspective | Mặc định |

**Bản gọn** chỉ dùng 3 profile đơn (`affine`, `color`, `perspective`) và `all`; bỏ các tổ hợp trung gian để giảm run.

**Cấm:** HorizontalFlip, VerticalFlip, xoay 90/180, hoặc bất kỳ biến đổi nào làm đổi nghĩa biển. Có test tự động chặn flip.

### 5.4. Checkpoint và log

Mỗi run tạo:

```
outputs/runs/v2_<config_id>_s<seed>[_f<fold>]/
├── config.yaml
├── env.json
├── metrics.csv
├── best.pt
├── curves.png
└── split_manifest.json   # bản sao hoặc hash
```

`best.pt` là dict:

```python
{
    "state_dict": model.state_dict(),
    "config": {...},
    "class_names": {...},
    "epoch": int,
    "val_macro_f1": float,
    "git_commit": str,
    "torch_version": str,
    "config_sha256": str,
    "split_sha256": str,
}
```

`metrics.csv` đúng thứ tự cột:

```
epoch,lr,train_loss,train_acc,val_loss,val_acc,val_macro_f1,epoch_time_s,gpu_mem_mb
```

Ghi 1 dòng/epoch, flush ngay.

### 5.5. Tái lập

- Gọi `set_seed(seed)` ở đầu entrypoint; seed cả DataLoader generator.
- `torch.use_deterministic_algorithms` nếu không ảnh hưởng tốc độ quá 20%.
- Lưu `env.json`: Python, torch, CUDA, GPU, OS.
- Không hard-code đường dẫn tuyệt đối; dùng `pathlib.Path` từ root repo.
- Windows: `if __name__ == "__main__":` khi `num_workers > 0`.

### 5.6. Chi phí dự kiến

| Loại run | Train | Thời gian/run | Ghi chú |
|---|---|---|---|
| Screening (25.131) | 80% dev | ~5,0–5,5 phút | 40 epoch |
| 3-fold (20.541–21.360) | 2/3 dev | ~4,3–4,6 phút | 40 epoch |
| Final (31.371) | all dev | ~6,0–6,5 phút | 40 epoch |
| Lockbox inference | – | ~10–20 giây | 7.830 ảnh |
| Official test inference | – | ~15–25 giây | 12.630 ảnh |
| HOG+SVC | 25.131 | ~10–30 phút | Tùy C search |

Tổng bản gọn ~112 run × ~5,5 phút ≈ **10–11 giờ GPU**; cộng phân tích/lỗi ≈ **1–2 ngày làm việc**.

### 5.7. mean/std

- Tính trên **train của từng split** (screening, từng fold, dev-all).
- Lưu vào config của run và `manifest.json`.
- Không dùng mean/std của random split cũ.
- Với input 96/48, tạo cache riêng; không ghi đè cache 64.

---

## 6. Ma trận thí nghiệm

> **Bản gọn cho bài tập lớn sinh viên.** Chỉ giữ các thí nghiệm trả lời trực tiếp 4 phản biện. Baseline chỉ B1. Tổng ~112 run, ~10–11 h GPU. Các thí nghiệm exploratory đã cắt được ghi ở 6.4/6.5; không tự chạy lại nếu không có yêu cầu.
>
> File `MA_TRAN_RUN.csv` chứa danh sách config để chạy. Mục này giải thích thiết kế và cách đọc.

### 6.0. Baseline

**Chỉ giữ B1** để giảm run; bỏ B0 majority và B1b refit.

| ID | Mô hình | Train | Val | Số run | Mục đích |
|---|---|---|---:|---:|---|
| B1 | HOG 64×64 + LinearSVC, C chọn trên val | screening train | screening val | 1 | Baseline cổ điển |

- HOG dùng ảnh xám 64×64; `StandardScaler` fit trên train; `LinearSVC` OvR; C chọn trên **screening val**, không chọn trên lockbox.
- Không refit B1 trên dev, không test B1 trên lockbox. So sánh với CNN dùng screening val; kết luận cuối dùng OOF/lockbox của CNN.

### 6.1. Block R — tái lập bảng cũ trên split mới

| ID | Cấu hình | Tương ứng bảng cũ | Số seed | Ghi chú |
|---|---|---|---:|---|
| R1 | CNN-A, không BN/DO/WD/aug | E1 | 3 | = F00 (alias) |
| R2 | CNN-B, BN+DO+WD, không aug | E2 | 3 | = F07 (alias) |
| R3 | CNN-B + aug | E3 | 3 | = F15 (alias) |
| R4 | CNN-C, 4 khối + GAP + DO 0,4 + aug | E4 | 3 | = D3 (alias) |

**Mục đích:** chứng minh bảng cũ thay đổi thế nào khi hết leak. **Không chạy trùng:** R1–R3 dùng lại F00/F07/F15; R4 dùng D3. Chỉ cần chạy 3 seed cho R4 nếu chưa có trong D.

### 6.2. Block F — factorial 2⁴ (phần quan trọng nhất)

**Mục đích:** trả lời "BN/dropout/WD/aug có cần không" bằng main effect + interaction, không bị confound.

**Cố định:** 3 khối 32/64/128, FC 256, GAP off, input 64, batch 128, lr 1e-3, 40 epoch, patience 7, 3 seed.

| ID | BN | Dropout | WD | Aug | So với | Ý nghĩa |
|---|---|---|---|---|---|---|
| F00 | off | 0 | 0 | off | CNN-A gốc | Mốc "chưa tối ưu" |
| F01 | **on** | 0 | 0 | off | F00 | Chỉ bật BN |
| F02 | off | **0,3** | 0 | off | F00 | Chỉ bật dropout |
| F03 | on | 0,3 | 0 | off | F00 | BN + dropout |
| F04 | off | 0 | **1e-4** | off | F00 | Chỉ bật WD |
| F05 | on | 0 | 1e-4 | off | F00 | BN + WD |
| F06 | off | 0,3 | 1e-4 | off | F00 | Dropout + WD |
| F07 | on | 0,3 | 1e-4 | off | F00 | **= E2 cũ** |
| F08 | off | 0 | 0 | **on** | F00 | Chỉ bật aug |
| F09 | on | 0 | 0 | on | F01 | BN + aug |
| F10 | off | 0,3 | 0 | on | F02 | Dropout + aug |
| F11 | on | 0,3 | 0 | on | F03 | BN + dropout + aug |
| F12 | off | 0 | 1e-4 | on | F04 | WD + aug |
| F13 | on | 0 | 1e-4 | on | F05 | BN + WD + aug |
| F14 | off | 0,3 | 1e-4 | on | F06 | Dropout + WD + aug |
| F15 | on | 0,3 | 1e-4 | on | F07 | **= E3 cũ** |

**Số run:** 16 config × 3 seed = **48 run**.

**Cách tính main effect:**

```
Effect(BN)   = mean(F1 của 8 config BN=on)  − mean(F1 của 8 config BN=off)
Effect(DO)   = mean(F1 của 8 config DO=0.3) − mean(F1 của 8 config DO=0)
Effect(WD)   = mean(F1 của 8 config WD=1e-4)− mean(F1 của 8 config WD=0)
Effect(Aug)  = mean(F1 của 8 config Aug=on) − mean(F1 của 8 config Aug=off)
```

**Interaction (ví dụ):**

```
BN × Aug = (F15 − F07) − (F08 − F00)
DO × WD  = (F06 − F04) − (F02 − F00)
```

Nếu interaction lớn (≥ 0,3 điểm), **không** được nói "aug giúp X điểm" chung chung; phải nói "aug giúp X điểm khi có BN, nhưng Y điểm khi không BN".

**Bảng kết quả cần điền (16 dòng):**

| ID | BN | DO | WD | Aug | Val acc (mean±sd) | Macro-F1 (mean±sd) | Δ vs ref [CI] | Train−val gap | Best epoch | Params | Time |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| F00 | | | | | | | | | | | |
| ... | | | | | | | | | | | |

**Kết luận bắt buộc:** một bảng "main effect" 4 dòng:

| Yếu tố | Mức bật | Mức tắt | Δ [CI] | p (Holm) | Kết luận |
|---|---|---|---:|---:|---|
| BN | | | | | giữ/bỏ/không đủ bằng chứng |
| Dropout | | | | | |
| WD | | | | | |
| Aug | | | | | |

### 6.3. Block D — tách kiến trúc

**Mục đích:** E3→E4 cũ đổi 4 thứ; block này tách depth, GAP, width. D3 giữ dropout 0,4 làm legacy, không dùng để kết luận cơ chế.

**Cố định:** BN on, WD 1e-4, aug on, input 64, batch 128, lr 1e-3, 40 epoch, 3 seed.

| ID | Khối | Widths | GAP | Dropout | So với | Trả lời |
|---|---|---|---:|---:|---|---|
| D4 | 3 | 32/64/128 | off | 0,3 | E3 | Mốc (= F15) |
| D1 | 3 | 32/64/128 | **on** | 0,3 | D4 | **Chỉ thêm GAP** |
| D2 | **4** | 32/64/128/256 | off | 0,3 | D4 | **Chỉ thêm depth** |
| D3 | 4 | 32/64/128/256 | on | **0,4** | E4 | Cả hai (E4 cũ, legacy) |
| W1 | 3 | 16/32/64 | off | 0,3 | D4 | Width nhỏ |
| W2 | 3 | 48/96/192 | off | 0,3 | D4 | Width lớn |

**Số run:** 5 config mới × 3 seed = **15 run** (D4 = F15, không chạy lại).

**Báo cáo:**
- Bảng `params vs macro-F1` (Pareto).
- Nếu D1 tương đương D4 trong CI → chọn D1 (5× ít params).
- Nếu D2 kém D4 → kết luận 4 khối overfit; kèm train acc / val gap.
- Nếu W1 kém, W2 không hơn D4 → kết luận 2,2M đủ.
- D3 chỉ để map E4 cũ; D3 đổi cả depth/GAP/dropout 0,4 nên **không dùng để kết luận cơ chế**. D1/D2 mới tách GAP/depth.

### 6.4. Block H — siêu tham số huấn luyện

**Cố định:** CNN-B (3 khối 32/64/128, BN, dropout 0,3, WD 1e-4, aug on), trừ yếu tố đang test.

| ID | Yếu tố | Giá trị | Số run | Câu hỏi |
|---|---|---|---:|---|
| L1 | LR | 1e-4 | 3 | So với 1e-3 (cách 10 lần)? |
| L3 | LR | 1e-3 | – | Mốc (= F15) |
| CW1 | Class weight | inv_sqrt | 3 | Có giúp lớp hiếm? |
| I2 | Input size | 96 | 3 | Được bao nhiêu điểm? |

**Số run:** 3 config mới × 3 seed = **9 run** (L3 = F15).

**Bắt buộc:** L1 chạy **đủ 40 epoch**. Không so 10 epoch với 40 epoch.

**Đã bỏ để giữ quy mô sinh viên:** L2, L4/L5 (warmup), P1/P2 (dropout 0,2/0,5), W4 (WD 1e-3), CW2 (inv_freq), LS1 (label smoothing), I1 (input 48), BS1/BS2 (batch 64/256). Chỉ làm exploratory nếu còn thời gian; không dùng để kết luận chính.

### 6.5. Block G — augmentation thành phần

**Mục đích:** trả lời "aug có cần không, phần nào có tác dụng".

**Cố định:** CNN-B, BN, dropout 0,3, WD 1e-4, input 64, 40 epoch, 3 seed.

| ID | Profile | Thành phần | So với |
|---|---|---|---|
| G0 | none | – | = F07 |
| G1 | affine | hình học | G0 |
| G2 | color | màu/sáng | G0 |
| G3 | perspective | phối cảnh | G0 |
| G5 | all | affine + color + perspective | = F15 |

**Số run:** 3 config mới × 3 seed = **9 run** (G0 = F07, G5 = F15).

**Đã bỏ:** G4, G6, G7, G8 (các tổ hợp trung gian và color mạnh) để giảm run; chỉ giữ 3 profile đơn + `all`.

**Corruption benchmark (không train thêm):**

| Loại | Mức 1 | Mức 2 | Mức 3 |
|---|---|---|---|
| Blur Gaussian | σ 0,5 | σ 1,0 | σ 2,0 |
| Noise Gaussian | σ 5 | σ 10 | σ 20 |
| Brightness | −20% | +20% | +40% |
| Contrast | 0,8 | 0,5 | 0,3 |
| JPEG | q 50 | q 30 | q 15 |
| Occlusion | 10% | 20% | 30% |

- Đánh giá trên **screening val** cho tất cả config G; lockbox chỉ dùng cho final sau freeze.
- Báo cáo accuracy từng điều kiện + trung bình theo loại + Δ so với clean.
- Kỳ vọng: aug tăng robustness; nếu không, kết luận "aug không cần" là hợp lệ.

### 6.6. Block X — xác nhận OOF và lockbox

**Bước 1 — chọn top 2 + control từ screening:**

- Sắp xếp theo mean macro-F1 (3 seed).
- Lấy các config trong khoảng 0,3 điểm so với tốt nhất (nhóm tương đương).
- Trong nhóm đó, chọn **top 2** theo tie-break (params, latency, đơn giản).
- Thêm **1 control** (F07; nếu F07 đã nằm trong top 2 thì dùng F00) để neo.
- Ghi rõ lý do chọn trước khi chạy OOF.

**Bước 2 — 3-fold × 3 seed:**

- Mỗi config: 3 fold × 3 seed = 9 run.
- Top 2 + control = 3 config → **27 run**.
- Mỗi ảnh dev được dự đoán đúng 1 lần trong mỗi seed (OOF).
- Tính macro-F1/43 trên OOF; CI cluster bootstrap theo track.

**Bước 3 — chọn winner:**

- Theo luật 4.5 trên OOF.
- Nếu tương đương: chọn ít params/nhanh/đơn giản.
- Freeze config + epoch = median best epoch từ 9 run (hoặc theo fold tương ứng).

**Bước 4 — final + lockbox:**

- Train trên **toàn bộ dev** (31.371), 3 seed, epoch cố định từ OOF.
- Không early stop trên lockbox.
- Đánh giá lockbox **một lần** cho mỗi seed; báo cáo mean ± sd và CI.
- Đánh giá official test raw/clean cho cùng model (phụ).

**Số run:** 27 OOF + 3 final = **30 run**.

### 6.7. Tổng hợp số run

| Block | Config | Seed/fold | Run | GPU (≈5,5 phút/run) | Ưu tiên |
|---|---:|---:|---:|---:|---|
| Baseline | 1 | 1 | 1 | 0,1 h | Bắt buộc |
| F factorial | 16 | 3 | 48 | 4,5 h | Bắt buộc |
| D architecture | 5 | 3 | 15 | 1,4 h | Bắt buộc |
| H hyperparam | 3 | 3 | 9 | 0,8 h | Nên có |
| G augmentation | 3 | 3 | 9 | 0,8 h | Bắt buộc |
| X confirmation | 3 | 9 | 27 | 2,5 h | Bắt buộc |
| Final lockbox | 1 | 3 | 3 | 0,3 h | Bắt buộc |
| **Tổng** | | | **112** | **~10–11 h** | |

**Nếu rất gấp (~58 run, ~5–6 h):** chạy screening 1 seed (F16 + D5 + H3 + G3 = 27 run), OOF vẫn 3 fold × 3 seed (27), final 3, B1 1. Chỉ dùng screening để go/no-go; kết luận cuối từ OOF/lockbox; ghi rõ trong báo cáo.

### 6.8. Thứ tự chạy

| Phase | Việc | GPU | Đầu ra |
|---|---|---|---|
| 0 | Chốt protocol, chia split, audit, hash, test | Không | `manifest.json`, `PRE_REGISTRATION` commit |
| 1 | Baseline + F factorial (3 seed) | ~4–5 h | Bảng main effect sơ bộ |
| 2 | D + G + H (3 seed) | ~3 h | Bảng kiến trúc/aug/hyperparam |
| 3 | Chọn top 2 + control; 3-fold OOF; freeze; final; lockbox | ~2–3 h | OOF, lockbox result |
| 4 | Phân tích thống kê, hình, viết báo cáo | Không | Báo cáo v2 |

### 6.9. Mapping bảng cũ → v2

| Bảng cũ | ID v2 | Thay đổi |
|---|---|---|
| E0a majority | – | Bỏ; baseline chỉ B1 |
| E0b HOG+SVC | B1 | C chọn trên screening val; không refit lockbox |
| E1 CNN-A | F00 / R1 | Split sạch, 3 seed |
| E2 CNN-B | F07 / R2 | Split sạch, 3 seed |
| E3 CNN-B+aug | F15 / R3 | Split sạch, 3 seed |
| E4 CNN-C+GAP+aug | D3 / R4 | D3 là legacy; D1/D2 tách GAP/depth |
| E5a lr 1e-4 (10 epoch) | L1 | **40 epoch**, cách 10 lần so với 1e-3 |
| E5b lr cao (10 epoch) | – | Bỏ; không khảo sát mức LR cao/warmup |
| E7 class weight | CW1 | 3 seed, CI; bỏ CW2 |

> Không được trộn số cũ với số v2 trong cùng một bảng kết luận. Số cũ chỉ để trong phụ lục `legacy_leaky`.
>
> **Đã bỏ khỏi bản gọn:** B0, B1b, C1, W3, L2, L4, L5, P1, P2, W4, CW2, LS1, I1, BS1, BS2, G4, G6, G7, G8. Không cần chạy lại nếu không có yêu cầu.

---

## 7. Phân tích thống kê

### 7.1. Metric và cách tính

| Metric | Công thức / tham số | Ghi chú |
|---|---|---|
| Accuracy | `accuracy_score(y_true, y_pred)` | Top-1 |
| Macro-F1/43 | `f1_score(y_true, y_pred, labels=range(43), average="macro", zero_division=0)` | **Endpoint chính** |
| Per-class F1 | `classification_report(..., labels=range(43), zero_division=0)` | Kèm support |
| Top-3 | Từ logits, lấy 3 lớp cao nhất | Demo |
| ECE | 15 bin, `|acc − conf|` trung bình theo bin | Calibration |
| Brier | Đa lớp: trung bình tổng bình phương xác suất | Calibration |
| Corruption accuracy | Accuracy trên từng điều kiện | Robustness |
| Params | `count_parameters` | Tie-break |
| Latency | Model-only, batch 1, 100 lần, median | Ghi rõ phạm vi đo |

**Quy tắc bắt buộc:**

- Macro-F1 luôn tính trên **đủ 43 lớp**. Nếu split thiếu lớp, báo kèm `macro_f1_present_classes`, support, và trần lý thuyết `số lớp có mặt / 43`; không so trực tiếp F1/43 giữa hai split khác support.
- Confusion matrix chuẩn hóa theo hàng; ghi rõ nếu không chuẩn hóa.
- Prediction lưu `predictions.csv` cho từng run; hash file để kiểm tra.

### 7.2. CI bằng cluster bootstrap theo track

Frame trong cùng track tương quan mạnh; bootstrap theo ảnh sẽ cho CI hẹp giả. Dùng **track** làm đơn vị resample:

```
tracks = unique(track_key của eval set)
for b in 1..B:                     # B = 2.000
    sampled = rng.choice(tracks, size=len(tracks), replace=True)
    frames = tất cả frame thuộc các track trong sampled
    f1_b = macro_f1_43(y_true[frames], y_pred[frames])
CI = percentile(f1_b, [2.5, 97.5])
```

**So sánh cặp:**

```
for b in 1..B:
    sampled = cùng một mẫu track cho cả hai model
    delta_b = f1_b(model A) - f1_b(model B)
CI_delta = percentile(delta_b, [2.5, 97.5])
p_boot = 2 * min(mean(delta_b <= 0), mean(delta_b >= 0))
```

- B = 2.000; nếu kết quả gần biên, chạy lại B = 10.000 để kiểm tra.
- Báo cáo CI percentile; có thể kèm BCa nếu phân phối lệch.
- Với per-class, báo support; CI per-class chỉ là phụ, không dùng để tuyên bố thắng/thua.

### 7.3. Kiểm định

| Mục đích | Cách làm | Vai trò |
|---|---|---|
| So sánh 2 cấu hình trên cùng eval set | Paired cluster bootstrap của Δ | **Chính** |
| Kiểm tra cặp bất đồng | McNemar trên prediction từng ảnh | Phụ; giải thích hướng |
| So sánh qua 3 seed × 3 fold | Wilcoxon signed-rank trên 9 cặp Δ | Phụ; kiểm tra tính nhất quán |
| Nhiều so sánh | Holm–Bonferroni cho 8 so sánh chính | Bắt buộc |
| Tương đương | CI nằm trong ±0,3 (hoặc TOST với biên ±0,3) | Báo cáo rõ |

**Lưu ý:** McNemar giả định các quan sát độc lập; frame trong track không độc lập. Vì vậy McNemar chỉ là phụ, không dùng làm p-value chính. Nếu muốn chặt, dùng cluster-adjusted McNemar bằng bootstrap.

### 7.4. Phân rã phương sai

Với 3 seed × 3 fold cho mỗi cấu hình, ước lượng:

```
y_ijk = mu + seed_i + fold_j + (seed × fold)_ij + e_ijk
```

Báo cáo:

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Phương sai do seed | `var_seed` | Thay đổi do khởi tạo/thứ tự dữ liệu |
| Phương sai do fold | `var_fold` | Thay đổi do cách chia track |
| Phương sai dư | `var_resid` | Nhiễu còn lại |
| ICC | `var_seed / (var_seed + var_fold + var_resid)` | Tỷ lệ do seed |

- Nếu `var_fold > var_seed`: **multi-seed cũ chưa đủ**; phải báo cáo theo fold, không chỉ mean ± sd qua seed.
- Nếu `var_seed` lớn: kết luận phải kèm CI qua seed; không so sánh 1 seed.
- Dùng ANOVA Type II hoặc `statsmodels` MixedLM; ghi rõ phương pháp.

### 7.5. Đa so sánh

- **8 so sánh chính** (mục 4.4): Holm.
- **Exploratory**: class weight, input size, batch, label smoothing, aug components. Ghi rõ "chưa hiệu chỉnh"; không dùng để tuyên bố "tốt nhất" tuyệt đối.
- Nếu một so sánh exploratory muốn thành kết luận, phải đăng ký lại và chạy xác nhận trên OOF/lockbox.

### 7.6. Mẫu bảng báo cáo

| Cấu hình | Macro-F1/43 (mean ± sd) | 95% CI | Δ vs ref | Δ 95% CI | p (Holm) | Kết luận |
|---|---:|---:|---:|---:|---:|---|
| F00 | | | – | – | – | |
| F07 | | | | | | |
| F15 | | | | | | |

**Quy tắc trình bày:**

- Luôn có mean ± sd **và** CI.
- Δ ghi theo **điểm %** (1 điểm % = 0,01).
- Nếu CI chứa 0, không dùng từ "tốt hơn"; dùng "tương đương" hoặc "chưa kết luận".
- Nếu p > 0,05 nhưng CI rộng, không kết luận "không có tác dụng"; chỉ nói "không đủ bằng chứng".

### 7.7. Độ mạnh thống kê

- Lockbox 7.830 ảnh / 261 track; CI dự kiến rộng hơn official test khoảng **1,27 lần** nếu cùng variance.
- OOF 31.371 ảnh / 1.046 track; CI dự kiến hẹp hơn, nhưng không độc lập hoàn toàn giữa các fold (cùng pool).
- Ngưỡng hiệu quả nhỏ nhất quan tâm: **0,3 điểm %**. Nếu CI rộng hơn ±0,5, không kết luận thắng/thua; báo cáo "inconclusive" và tăng fold/seed nếu còn thời gian.
- Không tính power trước vì phụ thuộc clustering; sẽ đo CI thực tế và phản hồi.

---

## 8. Giải thích cơ chế

> Mục tiêu: trả lời "vì sao" bằng **cơ chế**, không chỉ bằng điểm. Mỗi kết luận phải có ít nhất một hình/bảng cơ chế đi kèm.

### 8.1. Decision card

Mỗi yếu tố có một "thẻ quyết định":

| Trường | Nội dung |
|---|---|
| Yếu tố | BN / dropout / WD / aug / depth / GAP / width / LR / class weight / input / batch |
| Mức đã thử | Ví dụ dropout 0; 0,2; 0,3; 0,5 |
| Kết quả | Mean ± sd, CI cho từng mức |
| Δ so với mốc | Δ điểm % + CI |
| Cơ chế | 1–3 câu: vì sao yếu tố này có/không có tác dụng |
| Bằng chứng cơ chế | Curve / per-class / Grad-CAM / calibration / corruption |
| Kết luận | Giữ / bỏ / tương đương / chưa kết luận |
| Đánh đổi | Chi phí, latency, độ phức tạp |

**Ví dụ khung (chưa có số):**

| Yếu tố | Mức | Kết quả | Δ [CI] | Cơ chế | Kết luận |
|---|---|---|---|---|---|
| BN | off/on | ___ | ___ | Ổn định phân phối activation, cho phép lr cao hơn; nếu off, gradient không ổn định và cần lr thấp | ___ |
| Dropout | 0/0,3 | ___ | ___ | Giảm đồng thích nghi; quá cao → underfit | ___ |
| WD | 0/1e-4 | ___ | ___ | Co weight, giảm overfit; quá cao → underfit | ___ |
| Aug | off + 3 profile | ___ | ___ | Tăng bất biến với hình học/ánh sáng; có thể không tăng clean acc | ___ |
| GAP | off/on | ___ | ___ | Giảm params, mất chi tiết không gian | ___ |
| Depth | 3/4 | ___ | ___ | Receptive field tăng; quá sâu → overfit | ___ |
| Width | 3 mức | ___ | ___ | Dung lượng tăng; diminishing return | ___ |
| LR | 2 mức | ___ | ___ | Tốc độ/ổn định | ___ |
| Class weight | none/inv_sqrt | ___ | ___ | Tăng recall lớp hiếm, có thể giảm precision | ___ |
| Input | 64/96 | ___ | ___ | Chi tiết vs chi phí | ___ |

### 8.2. Learning curves

Vẽ 4 panel cho từng cấu hình chính:

1. Train loss vs epoch.
2. Val loss vs epoch.
3. Train accuracy vs epoch.
4. Val macro-F1 vs epoch.

Đánh dấu best epoch. So sánh:

- CNN-A: overfit sớm? train acc 1,0 nhưng val thấp.
- CNN-B: gap nhỏ hơn, val cao hơn.
- CNN-C: depth 4 có overfit không? GAP có làm chậm hội tụ?
- F00 vs F15: aug làm curve nhiễu hơn nhưng val cao hơn?

**Kết luận cơ chế:** "BN giảm gap từ ___ → ___"; "aug làm train acc thấp hơn nhưng val cao hơn, tức giảm overfit"; "4 khối đạt train acc ___ nhưng val ___".

### 8.3. Per-class Δ

- Bảng/heatmap: hàng = cấu hình, cột = 43 lớp, giá trị = ΔF1 so với reference.
- Kèm support từng lớp.
- Tập trung vào lớp hiếm (0, 19, 37, 27, 32, 41, 42) và lớp dễ nhầm (cặp 0/19? 21/23? 30/31?).
- Câu hỏi: BN/dropout/aug giúp lớp nào? Có lớp nào bị hy sinh để tăng macro-F1?
- Nếu aug chỉ giúp lớp bị mờ/sáng, đó là bằng chứng cơ chế.

### 8.4. Grad-CAM / occlusion

- Chọn 12 ảnh: 4 lớp × 3 trường hợp (đúng, sai, confidence thấp).
- So sánh CNN-B (3 khối) vs CNN-C (4 khối + GAP).
- Câu hỏi: model nhìn vào biển hay nền? 4 khối có focus tốt hơn không? GAP có làm mất định vị?
- Dùng Grad-CAM trên layer conv cuối; chuẩn hóa 0–1; hiển thị overlay.
- Không dùng ảnh có người/biển số.

### 8.5. Calibration

- Reliability diagram (15 bin) cho F00, F07, F15, D1, D3.
- ECE, Brier.
- Câu hỏi: dropout/WD/aug có làm model bớt tự tin quá mức? BN có làm calibration tốt hơn?
- Nếu model tự tin sai ở lớp hiếm, đó là lý do cần ngưỡng "không xác định" chọn trên val.

### 8.6. Corruption robustness

- 6 loại × 3 mức (mục 6.5).
- Vẽ 6 panel: accuracy vs severity, mỗi panel một loại; đường cho G0, G5.
- Bảng trung bình theo loại + Δ so với clean.
- Câu hỏi: aug có tăng robustness không? Phần nào (affine/color/perspective) đóng góp?
- Nếu aug không tăng clean nhưng tăng corruption, đây là bằng chứng để giữ aug cho demo.

### 8.7. Error taxonomy

- Dùng predictions v2; phân loại lỗi:
  1. Nhầm cặp giống nhau (hình dạng/màu).
  2. Ảnh mờ/tối/chói.
  3. Biển bị che một phần.
  4. Góc nhìn lạ/phối cảnh.
  5. Nhãn khó/ngoại lệ.
- Đếm tỷ lệ từng nhóm; 12 mẫu đại diện; nhận xét thủ công.
- So sánh F00 vs F15: aug có giảm nhóm 2/4 không?

### 8.8. Khung trả lời "vì sao CNN X"

Sau khi có kết quả, điền đoạn sau (giữ nguyên cấu trúc):

> Trên OOF 31.371 ảnh dev (3 fold × 3 seed), cấu hình **___**, đạt macro-F1/43 = **___** [CI **___**], accuracy = **___** [CI **___**]. Cấu hình này được chọn vì:
>
> 1. **So với CNN-A:** hơn **___** điểm [CI **___**]; cơ chế: **___** (BN ổn định tối ưu, giảm gap từ **___** → **___**).
> 2. **So với CNN-C / 4 khối / GAP:** hơn **___** điểm [CI **___**] hoặc tương đương; cơ chế: **___** (4 khối overfit: train **___** / val **___**; GAP mất chi tiết **___**).
> 3. **Về regularization:** BN **___**; dropout **___**; WD **___**; aug **___**. Các yếu tố không qua CI được báo là "không đủ bằng chứng", không giữ vì cảm tính.
> 4. **Về đánh đổi:** params **___**; latency **___** ms/ảnh; nếu tương đương, chọn cấu hình nhẹ hơn.
> 5. **Về aug:** clean Δ **___** [CI **___**]; corruption Δ **___** [CI **___**]; quyết định **___**.
>
> Theo luật quyết định đăng ký trước (Δ ≥ 0,3 điểm và CI > 0), kết luận: **___**. Nếu kết quả không đủ mạnh, báo cáo là **tương đương/chưa kết luận** và chọn phương án đơn giản hơn.

**Bảng nhánh kết quả (để không bị bất ngờ):**

| Nếu kết quả | Kết luận | Việc phải làm |
|---|---|---|
| CNN-A thắng | BN/dropout/WD không cần ở quy mô này | Chọn A; kiểm tra stability 3 seed; giải thích BN không cần vì lr thấp/dữ liệu đủ |
| CNN-C thắng | Depth/GAP có ích | Chọn C; kiểm tra params/latency; giải thích receptive field |
| GAP tương đương | GAP không mất mát | Chọn GAP; giảm params ~5× |
| 4 khối tương đương | Depth không cần | Chọn 3 khối; ít overfit hơn |
| Aug clean-neutral, corruption-positive | Aug cần cho robustness | Giữ aug; ghi rõ đánh đổi clean |
| Aug không tăng cả hai | Aug không cần | Bỏ aug; giảm chi phí |
| Class weight giảm macro-F1 | Không cần | Bỏ; báo trade-off |
| Input 96 thắng rõ | Chi tiết quan trọng | Chọn 96 nếu latency chấp nhận; nếu không, 64 |
| LR 1e-4 tốt hơn 1e-3 rõ | LR nhỏ ổn định hơn | Chọn 1e-4 nếu CI > 0; báo trade-off thời gian |
| Không gì hơn baseline | Bài toán dễ / model đủ | Báo tương đương; chọn đơn giản nhất |

### 8.9. Điều quan trọng nhất

**Kế hoạch này không ép chọn CNN-B.** Nếu dữ liệu nói CNN-A hoặc CNN-C tốt hơn, chọn theo dữ liệu và giải thích. Câu trả lời "vì sao CNN X" phải khớp với bảng số, không khớp với báo cáo cũ.

---

## 9. Đối chiếu 4 vấn đề và kịch bản phản biện

### 9.1. Bảng đối chiếu

| Vấn đề | Thí nghiệm / việc làm | Artifact | Chương báo cáo | Tiêu chí đạt |
|---|---|---|---|---|
| 1. Bài test đủ để chốt | Pre-registration; F 2⁴; D architecture; H hyperparam; G aug; X OOF + lockbox | `summary.csv`, `decision_cards.csv`, `oof_predictions.csv` | 3.3, 3.4, 4.2–4.5 | Mỗi kết luận có Δ + CI + luật quyết định |
| 2. Hết leak | Split 3 tầng track/sign-group; lockbox; mean/std theo split; audit official test | `manifest.json`, audit CSV | 2.3–2.5 | 0 track/sign-group overlap; lockbox one-shot; official test raw/clean |
| 3. Giải thích | Decision card; curve; per-class; Grad-CAM; calibration; corruption; error taxonomy | Hình 5.x, bảng 5.x | 5.1–5.6 | Mỗi kết luận có ít nhất 1 bằng chứng cơ chế |
| 4. Cấu hình cần không | Factorial; aug components; báo cáo "không đủ bằng chứng" | Bảng main effect, bảng aug | 4.3–4.5 | Mỗi yếu tố có kết luận giữ/bỏ/tương đương |

### 9.2. Kịch bản trả lời phản biện

| Câu hỏi có thể gặp | Trả lời ngắn | Bằng chứng |
|---|---|---|
| "Bài test của em có đủ để chốt mô hình không?" | "Có. Em đăng ký trước 8 so sánh chính, chạy factorial 2⁴ để tách BN/dropout/WD/aug, tách kiến trúc, xác nhận bằng 3-fold OOF, và test một lần trên lockbox track-disjoint. Mỗi kết luận có Δ kèm CI và luật quyết định." | Mục 4, 6, 7; bảng decision card |
| "Số liệu của em có bị leak không?" | "Không. Mọi run dùng stratified_group; lockbox 0 track/sign-group overlap. Official test không track-disjoint, nên em chỉ báo cáo như benchmark phụ, kèm bản clean bỏ 131 ảnh gần trùng; mức ảnh hưởng chỉ −0,007 điểm." | Mục 2.3–2.5; audit |
| "Vì sao em chọn CNN-B?" | "Em không chọn trước. Em so sánh A/B/C và các biến thể depth/width/GAP trên OOF; CNN-B thắng vì ___ điểm [CI], cơ chế ___. Nếu A/C thắng, em chọn theo dữ liệu." | Mục 6.3, 8.8 |
| "Augmentation có thực sự cần không?" | "Em bật/tắt 9 profile. Clean Δ = ___ [CI]; corruption Δ = ___ [CI]. Kết luận: ___." | Mục 6.5, 8.6 |
| "BN/dropout/WD có cần không?" | "Em chạy đủ 2⁴. Main effect: BN ___, dropout ___, WD ___ [CI]. Cái nào CI chứa 0 thì em báo tương đương, không giữ vì cảm tính." | Mục 6.2, 8.1 |
| "Vì sao 64×64?" | "Em so 48/64/96. 96 hơn ___ điểm nhưng chậm ___ lần; 48 kém ___ điểm. Em chọn 64 vì ___." | Mục 6.4, 8.1 |
| "Sao không dùng pretrained?" | "Đề yêu cầu CNN tự xây dựng cho mô hình chính. Em giữ baseline HOG+SVC và CNN tự xây; pretrained chỉ có thể là tham khảo ngoài phạm vi." | Mục 5.2 |
| "Kết quả có ý nghĩa thống kê không?" | "Em báo Δ kèm CI cluster bootstrap theo track, Holm cho 8 so sánh chính. Chỗ nào CI chứa 0 em nói tương đương/chưa kết luận." | Mục 7 |
| "Vì sao official test vẫn cao hơn lockbox?" | "Hai tập khác phân phối và official test không track-disjoint. Em không so trực tiếp; lockbox là số độc lập, official test là benchmark phụ." | Mục 3.7, 6.6 |
| "Nếu kết quả không đẹp thì sao?" | "Kết quả âm vẫn là kết quả. Em báo đúng, không chỉnh test, và chọn phương án đơn giản nhất trong nhóm tương đương." | Mục 4.5, 8.9 |

---

## 10. Kế hoạch triển khai

### 10.1. Phase 0 — Chốt giao thức và chia dữ liệu (0,5–1 ngày, không GPU)

| Việc | Đầu ra | Điều kiện xong |
|---|---|---|
| Chốt tài liệu này, commit `PRE_REGISTRATION` | Commit hash | Không sửa im lặng sau đó |
| Bổ sung script chia split v2 | `src/split_v2.py` (hoặc mở rộng `src/data.py`) | Chạy được, có test |
| Sinh lockbox/dev/screening/3-fold | `outputs/splits/v2/*.json` | Đủ 43 lớp, 0 overlap |
| Audit near-duplicate lockbox/train | `outputs/results/v2/audit/*.csv` | 0 cặp MAD < 5 |
| Kiểm tra mean/std theo split | `manifest.json` | Có mean/std từng split |
| Test split | `tests/test_split_v2.py` | Pass |
| Ghi hash split/config | `manifest.json` | Có SHA256 |

### 10.2. Phase 1 — Baseline + factorial (~4–5 h GPU)

| Việc | Số run | Đầu ra |
|---|---:|---|
| B1 | 1 | Bảng baseline |
| F00–F15 × 3 seed | 48 | Bảng 2⁴ + main effect |
| Phân tích sơ bộ | – | Quyết định có cần chỉnh gì không (không đổi luật) |

**Cửa kiểm tra:** nếu F15 (E3) trên split mới thấp hơn nhiều so với 97,06% cũ, kiểm tra lại split/mean/std trước khi chạy tiếp; không vội kết luận.

### 10.3. Phase 2 — Kiến trúc, hyperparam, augmentation (~3 h GPU)

| Việc | Số run | Đầu ra |
|---|---:|---|
| D1–D3, W1–W2 × 3 seed (D4 = F15) | 15 | Bảng kiến trúc/Pareto |
| L1, CW1, I2 × 3 seed | 9 | Bảng hyperparam |
| G1–G3 × 3 seed (G0 = F07, G5 = F15) | 9 | Bảng aug + corruption |
| Phân tích | – | Chọn top 2 + control |

### 10.4. Phase 3 — Xác nhận, freeze, lockbox (~2–3 h GPU)

| Việc | Số run | Đầu ra |
|---|---:|---|
| Top 2 + control × 3 fold × 3 seed | 27 | OOF predictions, CI |
| Chọn winner theo luật 4.5 | – | Quyết định + lý do |
| Freeze config/commit | – | `FREEZE.md` + hash |
| Train final trên dev × 3 seed | 3 | 3 checkpoint |
| Mở lockbox 1 lần | – | `lockbox_result.json` |
| Official test raw/clean | – | `official_test_v2.json` |

### 10.5. Phase 4 — Phân tích và báo cáo (2–3 ngày)

| Việc | Đầu ra |
|---|---|
| Bootstrap CI, Holm, variance decomposition | `bootstrap_ci.csv`, `variance.json` |
| Hình cơ chế | `outputs/figures/v2/*.png` |
| Decision cards | `decision_cards.csv` + bảng báo cáo |
| Viết lại chương 2–6 | `BAO_CAO_v2.md` |
| Kiểm tra số khớp artifact | Checklist cuối |

### 10.6. Đường tối thiểu nếu gấp

Bản gọn 112 run là mặc định. Nếu rất gấp:

1. Phase 0 đầy đủ (không cắt).
2. Screening 1 seed: F16 + D5 + H3 + G3 = 27 run.
3. Chọn top 2 + control; 3-fold × 3 seed (27 run).
4. Final 3 seed; lockbox.
5. Báo cáo rõ "screening 1 seed, xác nhận 3 seed"; không tuyên bố thắng/thua ở screening.

### 10.7. Việc code cần bổ sung (kế hoạch, chưa làm)

| File | Việc | Test |
|---|---|---|
| `src/split_v2.py` | Lockbox/dev/screening/3-fold; sign-group union-find; manifest/hash | `tests/test_split_v2.py` |
| `src/data.py` | Đọc split manifest; mean/std theo split; cache 64/96 | Smoke test |
| `src/train.py` | `fold`, `class_weight`, `aug_profile`; log hash | `tests/test_train_smoke.py` |
| `src/evaluate.py` | Per-class, top-3, ECE, corruption 6 loại, lưu prediction | `tests/test_eval.py` |
| `src/stats.py` | Cluster bootstrap, paired bootstrap, Holm, variance | `tests/test_stats.py` |
| `src/explain.py` | Grad-CAM, reliability, per-class Δ | Smoke test |
| `scripts/run_matrix.py` | Chạy theo CSV, resume, skip run đã có | Dry-run |
| `src/summarize.py` | `summary.csv` v2 + cờ `valid_for_conclusion` | Test schema |
| `configs/v2/*.yaml` | Sinh từ CSV | Validate YAML |

**Nguyên tắc:** không sửa logic cũ nếu không cần; thêm chế độ v2, giữ tương thích.

---

## 11. Artifact và quy ước

### 11.1. Cấu trúc

```
docs/kehoach/1/
├── README.md
├── KE_HOACH_V2_DAY_DU.md
├── MA_TRAN_RUN.csv
├── CHECKLIST_TRIEN_KHAI.md
├── MAU_KET_QUA_GIA_DINH.md
└── PROMPT_TRIEN_KHAI_V2.md

configs/v2/
├── b1.yaml
├── f00.yaml ... f15.yaml
├── d1.yaml ... d4.yaml, w1.yaml, w2.yaml
├── l1.yaml, l3.yaml, cw1.yaml, i2.yaml
├── g0.yaml ... g3.yaml, g5.yaml
└── final.yaml

outputs/splits/v2/
├── lockbox.json
├── dev.json
├── screening.json
├── fold_0.json, fold_1.json, fold_2.json
└── manifest.json

outputs/runs/v2_<config>_s<seed>[_f<fold>]/
├── config.yaml
├── env.json
├── metrics.csv
├── best.pt
├── curves.png
└── predictions_val.csv

outputs/results/v2/
├── summary.csv
├── oof_predictions.csv
├── decision_cards.csv
├── bootstrap_ci.csv
├── variance.json
├── corruption.csv
├── lockbox_result.json
├── official_test_v2.json
└── audit/
    ├── near_duplicate_test.csv
    ├── near_duplicate_lockbox.csv
    └── sign_groups.csv

outputs/results/legacy/
├── README.md
└── summary_random_split.csv
```

### 11.2. Quy ước tên

- `run_id = v2_<config_id>_s<seed>`; nếu 3-fold: `v2_<config_id>_s<seed>_f<fold>`.
- `config_id` viết thường: `f00`, `d1`, `l1`, `g5`, `x1`, `final`.
- Không ghi đè run cũ; nếu chạy lại, thêm `_r2`.
- `summary.csv` có cột `protocol` = `v2_clean` và `valid_for_conclusion` = 1/0.

### 11.3. Schema `summary.csv`

```
protocol,block,config_id,run_id,seed,fold,split,model,params,aug,aug_profile,
epochs_run,best_epoch,val_acc,val_macro_f1,val_macro_f1_ci_low,val_macro_f1_ci_high,
test_acc,test_macro_f1,train_time_s,infer_ms_per_img,git_commit,config_sha256,
split_sha256,valid_for_conclusion,notes
```

- `valid_for_conclusion=0` cho run lỗi, run random cũ, run screening 1 seed (nếu dùng).
- Không để trống `config_sha256`/`split_sha256` ở run kết luận.

### 11.4. Schema `oof_predictions.csv`

```
config_id,seed,image_index,track_key,true_label,pred_label,prob_true,prob_pred,fold
```

- Mỗi cặp `(config_id, seed, image_index)` đúng một dòng.
- Dùng để tính OOF, per-class, McNemar, calibration.

### 11.5. Schema `decision_cards.csv`

```
factor,levels,reference,metric,mean,ci_low,ci_high,delta,delta_ci_low,delta_ci_high,
p_holm,mechanism,evidence,decision,tradeoff
```

### 11.6. Schema `corruption.csv`

```
config_id,corruption,severity,accuracy,macro_f1,n
```

### 11.7. Legacy

- Chuyển bảng E1–E7 cũ vào `outputs/results/legacy/` và `docs/bao_cao/BAO_CAO_v2.md` Phụ lục.
- `legacy/README.md` ghi rõ: "Các run này dùng `split_random`; val bị leak; không dùng để kết luận."
- Không xóa số cũ; chỉ đánh dấu.

---

## 12. Thay đổi báo cáo

### 12.1. Theo chương

| Mục báo cáo cũ | Hành động | Nội dung mới |
|---|---|---|
| 2.3 Ba cách chia tập | **Thay** | Split 3 tầng v2; lockbox/dev/screening/3-fold; số track/ảnh; hash |
| 2.4 Rò rỉ frame | **Giữ + mở rộng** | Số random cũ; thêm audit official test 131 ảnh/24 track; kết luận không track-disjoint |
| 2.5 MD5 + pHash | **Sửa** | Bổ sung sign-group; sửa câu "17.214 cặp chỉ là hash collision" thành "có 131 ảnh gần trùng xác nhận; phần còn lại là ứng viên" |
| 3.3 Thiết lập huấn luyện | **Cập nhật** | Config v2; 40 epoch; mean/std theo split; split manifest |
| 3.4 Chỉ số và kỷ luật | **Cập nhật** | Macro-F1/43; CI cluster bootstrap; Holm; luật quyết định |
| 4.2 Baseline + ablation (val random) | **Bỏ khỏi kết luận** | Chuyển legacy; thay bằng bảng 2⁴ + kiến trúc + aug |
| 4.3 Chọn mô hình trên val sạch | **Thay** | OOF 3-fold × 3 seed; decision rule |
| 4.4 Multi-seed và test | **Thay** | Lockbox one-shot; official test raw/clean phụ |
| 4.5 (mới) | **Thêm** | Factorial, architecture, hyperparam, augmentation |
| 5. Phân tích lỗi | **Mở rộng** | Thêm per-class Δ, Grad-CAM, calibration, corruption |
| 6.3 Hạn chế | **Sửa** | Official test không độc lập; lockbox nhỏ hơn; không tuyên bố thực tế |
| Phụ lục | **Thêm** | Legacy leaky; bảng 2⁴ đầy đủ; decision cards |

### 12.2. Bảng/hình mới

**Bảng:**

1. Bảng split v2 (track/ảnh/vai trò).
2. Bảng audit near-duplicate official test.
3. Bảng 2⁴ đầy đủ 16 dòng.
4. Bảng main effect 4 dòng.
5. Bảng kiến trúc/Pareto.
6. Bảng hyperparam.
7. Bảng aug + corruption trung bình.
8. Bảng OOF + lockbox + official test.
9. Bảng decision cards.
10. Bảng variance decomposition.

**Hình:**

1. Sơ đồ split 3 tầng.
2. Interaction plot (BN×aug, DO×WD).
3. Pareto params vs macro-F1.
4. Learning curves A/B/C.
5. Per-class Δ heatmap.
6. Grad-CAM grid.
7. Reliability diagram.
8. Corruption curves.
9. Confusion matrix chuẩn hóa hàng.
10. Phân bố lỗi theo nhóm.

### 12.3. Câu chữ hạn chế (dùng trong báo cáo)

> "Official test của GTSRB không track-disjoint: 131 ảnh (1,04%) gần trùng ảnh train, ứng với 24 track. Bỏ 131 ảnh này chỉ làm accuracy thay đổi −0,007 điểm, nên không thể nói test bị thổi phồng nặng; tuy nhiên test không độc lập hoàn toàn. Vì vậy em báo cáo official test như benchmark phụ (raw 12.630 và clean 12.499) và dùng lockbox nội bộ 7.830 ảnh làm ước lượng độc lập chính."

> "Lockbox 7.830 ảnh nhỏ hơn official test 12.630, nên CI rộng hơn khoảng 1,27 lần. Em không so trực tiếp hai con số; mọi so sánh mô hình dùng cùng một tập và cùng một giao thức."

> "Các run E1–E7 cũ dùng `split_random`; val bị rò rỉ frame. Em giữ chúng trong phụ lục với nhãn `legacy_leaky`, không dùng để kết luận."

### 12.4. Câu phải xóa/sửa

- "Val random 99,9% chứng minh mô hình gần hoàn hảo" → **sai**, do leak.
- "17.214 cặp pHash chỉ là hash collision, không có gần trùng" → **quá mạnh**; có 131 ảnh gần trùng xác nhận.
- "E4 kém vì GAP" → **chưa chứng minh**; E4 đổi 4 thứ. Phải dùng D1/D2; D3 chỉ là legacy.
- "Aug luôn tốt" → **chưa chắc**; phải bật/tắt và đo corruption.
- "Test độc lập" → **không chính xác**; nói "test công khai, không track-disjoint".

---

## 13. Rủi ro và đối phó

| Rủi ro | Xác suất | Ảnh hưởng | Đối phó |
|---|---|---|---|
| Split seed làm lockbox lệch phân phối | Trung bình | CI rộng, kết quả khó so | Stratified theo lớp; báo per-class support; có thể lặp 3 lockbox seed nếu còn thời gian |
| Sign-group gộp quá nhiều, 3-fold thiếu nhóm | Trung bình | Không đủ 43 lớp/fold | Kiểm tra trước; nếu <6 nhóm/lớp, dùng track-level cho lớp đó và ghi rõ |
| Run NaN/phân kỳ | Thấp | Run lỗi | Báo cáo là kết quả; không xóa |
| 4 khối/GAP không hội tụ | Trung bình | Kết luận sai | Chạy đủ 40 epoch; kiểm tra curve; không so 10 epoch |
| Thời gian vượt | Trung bình | Trễ báo cáo | Dùng bản tối thiểu 58 run; ưu tiên F + OOF + lockbox |
| Ổ đĩa cache 96 | Thấp | Đầy đĩa | Cache riêng, xóa sau; 96 chỉ chạy I2 |
| Config drift | Trung bình | Không tái lập | Config hash; validate YAML; không sửa tay |
| Vô tình dùng test/lockbox | Thấp | Mất độc lập | Quyền read-only; log; checklist |
| Giảng viên hỏi "sao không pretrained" | Cao | Mất điểm | Đề yêu cầu tự xây; baseline HOG; pretrained ngoài phạm vi |
| Kết quả không có ý nghĩa | Trung bình | Khó bảo vệ | Báo tương đương/chưa kết luận; chọn đơn giản; không伪造 |
| Bug metric | Thấp | Sai số | Unit test với sklearn; so tay 1 batch |
| GPU 12 GB không đủ batch 128 | Thấp | Run lỗi | Giảm batch hoặc AMP; ghi rõ |
| Windows num_workers | Trung bình | Treo | `if __name__ == "__main__":`; test 2 worker |

---

## 14. Checklist nghiệm thu

### 14.1. Dữ liệu và split

- [ ] Pool 39.201; 8 ảnh MD5 đã loại; `usable_for_train` đúng.
- [ ] Lockbox 43/43 lớp; 0 track overlap; 0 sign-group overlap.
- [ ] Dev 43/43 lớp; 0 track overlap với lockbox.
- [ ] Screening train/val 0 track overlap; 43/43 val.
- [ ] 3 fold: 0 overlap; 43/43 val mỗi fold.
- [ ] `manifest.json` có hash, mean/std từng split.
- [ ] Audit lockbox/train: 0 cặp MAD < 5.
- [ ] Test không nằm trong train/lockbox.
- [ ] `tests/test_split_v2.py` pass.

### 14.2. Huấn luyện

- [ ] Mỗi run có config, env, metrics, best.pt, curves.
- [ ] `metrics.csv` đúng schema, flush từng epoch.
- [ ] 40 epoch tối đa; patience 7; không early stop trên test.
- [ ] Seed 42/123/2024; cùng split cho so sánh.
- [ ] Run lỗi được đánh dấu, không xóa.
- [ ] `config_sha256`/`split_sha256` khớp.

### 14.3. Thống kê

- [ ] Macro-F1/43 đúng `labels=range(43)`.
- [ ] CI cluster bootstrap theo track; B ≥ 2.000.
- [ ] Paired bootstrap cho Δ; Holm cho 8 so sánh chính.
- [ ] Variance decomposition (seed/fold/residual).
- [ ] Bảng kết quả khớp `summary.csv`/`oof_predictions.csv`.

### 14.4. Giải thích

- [ ] Decision card cho 10 yếu tố.
- [ ] Learning curves A/B/C.
- [ ] Per-class Δ heatmap.
- [ ] Grad-CAM 12 ảnh.
- [ ] Reliability diagram + ECE.
- [ ] Corruption 8×3.
- [ ] Error taxonomy 12 mẫu.
- [ ] Khung "vì sao CNN X" đã điền số.

### 14.5. Báo cáo và repo

- [ ] Bảng cũ `legacy_leaky` không dùng trong kết luận.
- [ ] Official test raw/clean; ghi rõ không độc lập.
- [ ] Lockbox one-shot; log.
- [ ] Hạn chế trung thực; không tuyên bố thực tế.
- [ ] `git status` không có file rác; không commit data/checkpoint.
- [ ] README/hướng dẫn chạy cập nhật.

### 14.6. Ký xác nhận

| Hạng mục | Người kiểm | Ngày | Đạt |
|---|---|---|---|
| Split | | | |
| Training | | | |
| Stats | | | |
| Report | | | |

---

## 15. Phụ lục

### Phụ lục A — Mẫu bảng factorial 2⁴

| ID | BN | DO | WD | Aug | Val acc | Macro-F1 | Δ vs F00 | CI Δ | Gap | Best epoch | Params | Time |
|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|
| F00 | off | 0 | 0 | off | | | – | – | | | | |
| F01 | on | 0 | 0 | off | | | | | | | | |
| F02 | off | 0,3 | 0 | off | | | | | | | | |
| F03 | on | 0,3 | 0 | off | | | | | | | | |
| F04 | off | 0 | 1e-4 | off | | | | | | | | |
| F05 | on | 0 | 1e-4 | off | | | | | | | | |
| F06 | off | 0,3 | 1e-4 | off | | | | | | | | |
| F07 | on | 0,3 | 1e-4 | off | | | | | | | | |
| F08 | off | 0 | 0 | on | | | | | | | | |
| F09 | on | 0 | 0 | on | | | | | | | | |
| F10 | off | 0,3 | 0 | on | | | | | | | | |
| F11 | on | 0,3 | 0 | on | | | | | | | | |
| F12 | off | 0 | 1e-4 | on | | | | | | | | |
| F13 | on | 0 | 1e-4 | on | | | | | | | | |
| F14 | off | 0,3 | 1e-4 | on | | | | | | | | |
| F15 | on | 0,3 | 1e-4 | on | | | | | | | | |

### Phụ lục B — Mẫu bảng kiến trúc

| ID | Khối | Widths | GAP | Dropout | Params | Macro-F1 | CI | Gap | Best epoch | Latency |
|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| D4 | 3 | 32/64/128 | off | 0,3 | | | | | | |
| D1 | 3 | 32/64/128 | on | 0,3 | | | | | | |
| D2 | 4 | 32/64/128/256 | off | 0,3 | | | | | | |
| D3 | 4 | 32/64/128/256 | on | 0,4 | | | | | | |
| W1 | 3 | 16/32/64 | off | 0,3 | | | | | | |
| W2 | 3 | 48/96/192 | off | 0,3 | | | | | | |

### Phụ lục C — Mẫu bảng augmentation + corruption

**C.1. Clean:**

| ID | Profile | Macro-F1 | CI | Δ vs G0 | Δ CI |
|---|---|---|---:|---|---|
| G0 | none | | | – | – |
| G1 | affine | | | | |
| G2 | color | | | | |
| G3 | perspective | | | | |
| G5 | all | | | | |

**C.2. Corruption trung bình:**

| ID | Blur | Noise | Bright | Contrast | JPEG | Occl | TB |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | | | | | | | |
| G5 | | | | | | | |

### Phụ lục D — Lệnh dự kiến (PowerShell)

> Đây là **interface đề xuất**; các script `split_v2.py`, `run_matrix.py` chưa có trong repo. Không chạy khi chưa triển khai.

```powershell
# Kích hoạt môi trường
.\.venv\Scripts\Activate.ps1

# Kiểm tra môi trường
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"

# Phase 0: chia split v2 (script sẽ bổ sung)
python -m src.split_v2 --pool data/cache --out outputs/splits/v2 --seed-lockbox 2024 --seed-screen 42

# Phase 1: chạy một config
python -m src.train --config configs/v2/f00.yaml --seed 42

# Chạy toàn bộ matrix (script sẽ bổ sung)
python scripts/run_matrix.py --csv docs/kehoach/1/MA_TRAN_RUN.csv --only F

# Đánh giá
python -m src.evaluate --run outputs/runs/v2_f00_s42 --split screening_val

# Tổng hợp
python -m src.summarize --protocol v2_clean
```

### Phụ lục E — Mẫu config YAML

```yaml
experiment:
  id: f15
  seed: 42
  protocol: v2_clean
  notes: "CNN-B + BN + dropout 0.3 + WD 1e-4 + aug all"

data:
  split: v2_screening
  split_manifest: outputs/splits/v2/manifest.json
  image_size: 64
  num_classes: 43

model:
  name: cnn_b
  widths: [32, 64, 128]
  use_bn: true
  dropout: 0.3
  gap: false
  fc_hidden: 256

augmentation:
  enabled: true
  profile: all

train:
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.0001
  batch_size: 128
  epochs: 40
  scheduler: cosine
  warmup_epochs: 0
  patience: 7
  label_smoothing: 0.0
  class_weight: none
  amp: true
  num_workers: 4
  deterministic: true

eval:
  metrics: [accuracy, macro_f1_43, per_class, top3, ece]
  bootstrap: 2000
```

### Phụ lục F — Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Track | Một đoạn video ngắn (thường 30 frame) của cùng một biển; khóa `<label>_<track>` |
| Sign-group | Nhóm các track được gộp do cùng một biển vật lý (near-duplicate) |
| Lockbox | Tập test nội bộ 20% track, chỉ mở một lần |
| Dev | 80% track còn lại, dùng train/val |
| Screening | Chia 80/20 trong dev để so sánh nhanh |
| OOF | Out-of-fold: mỗi ảnh dev được dự đoán đúng một lần |
| Macro-F1/43 | Macro-F1 tính trên đủ 43 lớp, kể cả lớp không có mẫu |
| Cluster bootstrap | Bootstrap theo track, không theo ảnh |
| CI | Khoảng tin cậy 95% |
| Holm | Hiệu chỉnh đa so sánh Holm–Bonferroni |
| ECE | Expected Calibration Error |
| Corruption | Ảnh bị làm hỏng có kiểm soát (mờ, nhiễu, sáng...) |
| Decision card | Thẻ ghi kết quả + cơ chế + quyết định cho một yếu tố |
| Legacy leaky | Kết quả cũ dùng random split, không dùng kết luận |
| Pre-registration | Đăng ký trước câu hỏi, endpoint, luật quyết định |

### Phụ lục G — Tài liệu tham khảo

1. Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.* Neural Networks.
2. scikit-learn: `f1_score`, `StratifiedGroupKFold`, `classification_report`.
3. Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap.*
4. Holm, S. (1979). *A simple sequentially rejective multiple test procedure.*
5. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. (2017). *On Calibration of Modern Neural Networks.*
6. Hendrycks, D., Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.*
7. Selvaraju, R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
8. Bouthillier, X. et al. (2021). *Accounting for Variance in Machine Learning Benchmarks.*

### Phụ lục H — Lịch sử thay đổi

| Phiên bản | Ngày | Nội dung |
|---|---|---|
| 2.0 | 02/11/2026 | Bản kế hoạch đầy đủ đầu tiên; 4 vấn đề; split 3 tầng; factorial 2⁴; OOF + lockbox; decision card |
| 2.1 | ___ | Sau khi rà soát với giảng viên/người thực hiện; ghi rõ thay đổi |

---

**Hết tài liệu.**

