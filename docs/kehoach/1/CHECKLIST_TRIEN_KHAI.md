# CHECKLIST TRIỂN KHAI — KẾ HOẠCH V2

> Dùng kèm `KE_HOACH_V2_DAY_DU.md` và `MA_TRAN_RUN.csv`.
> Mỗi ô chỉ được tick khi có **bằng chứng** (file, log, hash, hình). Không tick theo cảm tính.

| Thông tin | Giá trị |
|---|---|
| Người thực hiện | Kỹ sư ML chính kiêm auditor tái lập (AI agent) |
| Ngày bắt đầu | 02/11/2026 |
| Commit pre-registration | điền ở mục 0.1 sau khi commit |
| Trạng thái | Đang làm |

---

## Phase 0 — Chốt giao thức và chia dữ liệu (không GPU)

### 0.1. Chốt tài liệu

- [x] Đọc hết `KE_HOACH_V2_DAY_DU.md`.
- [x] Chốt 8 so sánh chính (mục 4.4) — ghi trong `PRE_REGISTRATION_V2.md` mục 3.
- [x] Chốt luật quyết định (mục 4.5) — ghi trong `PRE_REGISTRATION_V2.md` mục 4.
- [x] Chốt luật dừng (mục 4.6) — ghi trong `PRE_REGISTRATION_V2.md` mục 5.
- [x] Commit tài liệu + `MA_TRAN_RUN.csv`; ghi commit hash: `__________` (cập nhật sau commit).
- [x] Không sửa luật sau khi xem kết quả; nếu buộc sửa, tạo file `_v2` và ghi lý do.

**Điều kiện xong:** có commit hash và ngày chốt.

### 0.2. Chia split

- [ ] Bổ sung `src/split_v2.py` (hoặc mở rộng `src/data.py`) theo mục 3.4.
- [ ] Sinh lockbox 20% track, seed 2024.
- [ ] Sinh dev 80%.
- [ ] Sinh screening 80/20 trong dev, seed 42.
- [ ] Sinh 3 fold trong dev, seed 42.
- [ ] Ghi `outputs/splits/v2/manifest.json`.
- [ ] Ghi hash từng file split.
- [ ] Ghi mean/std theo từng split.

### 0.3. Kiểm tra split

| # | Kiểm tra | Tiêu chí | Kết quả | Đạt |
|---|---|---|---|---|
| C1 | Track overlap lockbox/dev | 0 | | [ ] |
| C2 | Sign-group overlap | 0 (nếu bật) | | [ ] |
| C3 | Lockbox đủ lớp | 43/43 | | [ ] |
| C4 | Dev đủ lớp | 43/43 | | [ ] |
| C5 | Screening val đủ lớp | 43/43 | | [ ] |
| C6 | Mỗi fold đủ lớp | 43/43 | | [ ] |
| C7 | Overlap screening train/val | 0 | | [ ] |
| C8 | Overlap từng fold | 0 | | [ ] |
| C9 | Test không nằm trong train/lockbox | 0 | | [ ] |
| C10 | Lockbox không có ảnh gần trùng train (MAD < 5) | 0 | | [ ] |
| C11 | MD5 trùng trong train | 0 | | [ ] |
| C12 | Số track/lớp nhỏ nhất trong dev | ≥ 2 | | [ ] |

**Điều kiện xong:** tất cả C1–C12 đạt; `manifest.json` có hash.

### 0.4. Test code

- [ ] `tests/test_split_v2.py` pass.
- [ ] Test forward `(2,3,64,64) → (2,43)` cho A/B/C.
- [ ] Test macro-F1/43 khớp sklearn.
- [ ] Test đọc manifest và mean/std.
- [ ] Test `num_workers=2` trên Windows không treo.

---

## Phase 1 — Baseline + factorial 2⁴

### 1.1. Baseline

- [ ] B1 HOG + LinearSVC; C chọn trên screening val.
- [ ] Ghi lại C tốt nhất và thời gian trích/fit.
- [ ] Không chạy B0 majority / B1b refit (đã bỏ để giảm run).

### 1.2. Factorial

- [ ] Chạy F00–F15 × 3 seed (48 run).
- [ ] Mỗi run có config, env, metrics, best.pt, curves.
- [ ] Kiểm tra F00/F07/F15 tương ứng CNN-A/E2/E3.
- [ ] Tính main effect BN/DO/WD/Aug.
- [ ] Tính interaction BN×Aug, DO×WD.
- [ ] Bảng 2⁴ đầy đủ 16 dòng.

**Cửa kiểm tra:** nếu F15 thấp hơn 97,06% quá xa, kiểm tra split/mean/std trước khi chạy tiếp. Không đổi luật.

**Điều kiện xong:** bảng 2⁴ + bảng main effect + 4 hình curve đại diện.

---

## Phase 2 — Kiến trúc, hyperparam, augmentation

### 2.1. Kiến trúc

- [ ] D1, D2, D3 × 3 seed.
- [ ] W1, W2 × 3 seed.
- [ ] Bảng params vs macro-F1 (Pareto).
- [ ] So sánh D1 vs D4 (GAP), D2 vs D4 (depth), W1/W2 vs D4 (width).
- [ ] D3 chỉ map E4 cũ; không dùng D3 để kết luận cơ chế.
- [ ] Kết luận GAP/depth/width.

### 2.2. Hyperparam

- [ ] L1 (LR 1e-4, cách 10 lần so với 1e-3) × 3 seed; L3 = F15.
- [ ] CW1 (inv_sqrt) × 3 seed.
- [ ] I2 (input 96) × 3 seed.
- [ ] Không so 10 epoch với 40 epoch.
- [ ] Không chạy L2/L4/L5/P1/P2/W4/CW2/LS1/I1/BS1/BS2 (đã bỏ để giảm run).

### 2.3. Augmentation

- [ ] G1, G2, G3 × 3 seed; G0 = F07, G5 = F15.
- [ ] Corruption 6 loại × 3 mức trên screening val.
- [ ] Bảng clean + corruption.
- [ ] Kết luận aug có cần, phần nào.
- [ ] Không chạy G4/G6/G7/G8 (đã bỏ).

### 2.4. Chọn top 2 + control

- [ ] Sắp xếp theo mean macro-F1.
- [ ] Lấy nhóm trong 0,3 điểm so với tốt nhất.
- [ ] Chọn top 2 theo tie-break + 1 control (F07; nếu F07 đã trong top 2 thì F00).
- [ ] Ghi lý do chọn **trước** khi chạy OOF.
- [ ] Điền X1–X3 vào `MA_TRAN_RUN.csv`.

**Điều kiện xong:** có danh sách top 2 + control và lý do, có bảng D/H/G.

---

## Phase 3 — Xác nhận, freeze, lockbox

### 3.1. OOF

- [ ] Chạy top 2 + control × 3 fold × 3 seed (27 run).
- [ ] Mỗi ảnh dev được dự đoán đúng 1 lần mỗi seed.
- [ ] `oof_predictions.csv` đủ dòng, không trùng.
- [ ] Tính macro-F1/43 trên OOF.
- [ ] Bootstrap CI theo track (B ≥ 2.000).
- [ ] Paired bootstrap cho Δ.
- [ ] Holm cho 8 so sánh chính.
- [ ] Variance decomposition seed/fold/residual.

### 3.2. Chọn winner

- [ ] Áp dụng luật 4.5.
- [ ] Nếu tương đương, chọn ít params/nhanh/đơn giản.
- [ ] Ghi decision card cho winner.
- [ ] Ghi rõ config, seed, epoch = median best epoch.
- [ ] Commit freeze; ghi hash: `__________`.

### 3.3. Final + lockbox

- [ ] Train final trên toàn dev × 3 seed.
- [ ] Không early stop trên lockbox.
- [ ] Kiểm tra checkpoint load được, forward đúng.
- [ ] Mở lockbox **một lần**; ghi log thời gian/lệnh.
- [ ] `lockbox_result.json` có accuracy, macro-F1/43, per-class, confusion, top-3, ECE.
- [ ] Official test raw 12.630.
- [ ] Official test clean 12.499.
- [ ] Không chỉnh gì sau khi xem lockbox.

**Điều kiện xong:** có OOF + lockbox + official test; có freeze hash.

---

## Phase 4 — Phân tích, hình, báo cáo

### 4.1. Thống kê

- [ ] Bảng summary v2 khớp artifact.
- [ ] Bảng CI cho 8 so sánh chính.
- [ ] Bảng variance decomposition.
- [ ] Kiểm tra macro-F1/43 với sklearn trên một mẫu.
- [ ] Không có số nào không trace được về file.

### 4.2. Hình cơ chế

- [ ] Learning curves A/B/C.
- [ ] Interaction plot (BN×Aug, DO×WD).
- [ ] Pareto params vs macro-F1.
- [ ] Per-class Δ heatmap.
- [ ] Grad-CAM 12 ảnh.
- [ ] Reliability diagram + ECE.
- [ ] Corruption curves 6 panel.
- [ ] Confusion matrix chuẩn hóa hàng.
- [ ] Error taxonomy 12 mẫu.

### 4.3. Decision cards

- [ ] BN.
- [ ] Dropout.
- [ ] WD.
- [ ] Aug.
- [ ] Depth.
- [ ] GAP.
- [ ] Width.
- [ ] LR.
- [ ] Class weight.
- [ ] Input size.

Mỗi card có: mức đã thử, mean ± sd, Δ [CI], cơ chế, kết luận, đánh đổi.

### 4.4. Báo cáo

- [ ] Chương 2: split v2 + audit official test.
- [ ] Chương 3: config v2 + metric/CI/luật.
- [ ] Chương 4: thay bảng ablation cũ bằng 2⁴ + kiến trúc + aug + OOF/lockbox.
- [ ] Chương 5: thêm cơ chế.
- [ ] Chương 6: hạn chế trung thực.
- [ ] Phụ lục: legacy leaky.
- [ ] Xóa/sửa các câu sai (mục 12.4).
- [ ] Kiểm tra số trong báo cáo khớp `summary.csv`/`lockbox_result.json`.

---

## Nghiệm thu cuối

| # | Điều kiện | Đạt |
|---|---|---|
| 1 | Lockbox 0 track/sign-group overlap; one-shot | [ ] |
| 2 | Mọi run kết luận dùng `stratified_group` | [ ] |
| 3 | Mỗi kết luận có Δ + CI + luật quyết định | [ ] |
| 4 | Factorial 2⁴ đủ 16 config | [ ] |
| 5 | Kiến trúc tách GAP/depth/width; D3 chỉ legacy | [ ] |
| 6 | Aug bật/tắt + corruption | [ ] |
| 7 | OOF 3-fold × 3 seed | [ ] |
| 8 | Decision card đủ 10 yếu tố | [ ] |
| 9 | Báo cáo không dùng số random cũ để kết luận | [ ] |
| 10 | Không tune trên lockbox/test | [ ] |
| 11 | Repo sạch; không commit data/checkpoint | [ ] |
| 12 | Tái lập: cùng seed → cùng kết quả (sai khác ghi rõ) | [ ] |

### Ký xác nhận

| Hạng mục | Người kiểm | Ngày | Ghi chú |
|---|---|---|---|
| Split | | | |
| Training | | | |
| Stats | | | |
| Report | | | |
| Final | | | |

---

## Điều kiện dừng khẩn cấp

Dừng và báo ngay nếu:

- Phát hiện track/sign-group overlap > 0.
- Lockbox bị dùng trước freeze.
- Test xuất hiện trong train.
- Config hash/split hash không khớp.
- Metric không tái lập được với cùng seed.
- Có số trong báo cáo không trace được về artifact.

Không tự sửa số; ghi log, tạo bản `_v2`, báo người phụ trách.
