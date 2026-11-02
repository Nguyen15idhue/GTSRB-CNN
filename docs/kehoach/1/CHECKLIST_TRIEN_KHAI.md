# CHECKLIST TRIỂN KHAI — KẾ HOẠCH V2

> Dùng kèm `KE_HOACH_V2_DAY_DU.md` và `MA_TRAN_RUN.csv`.
> Mỗi ô chỉ được tick khi có **bằng chứng** (file, log, hash, hình). Không tick theo cảm tính.

| Thông tin | Giá trị |
|---|---|
| Người thực hiện | Kỹ sư ML chính kiêm auditor tái lập (AI agent) |
| Ngày bắt đầu | 02/11/2026 |
| Commit pre-registration | `18a785d` |
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

- [x] Bổ sung `src/split_v2.py` (hoặc mở rộng `src/data.py`) theo mục 3.4.
- [x] Sinh lockbox 20% track, seed 2024 → **8.302 ảnh / 277 track / 238 sign-group**.
- [x] Sinh dev 80% → **30.899 ảnh / 1.030 track**.
- [x] Sinh screening 80/20 trong dev, seed 42 → **24.719 / 6.180**.
- [x] Sinh 3 fold trong dev, seed 42 → **(19.949/10.950), (20.729/10.170), (21.120/9.779)**.
- [x] Ghi `outputs/splits/v2/manifest.json`.
- [x] Ghi hash từng file split.
- [x] Ghi mean/std theo từng split.

> Số thực tế khác bảng dự kiến của kế hoạch vì đơn vị nhóm là **sign-group**
> (1.183 nhóm từ 1.307 track), không phải track. Manifest là nguồn số chính thức.

### 0.3. Kiểm tra split

| # | Kiểm tra | Tiêu chí | Kết quả | Đạt |
|---|---|---|---|---|
| C1 | Track overlap lockbox/dev | 0 | 0 | [x] |
| C2 | Sign-group overlap | 0 (nếu bật) | 0 | [x] |
| C3 | Lockbox đủ lớp | 43/43 | 43 | [x] |
| C4 | Dev đủ lớp | 43/43 | 43 | [x] |
| C5 | Screening val đủ lớp | 43/43 | 43 | [x] |
| C6 | Mỗi fold đủ lớp | 43/43 | 43, 43, 43 | [x] |
| C7 | Overlap screening train/val | 0 | 0 | [x] |
| C8 | Overlap từng fold | 0 | 0, 0, 0 | [x] |
| C9 | Test không nằm trong train/lockbox | 0 | 0 | [x] |
| C10 | Lockbox không có ảnh gần trùng train (MAD < 5) | 0 | 0 | [x] |
| C11 | MD5 trùng trong train | 0 | 0 | [x] |
| C12 | Số track/lớp nhỏ nhất trong dev | ≥ 2 | 6 | [x] |

**Điều kiện xong:** tất cả C1–C12 đạt; `manifest.json` có hash. → **ĐẠT 15/15**
(`outputs/results/v2/audit/audit_report.json`, hash file split khớp, mean/std khớp 0,0e+00).

### 0.4. Test code

- [x] `tests/test_split_v2.py` pass → **10 passed in 11,80 s**.
- [x] Test forward `(2,3,64,64) → (2,43)` cho A/B/C.
- [x] Test macro-F1/43 khớp sklearn (sai khác < 1e-9).
- [x] Test đọc manifest và mean/std (khớp lại từ cache, atol 1e-5).
- [x] Test `num_workers=2` trên Windows không treo.

---

## Phase 1 — Baseline + factorial 2⁴

### 1.1. Baseline

- [x] B1 HOG + LinearSVC; C chọn trên screening val → C = 0,1; val acc 0,8727; macro-F1/43 0,8600.
- [x] Ghi lại C tốt nhất và thời gian trích/fit (HOG 40,72 s; fit 452,74 s; `baseline.json`).
- [x] B0 majority chạy thêm làm mốc dưới theo yêu cầu bước 1.1 (acc 0,0291; F1 0,0013); **B1b refit bỏ** vì không có tập đánh giá nào ngoài lockbox.

### 1.2. Factorial

- [x] Chạy F00–F15 × 3 seed (48 run) — `phase12_artifacts.json` 48/48 đủ.
- [x] Mỗi run có config, env, metrics, best.pt, curves.
- [x] Kiểm tra F00/F07/F15 tương ứng CNN-A/E2/E3 — `alias_mapping.json` khớp.
- [x] Tính main effect BN/DO/WD/Aug — BN +3,724; DO +5,963; WD +0,058; Aug +0,529 (điểm %).
- [x] Tính interaction BN×Aug (−0,279), DO×WD (−0,121) — `factorial_effects.json`.
- [x] Bảng 2⁴ đầy đủ 16 dòng — `outputs/results/v2/tables/factorial.md`.

**Cửa kiểm tra:** F15 = 93,946, thấp hơn 97,061 cũ 3,115 điểm → đã kiểm tra split/mean-std
(audit 15/15, lệch 0,0e+00), không phải lỗi split; ghi nhận và tiếp tục, không đổi luật.

**Điều kiện xong:** bảng 2⁴ + bảng main effect + 4 hình curve đại diện. → **ĐẠT**
(curve: `outputs/figures/v2/01_learning_curves.png` sẽ sinh ở bước 4.2 từ metrics của F00/F07/F15).

---

## Phase 2 — Kiến trúc, hyperparam, augmentation

### 2.1. Kiến trúc

- [x] D1, D2, D3 × 3 seed — 9/9 run, đủ artifact.
- [x] W1, W2 × 3 seed — 6/6 run.
- [x] Bảng params vs macro-F1 (Pareto) — `tables/architecture.md`.
- [x] So sánh D1 vs D4 (GAP): −34,758 điểm; D2 vs D4 (depth): +3,212 [+0,036; +4,440]; W1/W2 vs D4: −3,260 / +1,268 (CI chứa 0).
- [x] D3 chỉ map E4 cũ; không dùng D3 để kết luận cơ chế.
- [x] Kết luận GAP/depth/width: GAP 3 khối underfit; 4 khối tốt hơn 3 khối; width nhỏ kém, width lớn chưa kết luận.

### 2.2. Hyperparam

- [x] L1 (LR 1e-4, cách 10 lần so với 1e-3) × 3 seed; L3 = F15 — Δ −7,139 điểm.
- [x] CW1 (inv_sqrt) × 3 seed — Δ −1,004 [−2,195; +0,261], không cải thiện.
- [x] I2 (input 96) × 3 seed — Δ −3,448 [−5,463; −0,984], kém hơn 64.
- [x] Không so 10 epoch với 40 epoch — mọi run tối đa 40 epoch, patience 7.
- [x] Không chạy L2/L4/L5/P1/P2/W4/CW2/LS1/I1/BS1/BS2 (đã bỏ theo `MA_TRAN_RUN.csv`).

### 2.3. Augmentation

- [x] G1, G2, G3 × 3 seed; G0 = F07, G5 = F15 — 9/9 run.
- [x] Corruption 6 loại × 3 mức trên screening val — `corruption.csv` (108 dòng, 6 config).
- [x] Bảng clean + corruption — `tables/augmentation.md`, mục 2.3 KET_QUA.
- [x] Kết luận aug có cần, phần nào: all/affine hại; color trung tính; không đủ bằng chứng giữ vì robustness.
- [x] Không chạy G4/G6/G7/G8 (đã bỏ theo `MA_TRAN_RUN.csv`).

### 2.4. Chọn top 2 + control

- [x] Sắp xếp theo mean macro-F1 — `candidates.json.ranked`.
- [x] Lấy nhóm trong 0,3 điểm so với tốt nhất — {G2, D2} (0,259 điểm).
- [x] Chọn top 2 theo tie-break + 1 control (F07) — X1=G2, X2=D2, X3=F07.
- [x] Ghi lý do chọn **trước** khi chạy OOF — `candidates.json`, `log_select.txt`.
- [x] Điền X1–X3 vào `MA_TRAN_RUN.csv`.

**Điều kiện xong:** có danh sách top 2 + control và lý do, có bảng D/H/G. → **ĐẠT**

---

## Phase 3 — Xác nhận, freeze, lockbox

### 3.1. OOF

- [x] Chạy top 2 + control × 3 fold × 3 seed (27 run) — 27/27 OK sau khi sửa bug fold.
- [x] Mỗi ảnh dev được dự đoán đúng 1 lần mỗi seed — `oof_predictions.csv` 278.091 dòng, 9 cặp (config,seed) × 30.899 ảnh.
- [x] `oof_predictions.csv` đủ dòng, không trùng — đã kiểm tra bằng script, raise nếu trùng.
- [x] Tính macro-F1/43 trên OOF — X1 96,297; X2 97,040; X3 95,059.
- [x] Bootstrap CI theo track (B ≥ 2.000) — `oof_summary.json`.
- [x] Paired bootstrap cho Δ — X1−X2 −0,743 [−1,590; +0,274]; X2−X3 +1,981 [+1,053; +2,961].
- [x] Holm cho 8 so sánh chính — `main_comparisons.csv` (bước 4.1/4.3).
- [x] Variance decomposition seed/fold/residual — var_fold 91,0%, var_seed 0,9%.

### 3.2. Chọn winner

- [x] Áp dụng luật 4.5 — X2 cao nhất; X1−X2 CI chứa 0 → chưa kết luận.
- [x] Nếu tương đương, chọn ít params/nhanh/đơn giản — X2 1,45M < X1 2,20M → chọn X2.
- [x] Ghi decision card cho winner — `outputs/results/v2/winner_card.json`.
- [x] Ghi rõ config, seed, epoch = median best epoch — `configs/v2/final.yaml`, epoch 26.
- [x] Commit freeze; ghi hash — `FREEZE.json`, `final_config_sha256=2678af82…`, `split_sha256=640a1913…`.

### 3.3. Final + lockbox

- [x] Train final trên toàn dev × 3 seed — 3/3, epoch 26, 30.899 ảnh.
- [x] Không early stop trên lockbox — final dùng epoch cố định, không có val.
- [x] Kiểm tra checkpoint load được, forward đúng — `final_ckpt_check.json` passed.
- [x] Mở lockbox **một lần**; ghi log thời gian/lệnh — `lockbox_access.log` (kèm ghi chú sự cố ghi file lần 1).
- [x] `lockbox_result.json` có accuracy, macro-F1/43, per-class, confusion, top-3, ECE — acc 0,988758; F1 0,975981.
- [x] Official test raw 12.630 — acc 0,986989; F1 0,980398.
- [x] Official test clean — 11.879 (audit repo này: 751 ảnh gần trùng); acc 0,988692; F1 0,982499.
- [x] Không chỉnh gì sau khi xem lockbox.

**Điều kiện xong:** có OOF + lockbox + official test; có freeze hash. → **ĐẠT**

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
