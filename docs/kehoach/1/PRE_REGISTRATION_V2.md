# ĐĂNG KÝ TRƯỚC (PRE-REGISTRATION) — THỰC NGHIỆM V2

| Thông tin | Giá trị |
|---|---|
| Ngày chốt | 02/11/2026 |
| Người thực hiện | Kỹ sư ML chính kiêm auditor tái lập (AI agent), theo yêu cầu chủ repo |
| Tài liệu gốc | `docs/kehoach/1/KE_HOACH_V2_DAY_DU.md` (v2.0) |
| Ma trận run | `docs/kehoach/1/MA_TRAN_RUN.csv` (bản gọn, 36 dòng cấu hình gồm alias) |
| Trạng thái | **ĐÓNG BĂNG trước run đầu tiên.** Mọi thay đổi phải tạo `PRE_REGISTRATION_V2_v2.md` kèm lý do |

> Tài liệu này chỉ chốt **giao thức**. Không chứa kết quả. Mọi số trong báo cáo phải
> sinh từ `outputs/results/v2/` và trace được về file/hash.

---

## 1. Câu hỏi nghiên cứu (đóng băng)

| Mã | Câu hỏi | Nơi trả lời |
|---|---|---|
| RQ1 | Kiến trúc nào cho macro-F1/43 tốt nhất trên track/sign-group-disjoint? | Block D + OOF |
| RQ2 | BN, dropout, WD, augmentation có thực sự cần? | Factorial 2⁴ (Block F) |
| RQ3 | Profile augmentation nào có tác dụng, kể cả dưới corruption? | Block G + corruption |
| RQ4 | LR, class weight, input size, batch size ảnh hưởng thế nào? | Block H |
| RQ5 | Cơ chế nào giải thích kết quả? | Phase 4 (curve, per-class, Grad-CAM, calibration, corruption) |

## 2. Endpoint

- **Chính:** macro-F1/43 = `f1_score(labels=range(43), average="macro", zero_division=0)`.
- **Phụ:** accuracy top-1, per-class F1/precision/recall/support, top-3, ECE (15 bin),
  Brier, corruption accuracy, params, latency model-only, train time, GPU memory.
- **Tập quyết định:** OOF trên toàn bộ dev (31.371 ảnh, 3 fold × 3 seed).
- **Tập xác nhận một lần:** lockbox 20% track (chỉ mở sau freeze).
- **Benchmark phụ:** official test raw 12.630 và clean 12.499; **không** dùng để chọn gì.

## 3. Tám so sánh chính (family Holm)

1. BN on vs off (trung bình qua 8 cặp config factorial).
2. Dropout 0,3 vs 0.
3. WD 1e-4 vs 0.
4. Augmentation on vs off.
5. GAP on vs off (3 khối, D1 vs D4).
6. Depth 4 vs 3 (D2 vs D4).
7. Width lớn 48/96/192 vs 32/64/128 (W2 vs D4).
8. LR 1e-3 vs 1e-4 (L3/F15 vs L1).

Hiệu chỉnh Holm–Bonferroni cho đúng 8 so sánh trên. Các so sánh khác (class weight,
input size, batch, label smoothing, aug components) là **exploratory**, ghi rõ
"chưa hiệu chỉnh", không dùng để tuyên bố thắng/thua tuyệt đối.

## 4. Luật quyết định (đóng băng)

Cho Δ = macro-F1(config) − macro-F1(reference), CI 95% **cluster bootstrap theo track**
(B = 2.000, đơn vị resample = `track_key`):

| Tình huống | Kết luận | Hành động |
|---|---|---|
| Δ ≥ +0,3 điểm % và CI_low > 0 | Tốt hơn | Giữ cấu hình |
| CI nằm hoàn toàn trong [−0,3; +0,3] điểm % | Tương đương | Chọn ít params/nhanh/đơn giản hơn |
| CI chứa 0 nhưng vượt ra ngoài ±0,3 | Chưa kết luận | Báo inconclusive; không tuyên bố |
| Δ ≤ −0,3 điểm % và CI_high < 0 | Kém hơn | Bỏ cấu hình |
| Aug clean-neutral, corruption Δ ≥ +0,5 và CI > 0 | Giữ có điều kiện | Ghi rõ đánh đổi clean |
| Aug không tăng clean, không tăng corruption | Bỏ | Giảm chi phí |
| Class weight tăng macro-F1 nhưng giảm accuracy > 0,5 | Trade-off | Chọn theo endpoint chính, báo accuracy |
| GAP/4 khối tương đương | Chọn GAP/3 khối | Ít params hơn |

**Tie-break khi tương đương:** (1) ít params; (2) latency thấp; (3) ít aug; (4) ít epoch;
(5) đơn giản để giải thích.

**Ngưỡng tương đương:** chênh lệch < 0,3 điểm % (0,003) là tương đương. CI chứa 0 →
không dùng từ "tốt hơn".

## 5. Luật dừng và tính hợp lệ của run

- Tối đa **40 epoch**, early stopping **patience 7** theo val macro-F1.
- Run không hợp lệ: loss NaN/Inf; > 10% epoch NaN; không có checkpoint; config/split hash
  không khớp; val thiếu lớp mà không ghi rõ.
- Run hợp lệ nhưng thất bại (NaN, phân kỳ) **vẫn báo cáo**, không xóa, không chạy lại để
  "làm đẹp".
- Không dùng lockbox/official test để early stop, chọn epoch, chọn ngưỡng.
- Chạy lại do bug: tạo `_r2` và ghi log lý do; không ghi đè artifact cũ.

## 6. Freeze và mở lockbox

1. Chốt winner từ OOF theo luật mục 4.
2. Ghi config winner, seed, epoch = median best epoch của 9 run OOF.
3. Commit code + config + split manifest; ghi `git_commit`, `config_sha256`,
   `split_sha256` vào `FREEZE_V2.md`.
4. Mở lockbox **đúng một lần**; ghi thời gian, lệnh, người chạy vào
   `outputs/results/v2/lockbox_access.log`.
5. Sau khi mở: không chỉnh config/metric. Nếu phát hiện bug, tạo
   `lockbox_result_v2.json` mới và ghi lý do; không ghi đè.

## 7. Phạm vi run (khớp `MA_TRAN_RUN.csv`)

Bản gọn của kế hoạch v2.0 (mục 6.4/6.5/6.9) đã **chủ động bỏ** các cấu hình
L2/L4/L5/P1/P2/W4/CW2/LS1/I1/BS1/BS2, G4/G6/G7/G8, B0/B1b, C1/W3 để giảm số run.
Ma trận chạy chính thức là `MA_TRAN_RUN.csv`: B1; F00–F15; D1–D3, W1–W2 (D4=F15);
L1, CW1, I2 (L3=F15); G1–G3 (G0=F07, G5=F15); X1–X3; FINAL. Alias không chạy lại.
Nếu chạy thêm cấu hình ngoài ma trận, phải ghi rõ là exploratory và không dùng để
kết luận chính.

## 8. Sai lệch so với kế hoạch (nếu có)

| Ngày | Sai lệch | Lý do | Ảnh hưởng kết luận |
|---|---|---|---|
| — | Chưa có | — | — |
