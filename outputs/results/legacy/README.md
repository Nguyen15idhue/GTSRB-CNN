# Legacy leaky — KHÔNG dùng để kết luận

Các run trong thư mục này dùng `split_random`: val chứa frame cùng track với train
(xem `outputs/results/leak_delta.json`: 95,22% frame val có frame train cách ±1).

- `summary_random_split.csv`: bản sao các dòng `split=random` từ `outputs/results/summary.csv`.
- Chỉ dùng trong phụ lục báo cáo, nhãn `legacy_leaky`.
- Không trộn số legacy với số v2 trong cùng bảng kết luận.
