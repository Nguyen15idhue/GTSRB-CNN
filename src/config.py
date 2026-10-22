"""Hằng số và đường dẫn dùng chung cho toàn bộ pipeline.

Giai đoạn 1 chỉ cần các hằng số nền tảng. Phần đọc/validate YAML sẽ được
triển khai đầy đủ ở Giai đoạn 2.
"""

from pathlib import Path

# Thư mục gốc repo (file này nằm trong src/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dữ liệu (không commit).
DATA_ROOT = PROJECT_ROOT / "data"
GTSRB_ROOT = DATA_ROOT / "gtsrb"
# Cấu trúc trên đĩa sau Giai đoạn 1:
#   data/gtsrb/GTSRB/Training/<class>/<track>_<frame>.ppm  (archive chính thức, 39.209 ảnh)
#   data/gtsrb/GTSRB/Final_Test/Images/<index>.ppm          (torchvision, 12.630 ảnh)
#   data/gtsrb/GT-final_test.csv
TRAIN_IMAGES_DIR = GTSRB_ROOT / "GTSRB" / "Training"
TEST_IMAGES_DIR = GTSRB_ROOT / "GTSRB" / "Final_Test" / "Images"
TEST_GT_CSV = GTSRB_ROOT / "GT-final_test.csv"

# Kết quả / hình (được commit nếu nhỏ).
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Nhãn lớp dùng chung cho train, evaluate, confusion matrix và app.
CLASS_NAMES_PATH = PROJECT_ROOT / "configs" / "class_names.json"

# Hằng số cốt lõi của bài toán.
NUM_CLASSES = 43
IMG_SIZE = 64
DEFAULT_SEED = 42

# Số liệu công bố để đối chiếu, không dùng thay cho số đếm thực tế.
PUBLISHED_TRAIN_COUNT = 39_209
PUBLISHED_TEST_COUNT = 12_630
