# HƯỚNG DẪN SỬ DỤNG & BÀN GIAO KẾT QUẢ MÔ HÌNH BASELINE (MÃ B – NGUYỄN XUÂN QUY)

Tài liệu này tóm tắt toàn bộ các tệp đầu ra, định dạng dữ liệu, kết quả thực nghiệm và cách tích hợp mã nguồn của **Thành viên B (Nguyễn Xuân Quy – Mô hình Baseline CNN `MobileNetV2` + `LSTM/GRU`)** để **Thuận Thiên (Mã D)**, **Thùy Linh (Mã C)** và **Như Hiếu (Mã A)** sử dụng trực tiếp cho việc chạy `evaluate.py` (T15, T19, T20), ghép giao diện `demo.py` (T22) và gộp báo cáo cuối kỳ (T24, T26).

---

## 1. Cấu trúc thư mục và Danh mục tệp bàn giao tại `c:\Thị giác máy tính\Công việc\T08-T09`

| Hạng mục | Đường dẫn tệp / thư mục | Mô tả chi tiết & Quy ước kỹ thuật |
|---|---|---|
| **T08 – Trích đặc trưng** | [extract_features.py](file:///c:/Thị giác máy tính/Công việc/T08-T09/extract_features.py)<br/>Thư mục `features/` | Đọc `splits.csv` (1.986 video RLVS) và `cross_test.csv` (400 video RWF-2000) từ thư mục của Hiếu (Mã A); trích vector đặc trưng `MobileNetV2` (`1280` chiều) cho cả 3 cấu hình $T \in \{8, 16, 20\}$ và bản tăng cường `augment_clip` ($T=16$). Lưu cả gói nén tốc độ cao `.npz` (`features_n8.npz`, `features_n16.npz`, `features_n20.npz`, `features_cross_n*.npz`) lẫn từng file `.npy` và bảng chỉ mục `.csv`. |
| **T09 – Huấn luyện & Checkpoint** | [train_baseline.py](file:///c:/Thị giác máy tính/Công việc/T08-T09/train_baseline.py)<br/>[baseline.pt](file:///c:/Thị giác máy tính/Công việc/T08-T09/baseline.pt) | Script huấn luyện và đánh giá toàn diện. Checkpoint tốt nhất `baseline.pt` (~`2,07 MB`, cấu hình `GRU` 1 lớp, `num_frames=16`, `hidden_size=128`, `dropout=0.3`, `threshold=0.50`) được chọn trên tập `val` RLVS tại Epoch 15 (`Val F1 = 0.9834`, `Val Loss = 0.0517`). |
| **T19 & T20 – File dự đoán cho Thiên (D)** | Thư mục [preds_baseline/](file:///c:/Thị giác máy tính/Công việc/T08-T09/preds_baseline/):<br/>• [preds_baseline_val.csv](file:///c:/Thị giác máy tính/Công việc/T08-T09/preds_baseline/preds_baseline_val.csv) (`298` dòng)<br/>• [preds_baseline_test.csv](file:///c:/Thị giác máy tính/Công việc/T08-T09/preds_baseline/preds_baseline_test.csv) (`297` dòng)<br/>• [preds_baseline.csv](file:///c:/Thị giác máy tính/Công việc/T08-T09/preds_baseline/preds_baseline.csv) (`595` dòng `val+test`)<br/>• [preds_baseline_cross_test.csv](file:///c:/Thị giác máy tính/Công việc/T08-T09/preds_baseline/preds_baseline_cross_test.csv) (`400` dòng)<br/>*(kèm bản sao [preds_baseline.csv](file:///c:/Thị giác máy tính/Công việc/T08-T09/preds_baseline.csv) ở thư mục gốc)* | Đúng yêu cầu của Thiên: *"gộp các file lại trong 1 folder có val và test cho gọn"*. Tất cả các file đều tuân thủ tuyệt đối 5 cột chuẩn:<br/>`video_id,split,y_true,y_prob,y_pred`<br/>• `y_true`: nhãn thực (`0` = NonViolence, `1` = Violence)<br/>• `y_prob`: xác suất lớp Violence $\in [0.0, 1.0]$<br/>• `y_pred`: nhãn dự đoán `0/1` tại ngưỡng $\tau = 0.50$. |
| **T10 – Bảng Ablation Study** | [ablation_baseline.csv](file:///c:/Thị giác máy tính/Công việc/T08-T09/ablation_baseline.csv) | Bảng so sánh **10 cấu hình biến thể** ($T \in \{8, 16, 20\}$, `LSTM` vs `GRU`, `hidden_size` $\in \{128, 256\}$, 1–2 lớp), có đầy đủ 8 cột tiếng Việt bắt buộc đã chốt với Thiên:<br/>`Mô hình (LSTM/GRU),Số frame (T),Hidden size,Accuracy,Precision,Recall (Violence),F1,ROC-AUC` (kèm các cột mở rộng trên tập Test và Cross-Test). |
| **T14 – So sánh Tăng cường dữ liệu** | [t14_augmentation_baseline.csv](file:///c:/Thị giác máy tính/Công việc/T08-T09/t14_augmentation_baseline.csv)<br/>`outputs_baseline/t14_augmentation_baseline.csv` | Bảng đối chiếu mô hình Baseline `GRU (T=16, H=128)` khi huấn luyện **Không tăng cường** vs. **Có tăng cường (`augment_clip`)** trên cả 3 tập `RLVS Val`, `RLVS Test` và `RWF-2000 Cross-Test`. |
| **T22 – Module tiền xử lý & suy luận cho Demo** | [baseline_inference.py](file:///c:/Thị giác máy tính/Công việc/T08-T09/baseline_inference.py) | Đóng gói hàm dùng chung `preprocess_video(video_path, num_frames=16)` trả về tensor `(1, 16, 3, 224, 224)` và `predict_baseline(video_path)` trả về `(prob, label)` để Thiên gắn vào `demo.py` (Gradio/Streamlit). |
| **T24 – Biểu đồ & Báo cáo mục 3** | • [bao_cao_baseline_T24.md](file:///c:/Thị giác máy tính/Công việc/T08-T09/bao_cao_baseline_T24.md)<br/>• [loss_curve.png](file:///c:/Thị giác máy tính/Công việc/T08-T09/loss_curve.png)<br/>• [confusion_matrix_baseline.png](file:///c:/Thị giác máy tính/Công việc/T08-T09/confusion_matrix_baseline.png)<br/>• [roc_curve_baseline.png](file:///c:/Thị giác máy tính/Công việc/T08-T09/roc_curve_baseline.png)<br/>• [ablation_comparison.png](file:///c:/Thị giác máy tính/Công việc/T08-T09/ablation_comparison.png) | Bản thảo hoàn chỉnh Mục 3 của báo cáo bằng tiếng Việt (có sơ đồ kiến trúc Mermaid, bảng số liệu, phân tích chuyên sâu T08–T10–T14–T19–T20–T21–T22) và 4 biểu đồ độ phân giải cao dùng nhãn trục tiếng Việt. |

---

## 2. Tóm tắt số liệu chính để Thiên (Mã D) đưa vào Bảng so sánh tổng hợp (T19, T20, T26)

### 2.1. Hiệu năng mô hình Baseline tốt nhất (`C05_GRU_T16_H128_L1` – `baseline.pt`)
| Tập đánh giá | Số mẫu | Accuracy | Precision | Recall (Violence) | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **RLVS Validation (`val`)** | 298 | **0.9832** | 0.9737 | **0.9933** | **0.9834** | **0.9986** |
| **RLVS Test (`test` – T19)** | 297 | **0.9764** | **0.9732** | **0.9797** | **0.9764** | **0.9977** |
| **RWF-2000 Cross-Test (`cross_test` – T20)** | 400 | 0.5550 | 0.5396 | **0.7500** | 0.6276 | 0.6514 |

*(Nếu dùng bản huấn luyện có tăng cường dữ liệu `outputs_baseline/baseline_with_aug.pt` trên tập `RWF-2000 Cross-Test`: `Accuracy = 0.5925`, `Precision = 0.5738`, `Recall = 0.7050`, `F1 = 0.6321`, `ROC-AUC = 0.6790`).*

### 2.2. Thông số tốc độ suy luận và dung lượng (để so sánh với mô hình Video của Linh – Mã C)
- **Kiến trúc:** `MobileNetV2` (Pretrained ImageNet, frozen, `2.223.872` tham số / `8,48 MB`) + `GRU` 1 lớp (`hidden_size = 128`, `541.569` tham số / `2,07 MB`).
- **Tổng số tham số toàn tuyến:** **`2.765.441` tham số** (**`10,55 MB`** trọng số `float32`).
- **Dung lượng file checkpoint `baseline.pt`:** **`2,07 MB`**.
- **Tốc độ suy luận trên GPU RTX 3050 (đã warm-up cache):**
  - Đọc trực tiếp từ video gốc `.mp4` (End-to-End gồm giải mã OpenCV + MobileNetV2 + GRU): **`293,14 ms / video`** (~**3,41 video/giây**), trong đó riêng `MobileNetV2` mất `52,67 ms` và `GRU` mất `1,76 ms`.
  - Đọc từ file frame `.npy` ($16 \times 224 \times 224 \times 3$): **`105,30 ms / video`** (~**9,50 video/giây**).

---

## 3. Hướng dẫn tích hợp cho Thiên (Mã D) và Linh (Mã C)

### 3.1. Cách đọc các file dự đoán trong `evaluate.py` (T15, T19, T20)
```python
import pandas as pd

# 1. Đọc riêng tập Val hoặc tập Test RLVS:
df_val = pd.read_csv("preds_baseline/preds_baseline_val.csv")      # 298 dòng (split == 'val')
df_test = pd.read_csv("preds_baseline/preds_baseline_test.csv")    # 297 dòng (split == 'test')

# 2. Hoặc đọc file gộp cả Val + Test (595 dòng) và lọc theo cột 'split':
df_all = pd.read_csv("preds_baseline/preds_baseline.csv")
df_test_filtered = df_all[df_all["split"] == "test"]

# 3. Đọc kết quả đánh giá ngoài phân phối trên 400 video RWF-2000 (T20):
df_cross = pd.read_csv("preds_baseline/preds_baseline_cross_test.csv")  # 400 dòng (split == 'cross')
```

### 3.2. Cách gọi hàm tiền xử lý và dự đoán trong `demo.py` (T22 – Gradio / Streamlit)
Trong `demo.py`, Thiên (Mã D) và Linh (Mã C) chỉ cần import trực tiếp từ `baseline_inference.py`:

```python
from baseline_inference import (
    preprocess_video,
    predict_baseline,
    predict_baseline_detailed,
)

# 1. Hàm tiền xử lý dùng chung cho cả mô hình của Quy (B) và Linh (C):
# Hỗ trợ cả đường dẫn video (.mp4, .avi, .mov, .mkv) lẫn file frame .npy
video_tensor = preprocess_video("duong_dan_video.mp4", num_frames=16)
# -> torch.FloatTensor kích thước (1, 16, 3, 224, 224), chuẩn RGB + ImageNet

# 2. Hàm dự đoán nhanh theo đúng chữ ký đã thống nhất (trả về prob và nhãn):
prob, label = predict_baseline("duong_dan_video.mp4")
# -> Ví dụ: prob = 0.9991 (float trong [0, 1]), label = "Violence" (hoặc "NonViolence")

# 3. Hàm dự đoán chi tiết (nếu muốn hiển thị thêm nhãn tiếng Việt và thời gian chạy trên UI):
info = predict_baseline_detailed("duong_dan_video.mp4")
# -> info["prob"], info["label_vi"] ("Bạo lực (Violence)" / "Bình thường (NonViolence)"),
#    info["inference_time_ms"]
```

---

## 4. Các câu lệnh chạy tái lập toàn bộ kết quả (CLI Commands)

Nếu muốn chạy kiểm tra lại bất kỳ bước nào từ dòng lệnh tại thư mục `c:\Thị giác máy tính\Công việc\T08-T09`:

```powershell
# 1. Kiểm thử nhanh module suy luận Demo (T22 - chạy trên 10 video thực tế RLVS & RWF-2000):
python baseline_inference.py --test

# 2. Dự đoán thử trên một file video hoặc file .npy bất kỳ:
python baseline_inference.py --video "c:\Lập Trình Trí Tuệ Nhân Tạo\violence\data\rlvs\Real Life Violence Dataset\Violence\V_100.mp4"

# 3. Huấn luyện lại toàn bộ T09, T10 (10 cấu hình Ablation), T14 (Augmentation), T19 và T20:
python train_baseline.py --run_ablation --run_t14

# 4. Trích xuất lại đặc trưng MobileNetV2 (T08 - tự động dùng cache nếu đã có):
python extract_features.py --all
```
