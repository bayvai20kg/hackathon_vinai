# Pipeline Pre-label 11 Lớp: YOLO11 + SAM 3 + Active Learning + CVAT

Pipeline gán nhãn bán tự động **Instance Segmentation (YOLO-seg polygon)** cho **11 lớp giao thông** cố định (`0..10`), kết hợp **YOLO11-seg**, **SAM 3** (tự động fallback sang **SAM 2.1** trên GPU VRAM nhỏ), **Active Learning**, và **CVAT** (xuất/nhập ZIP thủ công).

## Cài đặt nhanh
```bash
pip install -r requirements.txt
```
Đặt ảnh vào `images/` (và tùy chọn nhãn cũ vào `labels/`). Khi chạy, các file trọng số mặc định (`yolo11n-seg.pt`, `yoloe-11s-seg-pf.pt`, `sam2.1_t.pt`) sẽ tự động tải vào `weights/`.

## 1. Bảng 11 lớp cố định (`configs/classes_11.yaml`)

| ID | Tên lớp | Mô tả & Luật đặc biệt |
|---:|---|---|
| 0 | `Car` | Ô tô con, SUV, minivan, bán tải, taxi, xe 9–15 chỗ, cứu thương dạng van |
| 1 | `Bus` | Xe buýt / xe khách từ 16 chỗ trở lên |
| 2 | `Truck` | Xe tải, xe ben, xe bồn, cứu hỏa, **đầu kéo (chỉ phần cabin & thân đầu kéo)** |
| 3 | `Other vehicle` | Ba gác, xích lô, xe lam, xe công trình, **rơ-moóc & sơ-mi rơ-moóc** |
| 4 | `Motorcycle` | Xe máy không có người ngồi (đang đỗ hoặc đang dắt) |
| 5 | `Motorcyclist` | Xe máy có người ngồi (**1 instance duy nhất phủ cả xe và mọi người trên xe**) |
| 6 | `Bicycle` | Xe đạp không có người ngồi (đang đỗ hoặc đang dắt) |
| 7 | `Cyclist` | Xe đạp có người đi (**1 instance duy nhất phủ cả xe và người**) |
| 8 | `Pedestrian` | Người đi bộ, đứng, ngồi, trẻ em, người dắt xe |
| 9 | `Sign` | Biển báo đường bộ (**chỉ phủ mặt biển, không phủ cột**) |
| 10 | `Traffic light` | Đèn giao thông (**cả hộp đèn tín hiệu, không chỉ bóng sáng**) |

> **Lưu ý về dữ liệu `labels/` cũ:** Thư mục `labels/` hiện tại chỉ có 4 lớp bbox (`0..3`), thiếu phân đoạn đa giác và chưa tách các lớp `4..10` (đồng thời gộp đầu kéo + rơ-moóc). Pipeline giữ nguyên `images/` và `labels/` gốc, chỉ dùng `labels/` làm tín hiệu tham chiếu (prior) ở bước bootstrap và phát hiện đối tượng bị sót trong Active Learning.

---

## 2. Chuyển đổi giữa SAM 3 (`sam3.pt`) và SAM 2.1 Fallback

- **Mặc định an toàn trên máy VRAM 4GB / RAM 8GB:** Nếu chưa có `weights/sam3.pt` hoặc GPU VRAM < 7.5GB, `src/segmenter.py` tự động dùng `sam2.1_t.pt` qua cùng giao diện `Segmenter`.
- **Khi chạy trên máy GPU mạnh / Colab (hoặc có `weights/sam3.pt`):**
  1. Đặt checkpoint SAM 3 vào `weights/sam3.pt` (tải từ `facebook/sam3` trên Hugging Face sau khi được cấp quyền).
  2. Pipeline sẽ tự kích hoạt `SAM3Predictor` (`imgsz=1008`) khi thấy `weights/sam3.pt` và GPU >= 7.5GB VRAM, hoặc ép chạy bằng cờ `--force-sam3`.

---

## 3. Vòng Lặp Active Learning Đầy Đủ

### Bước 1 — Warm-up & Export lên CVAT
```bash
python run_pipeline.py warmup-export --num-images 200 --batch-name warmup_batch
```
- Chọn đều ảnh ngày/đêm trên 24 camera, chạy bootstrap (`YOLOE` + hợp nhất `labels/` cũ + gộp người/xe máy) và sinh mask bằng SAM.
- Kết quả xuất tại `pipeline_work/cvat_export/warmup_batch/`:
  - `cvat_labels.json`: Dán vào phần **Raw** khi tạo Project/Task trên CVAT để có đủ 11 lớp + màu chuẩn.
  - `warmup_batch_coco.zip` (khuyên dùng **Upload annotations -> COCO 1.0** trong CVAT) hoặc `warmup_batch_yolo_seg.zip`.
  - `review_manifest.csv`: Thứ tự ảnh cần ưu tiên sửa và lý do.

### Bước 1b — Nhập lại nhãn đã sửa từ CVAT & Huấn luyện Warm-up YOLO11-seg
Sau khi sửa trên CVAT và **Export dataset** (`COCO 1.0` hoặc `YOLO Segmentation 1.0` dạng `.zip`):
```bash
python run_pipeline.py import-cvat --source path/to/cvat_export.zip
python run_pipeline.py train --run-name round_01 --epochs 30
```

### Bước 2, 3, 4 — Chạy Inference + Active Learning + SAM Auto-Labeling + Xuất CVAT
```bash
python run_pipeline.py al-step --weights pipeline_work/runs/round_01/weights/best.pt --batch-name al_round_01
```
- **Ảnh dễ** (tất cả bbox có `conf > 0.85`, không nhập nhằng cặp lớp `Motorcycle`/`Motorcyclist`...): dùng bbox YOLO11 nạp vào SAM sinh mask và lưu tự động vào `pipeline_work/auto_labeled/`.
- **Ảnh khó** (có bbox `< 0.60`, nhập nhằng top-2 class, sót nhãn cũ, mask lỗi hoặc đầu kéo cần kiểm tra tách rơ-moóc): xuất ra `pipeline_work/cvat_export/al_round_01/` kèm sẵn mask gợi ý.

### Bước 5 — Nhập lại nhãn sửa từ CVAT & Retrain YOLO11-seg
```bash
python run_pipeline.py import-cvat --source path/to/al_round_01_reviewed.zip
python run_pipeline.py train --run-name round_02 --weights pipeline_work/runs/round_01/weights/best.pt --epochs 25
```
Lặp lại `al-step -> import-cvat -> train` cho đến khi toàn bộ dataset đạt chất lượng mong muốn.

---

## 4. Chạy thử nhanh toàn bộ vòng lặp trên 50 ảnh
```bash
python run_pipeline.py run-demo-50 --num-images 50 --epochs 5
```
Kết quả được lưu riêng tại `pipeline_work/demo_50/` (không làm bẩn `human_verified/` hay `auto_labeled/`), bao gồm báo cáo JSON/CSV, ảnh vẽ mask overlay (`visuals_bootstrap/`, `visuals_yolo11_al/`) và các gói CVAT ZIP.
