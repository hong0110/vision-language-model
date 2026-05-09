# 🧾 Ứng dụng Mô hình VLM (Donut) Trích Xuất Dữ Liệu Hóa Đơn

Dự án này là một ứng dụng quản lý chi tiêu cá nhân thông minh, sử dụng **Mô hình Ngôn ngữ - Thị giác (Vision-Language Model - VLM)**, cụ thể là kiến trúc **Donut (Document Understanding Transformer)** đã được tinh chỉnh (fine-tune) để đọc và tự động trích xuất thông tin từ ảnh chụp hóa đơn tiếng Việt.

---

## Các Tính Năng Chính

- **🤖 Trích xuất AI tự động (Zero-OCR):** Sử dụng mô hình VLM Donut để trực tiếp hiểu và đọc các trường thông tin quan trọng từ hóa đơn (Tên cửa hàng, Ngày tháng, Tổng tiền) mà không cần qua bước OCR truyền thống rườm rà.
- **📊 Giao diện Streamlit Trực quan:** Upload hóa đơn dễ dàng bằng kéo-thả. Cung cấp biểu đồ thống kê trực quan (Plotly), theo dõi phần trăm ngân sách chi tiêu.
- **🛡️ Luồng xác nhận (Human-in-the-loop):** Dữ liệu do AI đọc được sẽ được hiển thị trong "Hàng chờ xác nhận". Người dùng có thể kiểm tra, sửa đổi (nếu cần) trước khi chính thức đưa vào sổ chi tiêu.
- **📈 Thống kê & Phân bổ:** Phân loại chi tiêu theo danh mục (Ăn uống, Mua sắm...), lọc theo tháng/năm, và cảnh báo ngân sách.

---

## 🆚 Tại Sao Lại Dùng VLM Thay Vì OCR Truyền Thống?

Dự án có đi kèm một module `baseline_ocr` sử dụng **Tesseract OCR + Regex** để làm hệ quy chiếu so sánh. Qua thử nghiệm thực tế với hóa đơn Việt Nam:

| Phương pháp | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- |
| **Baseline (Tesseract + Regex)** | Dễ cài đặt ban đầu, nhẹ. | Dễ bị "đánh lừa" bởi nhiễu dấu câu, định dạng lệch. Regex thường xuyên bắt nhầm số (vd: bắt nhầm mã nhân viên thành tổng tiền). Không hiểu được cấu trúc hóa đơn. |
| **VLM (Donut - Fine-tuned)** | **Đọc chính xác, hiểu được ngữ cảnh** của tờ hóa đơn. Trực tiếp xuất ra JSON có cấu trúc. Không cần quy tắc Regex cứng nhắc. | Tốn tài nguyên tính toán (RAM/VRAM) hơn. Yêu cầu quá trình fine-tune mô hình. |

👉 **Kết luận:** VLM (Donut) tỏ ra vượt trội và ổn định hơn rất nhiều khi xử lý các hóa đơn chụp bằng điện thoại với ánh sáng, góc chụp và định dạng đa dạng.

---

## 📁 Cấu Trúc Dự Án

```text
Invoice-Reader/
│
├── app/
│   ├── app.py                  # Giao diện web chính (Streamlit)
│   └── requirements.txt        # Các thư viện cần thiết cho app
│
├── baseline_ocr/               # Phương pháp đối chứng (Tesseract + Regex)
│   ├── extract_total.py        # Logic trích xuất bằng Regex
│   ├── ocr_infer.py            # Chạy Tesseract OCR
│   └── test_ocr.py             # Script chạy thử nghiệm Baseline
│
├── data/                       # Dữ liệu hình ảnh hóa đơn và nhãn (annotations)
│   ├── test/                   # Dữ liệu kiểm thử
│   ├── train/                  # Dữ liệu huấn luyện
│   └── val/                    # Dữ liệu đánh giá (validation)
│
├── report/                     # Thư mục chứa các báo cáo kết quả và đánh giá
│
├── vlm/                        # Thư mục chứa cấu hình và inference của VLM
│   ├── donut_infer.py          # Script tải model và chạy suy luận (Inference)
│   └── model/                  # Trọng số của mô hình Donut (đã được fine-tune)
│
├── mainTrain.py                # Script dùng để huấn luyện (fine-tune) mô hình
└── README.md                   # Tài liệu dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt

### 1. Yêu cầu hệ thống
- Python 3.10+
- Khuyến nghị có GPU (CUDA) nếu muốn chạy mô hình nhanh hơn. Tuy nhiên, ứng dụng vẫn chạy tốt trên CPU/MacOS.

### 2. Cài đặt môi trường
Tạo và kích hoạt môi trường ảo:
```bash
python -m venv .venv
source .venv/bin/activate  # (Với MacOS/Linux)
# hoặc .venv\Scripts\activate (Với Windows)
```

Cài đặt các thư viện phụ thuộc:
```bash
pip install -r app/requirements.txt
pip install torch torchvision transformers 
```
*(Nếu muốn chạy thử Baseline OCR, bạn cần cài thêm thư viện `pytesseract` và phần mềm Tesseract OCR trên máy).*

### 3. Chạy Ứng Dụng
Từ thư mục gốc của dự án, khởi động Streamlit:
```bash
streamlit run app/app.py
```
Sau đó truy cập vào trình duyệt tại địa chỉ: `http://localhost:8501`

---

## 🧑‍💻 Hướng Dẫn Sử Dụng
1. Tại sidebar bên trái, nhập **Ngân sách** tối đa của bạn.
2. Ở màn hình chính, tab **Upload hóa đơn**, chọn các ảnh hóa đơn (hỗ trợ kéo thả nhiều ảnh cùng lúc).
3. Nhấn **Trích xuất**. VLM sẽ đọc và phân tích từng hóa đơn.
4. Cuộn xuống phần **Xác nhận Hóa Đơn**, đối chiếu thông tin AI đọc được với hóa đơn thực tế. Điền thêm danh mục và nhấn **✅ Đưa vào sổ chi tiêu**.
5. Xem các biểu đồ thống kê tổng quan tự động cập nhật ở cột bên phải!
