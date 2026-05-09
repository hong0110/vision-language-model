import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from PIL import Image
from vlm.donut_infer import extract_invoice, load_model
import time
import hashlib
import datetime
import plotly.express as px


# ================= CONFIG =================
st.set_page_config(page_title="Quản lý chi tiêu", layout="wide")

# ================= LOAD MODEL =================
with st.spinner("Đang khởi động VLM (chỉ lần đầu)..."):
    load_model()

# ================= SESSION STATE =================
if 'expenses' not in st.session_state:
    st.session_state['expenses'] = []

if 'processed_hashes' not in st.session_state:
    st.session_state['processed_hashes'] = []

if 'pending_invoices' not in st.session_state:
    st.session_state['pending_invoices'] = []


# ================= HASH FUNCTION =================
def get_image_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()


# ================= DATAFRAME =================
if st.session_state['expenses']:
    df = pd.DataFrame(st.session_state['expenses'])

    df['category'] = df.get('category', "Khác")

    df['purchase_date'] = pd.to_datetime(
        df['purchase_date'],
        errors='coerce'
    )

    df['Year'] = df['purchase_date'].dt.year
    df['Month'] = df['purchase_date'].dt.month
    df['Display_Date'] = df['purchase_date'].dt.strftime('%d/%m/%Y')

else:
    df = pd.DataFrame()


# ================= SIDEBAR =================
st.sidebar.title("Bảng điều khiển")

ngan_sach = st.sidebar.number_input(
    "Ngân sách (VNĐ)",
    0,
    5000000,
    step=500000
)

if not df.empty:

    years = sorted(
        df['Year'].dropna().unique().astype(int),
        reverse=True
    )

    months = sorted(
        df['Month'].dropna().unique().astype(int)
    )

    y = st.sidebar.selectbox(
        "Năm",
        ["Tất cả"] + list(years)
    )

    m = st.sidebar.selectbox(
        "Tháng",
        ["Tất cả"] + list(months)
    )

    filtered_df = df.copy()

    if y != "Tất cả":
        filtered_df = filtered_df[
            filtered_df['Year'] == y
        ]

    if m != "Tất cả":
        filtered_df = filtered_df[
            filtered_df['Month'] == m
        ]

else:
    filtered_df = pd.DataFrame()


# ================= MAIN =================
st.title("Quản lý chi tiêu với VLM (Donut)")

col1, col2 = st.columns([1, 1.5])


# ================= INPUT =================
with col1:

    tab1, tab2 = st.tabs([
        "Upload hóa đơn",
        "Nhập tay"
    ])

    # =========================================================
    # DONUT TAB
    # =========================================================
    with tab1:

        uploaded_files = st.file_uploader(
            "Upload ảnh hóa đơn",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Trích xuất"):

            for file in uploaded_files:

                file_bytes = file.getvalue()
                file_hash = get_image_hash(file_bytes)

                already_processed = (
                    file_hash in st.session_state['processed_hashes']
                )

                already_pending = any(
                    p.get('file_hash') == file_hash
                    for p in st.session_state['pending_invoices']
                )

                if already_processed or already_pending:
                    st.warning(
                        f"Đã xử lý hoặc đang chờ xác nhận: {file.name}"
                    )
                    continue

                image = Image.open(file)

                try:
                    progress = st.progress(0)
                    status = st.empty()

                    status.text(
                        f"Đang khởi động VLM cho {file.name}..."
                    )

                    progress.progress(30)

                    with st.spinner("Đang xử lý ảnh..."):

                        data = extract_invoice(image)

                    # ================= DEBUG =================
                    st.write("DEBUG OUTPUT:", data)

                    progress.progress(100)

                    status.success(
                        f"Quét thành công: {file.name}!"
                    )

                    # ================= FIXED TOTAL =================
                    total = None

                    if data and "parsed" in data:
                        parsed_data = data["parsed"]
                    
                        if "total" in parsed_data:
                            total = parsed_data["total"]

                    # ================= CONVERT =================
                    try:
                        amount = int(str(total).replace(",", ""))
                    except:
                        amount = 0

                    # ================= ADD TO PENDING =================
                    st.session_state['pending_invoices'].append({
                        "file_name": file.name,
                        "file_hash": file_hash,
                        "amount": amount,
                        "data": data,
                        "total": total
                    })

                except Exception as e:
                    st.error(f"Lỗi: {e}")

            time.sleep(1)
            st.rerun()

        # =========================================================
        # PENDING INVOICE CONFIRMATION
        # =========================================================
        if st.session_state['pending_invoices']:

            st.markdown("---")
            st.subheader("📝 Xác nhận Hóa Đơn")

            for idx, inv in enumerate(
                st.session_state['pending_invoices']
            ):

                with st.expander(
                    f"📄 Hóa đơn: {inv['file_name']}",
                    expanded=True
                ):

                    # ================= SHOW EXTRACTED INFO =================
                    data = inv.get("data", {})

                    st.write("📌 Thông tin VLM đọc được:")

                    st.json(data)

                    # ================= METRIC =================
                    if inv['total']:

                        st.metric(
                            "💰 Tổng tiền",
                            f"{int(inv['amount']):,} VNĐ"
                        )

                    else:
                        st.warning(
                            "Không đọc được tổng tiền. "
                            "Vui lòng nhập tay."
                        )

                    # ================= FORM =================
                    with st.form(f"confirm_form_{idx}"):

                        default_desc = "Hóa đơn"

                        parsed_data = data.get("parsed", {})

                        if "store" in parsed_data:
                            default_desc = parsed_data["store"]

                        # ===== DATE =====
                        try:
                            default_date = datetime.datetime.strptime(
                                parsed_data.get("date", ""),
                                "%Y-%m-%d"
                            ).date()
                        except:
                            default_date = datetime.datetime.now().date()

                        date = st.date_input(
                            "Ngày hóa đơn",
                            default_date
                        )

                        desc = st.text_input(
                            "Nội dung",
                            value=default_desc
                        )

                        cat = st.selectbox(
                            "Danh mục",
                            [
                                "Ăn uống",
                                "Mua sắm",
                                "Di chuyển",
                                "Tiện ích",
                                "Khác"
                            ]
                        )

                        amt = st.number_input(
                            "Số tiền",
                            min_value=0,
                            value=inv['amount']
                        )

                        btn_col1, btn_col2 = st.columns([2, 5])

                        with btn_col1:
                            submit = st.form_submit_button(
                                "✅ Đưa vào sổ chi tiêu"
                            )

                        with btn_col2:
                            cancel = st.form_submit_button(
                                "❌ Bỏ qua"
                            )

                        # ================= SAVE =================
                        if submit:

                            st.session_state['expenses'].append({
                                "description": desc,
                                "purchase_date": date.strftime('%Y-%m-%d'),
                                "total_amount": amt,
                                "category": cat
                            })

                            st.session_state['processed_hashes'].append(
                                inv['file_hash']
                            )

                            st.session_state['pending_invoices'].pop(idx)

                            st.success("Đã thêm vào sổ chi tiêu!")

                            st.rerun()

                        # ================= CANCEL =================
                        if cancel:

                            st.session_state['pending_invoices'].pop(idx)

                            st.warning("Đã bỏ qua hóa đơn.")

                            st.rerun()

    # =========================================================
    # MANUAL INPUT
    # =========================================================
    with tab2:

        with st.form("manual_form"):

            desc = st.text_input("Nội dung")

            date = st.date_input("Ngày")

            cat = st.selectbox(
                "Danh mục",
                [
                    "Ăn uống",
                    "Mua sắm",
                    "Di chuyển",
                    "Tiện ích",
                    "Khác"
                ]
            )

            amt = st.number_input(
                "Số tiền",
                min_value=0
            )

            if st.form_submit_button("Lưu"):

                st.session_state['expenses'].append({
                    "description": desc,
                    "purchase_date": date.strftime("%Y-%m-%d"),
                    "total_amount": int(amt),
                    "category": cat
                })

                st.success("Đã lưu!")

                st.rerun()


# ================= REPORT =================
with col2:

    st.subheader("📊 Thống kê chi tiêu")

    st.write(st.session_state['expenses'])

    if not filtered_df.empty:

        total = filtered_df['total_amount'].sum()

        st.metric(
            "Tổng chi",
            f"{total:,.0f} VNĐ"
        )

        # ================= BUDGET =================
        if ngan_sach > 0:

            percent = min(total / ngan_sach, 1.0)

            st.progress(percent)

            st.write(
                f"Đã dùng {percent * 100:.1f}% ngân sách"
            )

        # ================= PIE CHART =================
        cat_df = (
            filtered_df
            .groupby("category")["total_amount"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            cat_df,
            values='total_amount',
            names='category',
            title="Phân bổ chi tiêu"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ================= TABLE =================
        show_df = filtered_df[
            [
                'Display_Date',
                'description',
                'category',
                'total_amount'
            ]
        ]

        st.dataframe(
            show_df,
            use_container_width=True
        )

    else:
        st.info("Chưa có dữ liệu.")