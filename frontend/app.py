import streamlit as st
import requests
import uuid
from PIL import Image
import io
import json
import os
import folium
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

# ==========================
# Cấu hình trang
# ==========================
st.set_page_config(
    page_title="Chatbot Nhận thức UXO",
    page_icon="⚠️",
    layout="wide"
)

# ==========================
# Local storage helpers
# ==========================
LOCAL_STORAGE_FILE = "chat_sessions.json"

def load_local_sessions():
    if os.path.exists(LOCAL_STORAGE_FILE):
        with open(LOCAL_STORAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_local_sessions(sessions):
    with open(LOCAL_STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

# ==========================
# Khởi tạo session state
# ==========================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "language" not in st.session_state:
    st.session_state.language = "vi"
if "admin_token" not in st.session_state:
    st.session_state.admin_token = None
if "chat_logs" not in st.session_state:
    st.session_state.chat_logs = []
if "last_log_count" not in st.session_state:
    st.session_state.last_log_count = 0

# Load từ local
all_sessions = load_local_sessions()
if st.session_state.session_id in all_sessions:
    st.session_state.chat_history = all_sessions[st.session_state.session_id].get("chat_history", [])

# ==========================
# API endpoint
# ==========================
API_URL = "http://localhost:8000"

# ==========================
# Helper functions
# ==========================
def get_auth_headers():
    if st.session_state.admin_token:
        return {"Authorization": f"Bearer {st.session_state.admin_token}"}
    return {}

def save_session():
    all_sessions[st.session_state.session_id] = {
        "chat_history": st.session_state.chat_history,
        "language": st.session_state.language
    }
    save_local_sessions(all_sessions)

def fetch_chat_logs(limit: int = 50):
    """Lấy chat logs từ backend và lưu vào session_state"""
    if not st.session_state.admin_token:
        return
    headers = get_auth_headers()
    try:
        response = requests.get(f"{API_URL}/admin/chatlogs?skip=0&limit={limit}", headers=headers)
        if response.status_code == 200:
            st.session_state.chat_logs = response.json()
        else:
            st.error(response.json().get("detail", "Lỗi không xác định"))
    except Exception as e:
        st.error(f"Lỗi API chatlogs: {e}")

def send_chat_message(prompt: str) -> str:
    """Gửi câu hỏi đến backend và cập nhật chat history + log"""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "message": prompt,
                "session_id": st.session_state.session_id,
                "language": st.session_state.language
            }
        )
        if response.status_code == 200:
            result = response.json()
            bot_response = result["answer"]
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            save_session()

            # Gửi log backend
            try:
                requests.post(
                    f"{API_URL}/admin/log-chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                        "response": bot_response
                    },
                    headers=get_auth_headers()
                )
            except:
                pass
            return bot_response
        elif response.status_code == 401:
            st.session_state.admin_token = None
            return "❌ Token hết hạn. Vui lòng đăng nhập lại."
        else:
            return "❌ Lỗi kết nối đến chatbot."
    except Exception as e:
        return f"❌ Lỗi API: {e}"

def switch_session(new_session_id: str):
    st.session_state.session_id = new_session_id
    st.session_state.chat_history = all_sessions.get(new_session_id, {}).get("chat_history", [])

def logout_admin():
    st.session_state.admin_token = None
    st.session_state.chat_logs = []
    st.session_state.last_log_count = 0
    st.success("✅ Đã đăng xuất")

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.title("⚠️ Chatbot UXO")
    st.markdown("Chatbot hỗ trợ nhận thức về vật nổ chưa nổ (UXO) tại Việt Nam.")

    # Multi-session → chỉ hiển thị khi admin đã đăng nhập
    if st.session_state.admin_token:
        st.subheader("🗂 Quản lý session")
        if all_sessions:
            selected = st.selectbox("Chọn session", options=list(all_sessions.keys()))
            if st.button("Chuyển session"):
                switch_session(selected)
        if st.button("Tạo session mới"):
            new_id = str(uuid.uuid4())
            switch_session(new_id)

    # Ngôn ngữ
    language = st.radio(
        "Ngôn ngữ:",
        ["Tiếng Việt", "English"],
        index=0 if st.session_state.language == "vi" else 1
    )
    st.session_state.language = "vi" if language == "Tiếng Việt" else "en"

    # Upload ảnh UXO
    st.subheader("Phân tích ảnh")
    uploaded_image = st.file_uploader("Tải lên ảnh vật nghi ngờ", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
        if st.button("Phân tích ảnh"):
            files = {"file": (uploaded_image.name, uploaded_image, uploaded_image.type)}
            try:
                response = requests.post(f"{API_URL}/detect-uxo/", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.warning(result.get("warning_message",""))
                    if result.get("detections"):
                        st.subheader("Kết quả phát hiện:")
                        for det in result["detections"]:
                            st.write(f"- {det['class']} (độ tin cậy: {det['confidence']:.2f})")
                    else:
                        st.info("Không phát hiện vật thể nghi ngờ nào.")
                else:
                    st.error("Lỗi phân tích ảnh.")
            except Exception as e:
                st.error(f"Lỗi API: {e}")

    # Admin login/logout
    st.subheader("🔑 Quản lý Admin")
    if st.session_state.admin_token:
        st.button("Đăng xuất Admin", on_click=logout_admin)
        # Hiển thị số lượng log mới
        new_count = len(st.session_state.chat_logs) - st.session_state.last_log_count
        if new_count > 0:
            st.info(f"📢 Có {new_count} log mới")
    else:
        with st.expander("Đăng nhập Admin"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Mật khẩu", type="password", key="login_password")
            if st.button("Đăng nhập", key="login_btn"):
                try:
                    response = requests.post(f"{API_URL}/admin/login", json={
                        "email": email,
                        "password": password
                    })
                    if response.status_code == 200:
                        st.session_state.admin_token = response.json()["access_token"]
                        st.success("✅ Đăng nhập thành công")
                    else:
                        st.error(response.json().get("detail", "Lỗi đăng nhập"))
                except Exception as e:
                    st.error(f"Lỗi API: {e}")

    # Hotline
    st.markdown("---")
    st.subheader("📞 Hotline khẩn cấp")
    st.info("""
**MAG Vietnam:** 0914 555 247 / 0913 888 27  
**Quân đội địa phương:** 113  
**Công an:** 113  
**Cấp cứu:** 115  

Không chạm vào vật nghi ngờ và gọi ngay hotline!
""")
# ==========================
# Báo cáo vị trí UXO
# ==========================
    st.markdown("---")
    st.subheader("📍 Báo cáo vị trí UXO")

    m = folium.Map(location=[16.8, 107.1], zoom_start=6)
    m.add_child(folium.LatLngPopup())
    output = st_folium(m, width=300, height=200)

    if output["last_clicked"]:
        lat = output["last_clicked"]["lat"]
        lon = output["last_clicked"]["lng"]
        st.info(f"📍 Vị trí chọn: {lat}, {lon}")
        desc = st.text_area("Mô tả thêm", key="uxo_desc")
        if st.button("🚨 Gửi báo cáo", key="send_uxo_report"):
            try:
                response = requests.post(
                    f"{API_URL}/admin/report-uxo",
                    json={"latitude": lat, "longitude": lon, "description": desc},
                    headers=get_auth_headers()
                )
                if response.status_code == 200:
                    st.success("✅ Đã gửi báo cáo UXO thành công!")
                else:
                    st.error(response.json().get("detail", "❌ Lỗi gửi báo cáo"))
            except Exception as e:
                st.error(f"❌ Lỗi API: {e}")
# ==========================
# Main Page Chat UXO
# ==========================
st.title("🤖 Chatbot Nhận thức UXO")
st.markdown("Hỏi tôi về bom mìn, vật nổ và an toàn UXO tại Việt Nam")

# Hiển thị lịch sử chat
for message in st.session_state.chat_history:
    with st.chat_message("user" if message["role"]=="user" else "assistant"):
        st.markdown(message["content"])

# Nhập câu hỏi
prompt = st.chat_input("Nhập câu hỏi của bạn...")
if prompt:
    st.session_state.chat_history.append({"role":"user","content":prompt})
    save_session()
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            bot_response = send_chat_message(prompt)
            st.markdown(bot_response)

# ==========================
# Chat logs admin (main page) với highlight
# ==========================
if st.session_state.admin_token:
    # Auto-refresh mỗi 5 giây
    st_autorefresh(interval=5000, key="autorefresh_logs")
    fetch_chat_logs()
    logs = st.session_state.chat_logs
    st.subheader("📄 Chat Logs (Admin)")
    if logs:
        new_logs_start = st.session_state.last_log_count
        st.session_state.last_log_count = len(logs)
        with st.expander("Xem log", expanded=True):
            for idx, log in enumerate(reversed(logs)):
                log_time = log.get('created_at','?')
                session_id = log.get('session_id','?')
                message = log.get('message','?')
                response = log.get('response','?')
                # Highlight log mới
                if idx < len(logs) - new_logs_start:
                    st.markdown(
                        f"<div style='background-color: #fff3b0; padding:5px; border-radius:5px;'>"
                        f"[{log_time}] `{session_id}`: {message} → **{response}**"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"[{log_time}] `{session_id}`: {message} → **{response}**"
                    )
    else:
        st.info("Chưa có log chat nào.")

    # ==========================
    # Xem báo cáo UXO (Admin)
    # ==========================
    st.subheader("📍 Báo cáo UXO (Admin)")
    try:
        response = requests.get(f"{API_URL}/admin/uxo-reports", headers=get_auth_headers())
        if response.status_code == 200:
            reports = response.json()
            if reports:
                m_admin = folium.Map(location=[16.8, 107.1], zoom_start=6)
                for r in reports:
                    folium.Marker(
                        location=[r["latitude"], r["longitude"]],
                        popup=f"📍 ID: {r['id']}<br>{r.get('description','(không có mô tả)')}",
                        icon=folium.Icon(color="red", icon="exclamation-sign")
                    ).add_to(m_admin)
                st_folium(m_admin, width=700, height=400)
            else:
                st.info("✅ Chưa có báo cáo UXO nào")
        else:
            st.error(response.json().get("detail", "❌ Lỗi tải báo cáo UXO"))
    except Exception as e:
        st.error(f"❌ Lỗi API báo cáo UXO: {e}")

