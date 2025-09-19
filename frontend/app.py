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
from PIL import ImageDraw, ImageFont

# ==============================
# Dictionary giao diện song ngữ
# ==============================
UI_TEXT = {
    "title": {"vi": "🤖 Chatbot Nhận thức UXO", "en": "🤖 UXO Awareness Chatbot"},
    "chat_placeholder": {"vi": "Nhập câu hỏi của bạn...", "en": "Type your question..."},
    "upload_image": {"vi": "Tải lên ảnh vật nghi ngờ", "en": "Upload suspected object image"},
    "analyze_image": {"vi": "Phân tích ảnh", "en": "Analyze image"},
    "admin_manage": {"vi": "Quản lý Admin", "en": "Admin Management"},
    "admin_login": {"vi": "Đăng nhập Admin", "en": "Admin Login"},
    "admin_logout": {"vi": "Đăng xuất Admin", "en": "Admin Logout"},
    "hotline emergency": {"vi": "Hotline khẩn cấp", "en": "Emergency Hotline"},
    "hotline": {"vi": """
**MAG Vietnam:** 0914 555 247 / 0913 888 27  
**Quân đội địa phương:** 113  
**Công an:** 113  
**Cấp cứu:** 115  

Không chạm vào vật nghi ngờ và gọi ngay hotline!
""",
"en": """
**MAG Vietnam:** 0914 555 247 / 0913 888 27  
**Local Army:** 113  
**Police:** 113  
**Ambulance:** 115  

Do not touch the suspected object and call the hotline immediately!
"""},
    "report_uxo": {"vi": "📍 Báo cáo vị trí UXO", "en": "📍 Report UXO location"},
    "send_report": {"vi": "🚨 Gửi báo cáo", "en": "🚨 Send report"},
    "description": {"vi": "Mô tả thêm", "en": "Additional description"},
    "image_result": {"vi": "Kết quả phát hiện:", "en": "Detection results:"},
    "no_detection": {"vi": "Không phát hiện vật thể nghi ngờ nào.", "en": "No suspected objects detected."},
    "no_chat_logs": {"vi": "Chưa có log chat nào.", "en": "No chat logs yet."},
    "no_uxo_reports": {"vi": "✅ Chưa có báo cáo UXO nào", "en": "✅ No UXO reports yet"},
    "no_description": {"vi": "(không có mô tả)", "en": "(No description)"},

    "sidebar_description": {
        "vi": "Chatbot hỗ trợ nhận thức về vật nổ chưa nổ (UXO) tại Việt Nam.",
        "en": "Chatbot supports awareness of unexploded ordnance (UXO) in Vietnam."
    },
    "language_label": {"vi": "Ngôn ngữ:", "en": "Language:"},
    "main_page_intro": {
        "vi": "Hỏi tôi về bom mìn, vật nổ và an toàn UXO tại Việt Nam",
        "en": "Ask me about mines, explosives, and UXO safety in Vietnam"
    }
}

# ==============================
# Hien thi loi
# ==============================
def parse_api_error_friendly(response_json):
    if "detail" not in response_json:
        return "Có lỗi không xác định. Vui lòng thử lại."
    detail = response_json["detail"]
    if isinstance(detail, list):
        msgs = []
        for err in detail:
            loc = err.get("loc", [])
            msg = err.get("msg", "")
            if loc and loc[-1] == "email":
                msgs.append("Email không hợp lệ. Vui lòng nhập đúng định dạng.")
            elif loc and loc[-1] == "password":
                msgs.append("Mật khẩu không hợp lệ.")
            else:
                msgs.append(msg)
        return "\n".join(msgs)
    if isinstance(detail, str):
        return detail
    return "Có lỗi không xác định. Vui lòng thử lại."

# ==============================
# Cấu hình trang
# ==============================
st.set_page_config(
    page_title="Chatbot Nhận thức UXO",
    page_icon="⚠️",
    layout="wide"
)

# ==============================
# Local storage helpers
# ==============================
LOCAL_STORAGE_FILE = "chat_sessions.json"

def load_local_sessions():
    if os.path.exists(LOCAL_STORAGE_FILE):
        with open(LOCAL_STORAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_local_sessions(sessions):
    with open(LOCAL_STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

# ==============================
# Khởi tạo session state
# ==============================
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
if "login_password_value" not in st.session_state:
    st.session_state.login_password_value = ""

# Load từ local
all_sessions = load_local_sessions()
if st.session_state.session_id in all_sessions:
    st.session_state.chat_history = all_sessions[st.session_state.session_id].get("chat_history", [])

# ==============================
# API endpoint
# ==============================
API_URL = "http://localhost:8000"

# ==============================
# Helper functions
# ==============================
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
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"message": prompt, "session_id": st.session_state.session_id, "language": st.session_state.language}
        )
        if response.status_code == 200:
            result = response.json()
            bot_response = result["answer"]
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            save_session()
            try:
                requests.post(
                    f"{API_URL}/admin/log-chat",
                    json={"session_id": st.session_state.session_id, "message": prompt, "response": bot_response},
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
    st.success(UI_TEXT["admin_logout"][st.session_state.language])

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.title("⚠️ Chatbot UXO")
    st.markdown(UI_TEXT["sidebar_description"][st.session_state.language])

    # Ngôn ngữ
    def set_language():
        lang = st.session_state.language_radio
        st.session_state.language = "vi" if lang == "Tiếng Việt" else "en"
    st.radio(
        UI_TEXT["language_label"][st.session_state.language],
        ["Tiếng Việt", "English"],
        index=0 if st.session_state.language == "vi" else 1,
        key="language_radio",
        on_change=set_language,
        
    )
    

    # Multi-session → chỉ hiển thị khi admin đã đăng nhập
    if st.session_state.admin_token:
        st.subheader("🗂 Quản lý session")
        if all_sessions:
            selected = st.selectbox("Chọn session", options=list(all_sessions.keys()))
            if st.button("Chuyển session"):
                switch_session(selected)
        #if st.button("Tạo session mới"):
        #    new_id = str(uuid.uuid4())
        #    switch_session(new_id)

    # Upload ảnh UXO
    st.subheader(UI_TEXT["analyze_image"][st.session_state.language])
    uploaded_image = st.file_uploader(UI_TEXT["upload_image"][st.session_state.language], type=["jpg", "jpeg", "png"])

    # ✅ Lưu trạng thái phân tích trong session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'detection_result' not in st.session_state:
        st.session_state.detection_result = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
        
        # ✅ Sử dụng form để tránh rerun toàn bộ
        with st.form(key="image_analysis_form"):
            if st.form_submit_button(UI_TEXT["analyze_image"][st.session_state.language]):
                try:
                    uploaded_image.seek(0)
                    image_bytes = uploaded_image.getvalue()
                    
                    files = {"file": (uploaded_image.name, image_bytes, uploaded_image.type)}
                    
                    with st.spinner("Đang phân tích ảnh..."):
                        response = requests.post(f"{API_URL}/admin/detect-uxo/", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # ✅ Lưu kết quả vào session state (KHÔNG dùng rerun)
                        st.session_state.detection_result = result
                        st.session_state.analysis_done = True
                        
                        # ✅ Xử lý và lưu ảnh đã vẽ bounding box
                        detected_image = image.copy()
                        draw = ImageDraw.Draw(detected_image)
                        
                        if "detections" in result and result["detections"]:
                            for det in result["detections"]:
                                bbox = det.get('bbox', [])
                                if len(bbox) == 4:
                                    x1, y1, x2, y2 = bbox
                                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                                    
                                    class_name = det.get('class', 'Unknown')
                                    confidence = det.get('confidence', 0)
                                    label = f"{class_name}: {confidence:.2f}"
                                    
                                    # Vẽ text
                                    text_bbox = draw.textbbox((x1, y1-25), label)
                                    draw.rectangle(text_bbox, fill="red")
                                    draw.text((x1, y1-25), label, fill="white")
                        
                        st.session_state.processed_image = detected_image
                        
                    else:
                        st.error(f"Lỗi phân tích ảnh. Status code: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Lỗi API: {e}")

    # ✅ HIỂN THỊ KẾT QUẢ PHÂN TÍCH ẢNH (nếu có)
    if st.session_state.analysis_done and st.session_state.detection_result:
        result = st.session_state.detection_result
        
        if "warning_message" in result:
            st.warning(result["warning_message"])
        
        if st.session_state.processed_image:
            st.image(st.session_state.processed_image, caption="Ảnh đã nhận diện", use_container_width=True)
        
        if "detections" in result and result["detections"]:
            st.write("**Chi tiết phát hiện:**")
            for det in result["detections"]:
                class_name = det.get('class', 'Unknown')
                confidence = det.get('confidence', 0)
                st.write(f"- {class_name} (độ tin cậy: {confidence:.2f})")
        else:
            st.info(UI_TEXT["no_detection"][st.session_state.language])

    # Admin login/logout
    #st.subheader("🔑 Quản lý Admin")
    st.subheader(UI_TEXT["admin_manage"][st.session_state.language])
    if st.session_state.admin_token:
        st.button(UI_TEXT["admin_logout"][st.session_state.language], on_click=logout_admin)
        new_count = len(st.session_state.chat_logs) - st.session_state.last_log_count
        if new_count > 0:
            st.info(f"📢 Có {new_count} log mới")
    else:
        with st.expander(UI_TEXT["admin_login"][st.session_state.language]):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Mật khẩu", type="password", key="login_password",
                                     value=st.session_state.login_password_value)
            if st.button(UI_TEXT["admin_login"][st.session_state.language], key="login_btn"):
                try:
                    response = requests.post(f"{API_URL}/admin/login", json={"email": email, "password": password})
                    if response.status_code == 200:
                        st.session_state.admin_token = response.json()["access_token"]
                        st.session_state.login_password_value = password
                        st.success("✅ Đăng nhập thành công")
                    else:
                        error_msg = parse_api_error_friendly(response.json())
                        st.error(f"{error_msg}")
                except Exception as e:
                    st.error(f"Lỗi API: {e}")

    
    # Báo cáo vị trí UXO
    st.markdown("---")
    st.subheader(UI_TEXT["report_uxo"][st.session_state.language])
    m = folium.Map(location=[16.8, 107.1], zoom_start=6)
    m.add_child(folium.LatLngPopup())
    output = st_folium(m, width=300, height=200)

    if output["last_clicked"]:
        lat = output["last_clicked"]["lat"]
        lon = output["last_clicked"]["lng"]
        st.info(f"📍 Vị trí chọn: {lat}, {lon}")
        desc = st.text_area(UI_TEXT["description"][st.session_state.language], key="uxo_desc")
        if st.button(UI_TEXT["send_report"][st.session_state.language], key="send_uxo_report"):
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

    # Hotline
    st.markdown("---")
    #st.subheader("📞 Hotline khẩn cấp")
    st.subheader(UI_TEXT["hotline emergency"][st.session_state.language])
    st.info(UI_TEXT["hotline"][st.session_state.language])

# ==============================
# Main Page Chat UXO
# ==============================
st.title(UI_TEXT["title"][st.session_state.language])
st.markdown(UI_TEXT["main_page_intro"][st.session_state.language])

# Hiển thị lịch sử chat
for message in st.session_state.chat_history:
    with st.chat_message("user" if message["role"]=="user" else "assistant"):
        st.markdown(message["content"])

# ✅ Tạo drag and drop area tích hợp
st.markdown("""
<style>
.upload-area {
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
    background-color: #f9f9f9;
}
.upload-area:hover {
    border-color: #666;
    background-color: #f0f0f0;
}
.upload-text {
    color: #666;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Tạo vùng drag and drop
uploaded_chat_image = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    key="chat_image_uploader",
    label_visibility="collapsed",
    help=""
)

# Hiển thị UI drag and drop custom
if uploaded_chat_image is None:
    st.markdown(f"""
    <div class="upload-area">
        <div class="upload-text">
            <strong>Drag and drop file here</strong><br>
            Limit 200MB per file • JPG, JPEG, PNG
        </div>
        <div>
            {st.session_state.language == "vi" and "Duyệt files" or "Browse files"}
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info(f"✅ Đã chọn file: {uploaded_chat_image.name}")

# ✅ Text input riêng cho chat
prompt = st.chat_input(UI_TEXT["chat_placeholder"][st.session_state.language])

# ✅ Xử lý ảnh được upload từ drag and drop
if uploaded_chat_image:
    try:
        # Hiển thị ảnh đã upload
        chat_image = Image.open(uploaded_chat_image)
        st.session_state.chat_history.append({"role": "user", "content": f"📸 Đã tải lên ảnh: {uploaded_chat_image.name}"})
        with st.chat_message("user"):
            st.image(chat_image, caption=uploaded_chat_image.name, width=200)
        
        # Tự động phân tích ảnh
        with st.spinner("Đang phân tích ảnh..."):
            uploaded_chat_image.seek(0)
            image_bytes = uploaded_chat_image.getvalue()
            files = {"file": (uploaded_chat_image.name, image_bytes, uploaded_chat_image.type)}
            
            response = requests.post(f"{API_URL}/admin/detect-uxo/", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                # Xử lý và hiển thị kết quả
                detected_image = chat_image.copy()
                draw = ImageDraw.Draw(detected_image)
                
                detection_message = "📊 **Kết quả phân tích ảnh:**\n"
                
                if "detections" in result and result["detections"]:
                    for det in result["detections"]:
                        bbox = det.get('bbox', [])
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            
                            class_name = det.get('class', 'Unknown')
                            confidence = det.get('confidence', 0)
                            label = f"{class_name}: {confidence:.2f}"
                            
                            # Vẽ text
                            text_bbox = draw.textbbox((x1, y1-25), label)
                            draw.rectangle(text_bbox, fill="red")
                            draw.text((x1, y1-25), label, fill="white")
                            
                            detection_message += f"- {class_name} (độ tin cậy: {confidence:.2f})\n"
                    
                    # Hiển thị ảnh đã được vẽ bounding box
                    st.session_state.chat_history.append({"role": "assistant", "content": detection_message})
                    with st.chat_message("assistant"):
                        st.markdown(detection_message)
                        st.image(detected_image, caption="Ảnh đã nhận diện", width=300)
                
                else:
                    detection_message += UI_TEXT["no_detection"][st.session_state.language]
                    st.session_state.chat_history.append({"role": "assistant", "content": detection_message})
                    with st.chat_message("assistant"):
                        st.markdown(detection_message)
            
            else:
                error_msg = f"❌ Lỗi phân tích ảnh: {response.status_code}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                    
    except Exception as e:
        error_msg = f"❌ Lỗi khi xử lý ảnh: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.markdown(error_msg)
    
    # Lưu session
    save_session()

# ✅ Xử lý tin nhắn text
if prompt:
    st.session_state.chat_history.append({"role":"user","content":prompt})
    save_session()
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            bot_response = send_chat_message(prompt)
            st.markdown(bot_response)

# ==============================
# Chat logs admin (main page) với highlight
# ==============================
if st.session_state.admin_token:
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
                if idx < len(logs) - new_logs_start:
                    st.markdown(
                        f"<div style='background-color: #fff3b0; padding:5px; border-radius:5px;'>"
                        f"[{log_time}] `{session_id}`: {message} → **{response}**"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"[{log_time}] `{session_id}`: {message} → **{response}**")
    else:
        st.info(UI_TEXT["no_chat_logs"][st.session_state.language])

    # ==============================
    # Xem báo cáo UXO (Admin)
    # ==============================
    st.subheader(UI_TEXT["report_uxo"][st.session_state.language] + " (Admin)")
    try:
        response = requests.get(f"{API_URL}/admin/uxo-reports", headers=get_auth_headers())
        if response.status_code == 200:
            reports = response.json()
            if reports:
                m_admin = folium.Map(location=[16.8, 107.1], zoom_start=6)
                for r in reports:
                    folium.Marker(
                        location=[r["latitude"], r["longitude"]],
                        popup=f"📍 ID: {r['id']}<br>{r.get('description', UI_TEXT['no_description'][st.session_state.language])}",
                        icon=folium.Icon(color="red", icon="exclamation-sign")
                    ).add_to(m_admin)
                st_folium(m_admin, width=700, height=400)
            else:
                st.info(UI_TEXT["no_uxo_reports"][st.session_state.language])
        else:
            st.error(response.json().get("detail", "❌ Lỗi tải báo cáo UXO"))
    except Exception as e:
        st.error(f"❌ Lỗi API báo cáo UXO: {e}")
