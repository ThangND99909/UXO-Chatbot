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
import logging
from datetime import datetime

# ==============================
# Logging configuration
# ==============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # ✅ THÊM LOGGER

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
    "detection_history": {"vi": "📸 Lịch sử phát hiện ảnh", "en": "📸 Detection History"},
    "view_detected_image": {"vi": "Xem ảnh đã phát hiện", "en": "View detected image"},
    "no_detection_history": {"vi": "Chưa có lịch sử phát hiện nào.", "en": "No detection history yet."},
    "upload_for_detection": {"vi": "📤 Upload ảnh để phát hiện UXO", "en": "📤 Upload image for UXO detection"},

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
    layout="wide",
    initial_sidebar_state="expanded"
)
# CSS cho sidebar rộng
st.markdown("""
<style>
    /* Sidebar chính */
    section[data-testid="stSidebar"] {
        min-width: 450px !important;
        max-width: 500px !important;
        background-color: #f8f9fa;
    }
    
    /* Nội dung sidebar */
    .css-1d391kg, .css-1lcbmhc {
        width: 450px !important;
        padding: 1rem;
    }
    
    /* Cải thiện scroll */
    .sidebar .sidebar-content {
        overflow-y: auto;
        height: 100vh;
        padding-bottom: 2rem;
    }
    
    /* Điều chỉnh bản đồ */
    .sidebar .folium-map {
        width: 100% !important;
        height: 280px !important;
    }
    
    /* Form elements */
    .sidebar .stTextInput input,
    .sidebar .stTextArea textarea {
        width: 100% !important;
    }
    
    /* Images trong sidebar */
    .sidebar .stImage {
        max-width: 100% !important;
        border-radius: 10px;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            min-width: 85vw !important;
            max-width: 85vw !important;
        }
    }
</style>
""", unsafe_allow_html=True)
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
if "detection_reports" not in st.session_state:
    st.session_state.detection_reports = []
if "selected_detection_image" not in st.session_state:
    st.session_state.selected_detection_image = None
if "admin_detection_history" not in st.session_state:
    st.session_state.admin_detection_history = []
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}

if "show_report_map" not in st.session_state:
    st.session_state.show_report_map = False
if "last_intent" not in st.session_state:  
    st.session_state.last_intent = None
# Load từ local
all_sessions = load_local_sessions()
if st.session_state.session_id in all_sessions:
    data = all_sessions[st.session_state.session_id]
    st.session_state.chat_history = data.get("chat_history", [])
    st.session_state.show_report_map = data.get("show_report_map", False)
    st.session_state.last_intent = data.get("last_intent")
    st.session_state.analysis_done = data.get("analysis_done", False)
    st.session_state.detection_result = data.get("detection_result")

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
        "language": st.session_state.language,
        "show_report_map": st.session_state.get("show_report_map", False),
        "last_intent": st.session_state.get("last_intent"),
        "analysis_done": st.session_state.get("analysis_done", False),
        "detection_result": st.session_state.get("detection_result"),
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

def submit_detection(detection_data: dict):
    """
    chỉ log thông tin detection
    Detection thực tế đã được lưu trong backend
    """
    try:
        detection_id = detection_data.get("detection_id")
        filename = detection_data.get("filename")
        
        if detection_id:
            logger.info(f"✅ Detection saved - ID: {detection_id}, File: {filename}")
            return {"id": detection_id, "status": "logged"}
        else:
            logger.warning("⚠️ No detection ID in response")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error logging detection: {e}")
        return None


def get_detection_image(report_id: str):
    """Lấy ảnh detection từ API"""
    if not st.session_state.admin_token:
        return None
    
    headers = get_auth_headers()
    try:
        response = requests.get(f"{API_URL}/admin/detections/{report_id}", headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Lỗi khi lấy ảnh detection: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Lỗi API detection image: {e}")
        return None

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
            # ✅ Nếu intent là report_bomb thì mở bản đồ
            if "intent" in result:
                st.session_state.last_intent = result["intent"]
                if result["intent"] == "report_bomb":
                    st.session_state.show_report_map = True
                    st.write("DEBUG last_intent:", st.session_state.last_intent)
                    st.write("DEBUG show_report_map:", st.session_state.show_report_map)
                
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
    st.session_state.detection_reports = []
    st.session_state.selected_detection_image = None
    st.session_state.admin_detection_history = []
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
        
        if uploaded_image.name not in st.session_state.processed_images:
            # ✅ Sử dụng form để tránh rerun toàn bộ
            with st.form(key="image_analysis_form"):
                if st.form_submit_button(UI_TEXT["analyze_image"][st.session_state.language]):
                    try:
                        if uploaded_image is None:
                            st.error("❌ Lỗi: Không có ảnh để xử lý")
                        else:
                            uploaded_image.seek(0)
                            image_bytes = uploaded_image.getvalue()
                            
                            files = {"file": (uploaded_image.name, image_bytes, uploaded_image.type)}
                            data = {
                                "session_id": st.session_state.session_id,  # ⬅️ THÊM DÒNG NÀY
                                "confidence_threshold": 0.3
                            }
                            with st.spinner("Đang phân tích ảnh..."):
                                response = requests.post(f"{API_URL}/admin/detect-uxo/", files=files, data=data)
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.session_state.processed_images[uploaded_image.name] = True
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
                                
                                #Lưu detection vào database
                                try:
                                    if "detection_id" in result:
                                        # Thêm vào local history
                                        st.session_state.admin_detection_history.append({
                                            "id": result["detection_id"],
                                            "filename": uploaded_image.name,
                                            "detections": result.get("detections", []),
                                            "created_at": datetime.now().isoformat(),
                                            "session_id": st.session_state.session_id
                                        })
                                        st.success("✅ Đã lưu detection vào database")
                                    else:
                                        st.warning("⚠️ Không thể lấy detection ID từ response")
                                        
                                except Exception as save_error:
                                    st.error(f"❌ Lỗi lưu history: {save_error}")
                            
                            else:
                                st.error(f"Lỗi phân tích ảnh. Status code: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Lỗi API: {e}")
        else:
            st.info("Ảnh đã được phân tích trước đó trong phiên này.")
            

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
        # Đã đăng nhập
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚪 " + UI_TEXT["admin_logout"][st.session_state.language], use_container_width=True):
                logout_admin()
                st.rerun()
        with col2:
            new_count = len(st.session_state.chat_logs) - st.session_state.last_log_count
            if new_count > 0:
                st.info(f"📢 {new_count} mới")
        
    else:
        # Chưa đăng nhập
        login_col1, login_col2 = st.columns([3, 1])
        
        with login_col1:
            login_icon = "🔓" if not st.session_state.get('show_login_form', False) else "🔒"
            if st.button(f"{login_icon} {UI_TEXT['admin_login'][st.session_state.language]}", 
                        use_container_width=True, key="login_toggle_btn"):
                st.session_state.show_login_form = not st.session_state.get('show_login_form', False)
                st.rerun()
        
        with login_col2:
            if st.session_state.get('show_login_form', False):
                if st.button("❌", help="Đóng form đăng nhập"):
                    st.session_state.show_login_form = False
                    st.rerun()
        
        # Hiển thị form đăng nhập khi được toggle
        if st.session_state.get('show_login_form', False):
            with st.expander("🔐 **Form Đăng Nhập**", expanded=True):
                with st.form(key="admin_login_form", clear_on_submit=False):
                    email = st.text_input("📧 Email", placeholder="admin@example.com", key="login_email")
                    password = st.text_input("🔑 Mật khẩu", type="password", 
                                        placeholder="Nhập mật khẩu...", key="login_password")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        login_clicked = st.form_submit_button("Đăng nhập", use_container_width=True)
                    with col2:
                        if st.form_submit_button("Hủy", use_container_width=True):
                            st.session_state.show_login_form = False
                            st.rerun()
                    
                    if login_clicked:
                        if not email or not password:
                            st.error("⚠️ Vui lòng nhập đầy đủ email và mật khẩu")
                        else:
                            try:
                                with st.spinner("🔄 Đang đăng nhập..."):
                                    response = requests.post(f"{API_URL}/admin/login", 
                                                        json={"email": email, "password": password})
                                    if response.status_code == 200:
                                        st.session_state.admin_token = response.json()["access_token"]
                                        st.session_state.login_password_value = password
                                        st.session_state.show_login_form = False
                                        st.success("✅ Đăng nhập thành công!")
                                        st.rerun()
                                    else:
                                        error_msg = parse_api_error_friendly(response.json())
                                        st.error(f"❌ {error_msg}")
                            except Exception as e:
                                st.error(f"❌ Lỗi kết nối: {e}")

    
    # Báo cáo vị trí UXO
    st.markdown("---")
    st.subheader(UI_TEXT["report_uxo"][st.session_state.language])
    m = folium.Map(location=[16.8, 107.1], zoom_start=6)
    m.add_child(folium.LatLngPopup())
    output = st_folium(m, width=300, height=200, key="sidebar_uxo_map")

    if output["last_clicked"]:
        lat = output["last_clicked"]["lat"]
        lon = output["last_clicked"]["lng"]
        st.info(f"📍 Vị trí chọn: {lat}, {lon}")
        desc = st.text_area(UI_TEXT["description"][st.session_state.language], key="sidebar_uxo_desc")
        if st.button(UI_TEXT["send_report"][st.session_state.language], key="sidebar_send_uxo_report"):
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
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
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
    "Upload image",
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
if uploaded_chat_image and uploaded_chat_image.name not in st.session_state.processed_images:
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
            data = {
                "session_id": st.session_state.session_id,
                "confidence_threshold": 0.3
            }
            response = requests.post(f"{API_URL}/admin/detect-uxo/", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.processed_images[uploaded_chat_image.name] = True
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
               
                
                try:
                    if "detection_id" in result:
                        # Thêm vào local history
                        st.session_state.admin_detection_history.append({
                            "id": result["detection_id"],
                            "filename": uploaded_chat_image.name,
                            "detections": result.get("detections", []),
                            "created_at": datetime.now().isoformat(),
                            "session_id": st.session_state.session_id
                        })
                        st.success("✅ Đã lưu detection vào database")
                    else:
                        st.warning("⚠️ Không thể lấy detection ID từ response")
                        
                except Exception as save_error:
                    st.error(f"❌ Lỗi lưu database: {save_error}")
            
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
# HIỂN THỊ MAP BÁO CÁO UXO 
# ==============================
if st.session_state.get("show_report_map", False):
    st.markdown("---")
    st.subheader("📍 BÁO CÁO VỊ TRÍ UXO KHẨN CẤP")
    st.warning("⚠️ **CẢNH BÁO: KHÔNG TỰ Ý XỬ LÝ!**")
    st.info("Vui lòng click trên bản đồ để xác định vị trí chính xác nơi bạn phát hiện vật nghi ngờ:")
    
    # Tạo bản đồ
    m = folium.Map(location=[16.4637, 107.5909], zoom_start=7)
    m.add_child(folium.LatLngPopup())
    
    # Hiển thị bản đồ
    map_data = st_folium(m, width=700, height=400, key="main_uxo_report_map")
    
    # Hiển thị debug info
    if st.session_state.get("last_intent"):
        st.write(f"DEBUG: Last intent: {st.session_state.last_intent}")
    
    # Xử lý khi user click trên bản đồ
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]
        
        st.success(f"✅ **Đã chọn vị trí:** {lat:.6f}, {lng:.6f}")
        
        # Mô tả thêm
        desc = st.text_area("💬 **Mô tả thêm về hiện trường:**", 
                          placeholder="Ví dụ: Quả bom nằm trong ruộng, kích thước khoảng 50cm, màu nâu...",
                          key="main_uxo_description")
        
        # Nút gửi báo cáo
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚨 GỬI BÁO CÁO", type="primary", use_container_width=True):
                try:
                    headers = get_auth_headers()
                    response = requests.post(
                        f"{API_URL}/admin/report-uxo",
                        json={"latitude": lat, "longitude": lng, "description": desc},
                        headers=headers
                    )
                    if response.status_code == 200:
                        st.success("✅ **Báo cáo đã được gửi thành công!**")
                        st.info("Cơ quan chức năng sẽ liên hệ ngay. Vui lòng giữ khoảng cách an toàn.")
                        st.session_state.show_report_map = False
                    else:
                        error_msg = response.json().get("detail", "❌ Lỗi khi gửi báo cáo")
                        st.error(f"{error_msg}")
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối: {e}")
        
        with col2:
            if st.button("🗺️ Xem lại bản đồ", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("❌ Hủy báo cáo", use_container_width=True):
                st.session_state.show_report_map = False
                st.rerun()
    
    # Hotline khẩn cấp
    st.markdown("---")
    st.error("📞 **HOTLINE KHẨN CẤP:**")
    st.info(UI_TEXT["hotline"][st.session_state.language])
 
# ==============================
# Chat logs admin (main page) với highlight
# ==============================
if st.session_state.admin_token:
    #st_autorefresh(interval=60000, key="autorefresh_logs")
    #fetch_chat_logs()
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh Logs", key="manual_refresh"):
            fetch_chat_logs()
    with col2:
        st.write("")  # Spacer
    
    fetch_chat_logs()  # Vẫn load lần đầu
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

    # ==============================
    # LỊCH SỬ PHÁT HIỆN ẢNH (Admin)
    # ==============================
  
    st.subheader(UI_TEXT["detection_history"][st.session_state.language])

    def fetch_all_detections():
        """Lấy tất cả detection reports từ database"""
        if st.session_state.admin_token:
            headers = get_auth_headers()
            try:
                response = requests.get(f"{API_URL}/admin/all-detections", headers=headers)
                if response.status_code == 200:
                    return response.json()
                else:
                    st.error(f"❌ Lỗi API: {response.status_code}")
                    return []
            except Exception as e:
                st.error(f"❌ Lỗi kết nối: {e}")
                return []
        return []

    # Nút refresh để tải danh sách mới nhất
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Refresh từ Database", use_container_width=True):
            st.session_state.detection_reports = fetch_all_detections()
            if st.session_state.detection_reports:
                st.success(f"✅ Đã tải {len(st.session_state.detection_reports)} detection reports từ database")

    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state.detection_reports = []
            st.rerun()

    # Tự động load từ database khi vào trang
    if not st.session_state.detection_reports:
        st.session_state.detection_reports = fetch_all_detections()

    # Hiển thị thống kê ĐÚNG từ database
    if st.session_state.detection_reports:
        total_reports = len(st.session_state.detection_reports)
        
        # ✅ SỬA: Đếm objects từ database thực tế
        total_objects = 0
        reports_with_objects = 0
        
        for report in st.session_state.detection_reports:
            if report.get('detected_objects') and len(report['detected_objects']) > 0:
                reports_with_objects += 1
                total_objects += len(report['detected_objects'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Reports", total_reports)
        with col2:
            st.metric("🎯 Objects Found", total_objects)
        with col3:
            st.metric("✅ Positive Cases", reports_with_objects)
        with col4:
            st.metric("📭 Empty Cases", total_reports - reports_with_objects)

    # Hiển thị danh sách detection reports từ database
    detection_reports = st.session_state.detection_reports

    if detection_reports:
        st.subheader(f"📋 Danh sách Detection Reports (Từ Database)")
        
        # Phân trang
        items_per_page = 10
        total_pages = max(1, (len(detection_reports) + items_per_page - 1) // items_per_page)
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            page_number = st.number_input("Trang:", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page_number - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(detection_reports))
        current_page_reports = detection_reports[start_idx:end_idx]
        
        for report in current_page_reports:
            # Định dạng thời gian
            uploaded_time = report.get('created_at', 'Unknown')
            if uploaded_time and uploaded_time != 'Unknown':
                try:
                    if 'T' in uploaded_time:
                        uploaded_time = uploaded_time.replace('T', ' ').split('.')[0]
                    elif isinstance(uploaded_time, str) and len(uploaded_time) > 16:
                        uploaded_time = uploaded_time[:19].replace('T', ' ')
                except:
                    uploaded_time = "Invalid date"
            else:
                uploaded_time = "Chưa có thời gian"
            
            with st.expander(f"📸 {report.get('filename', 'Unknown')} - {uploaded_time}", expanded=False):
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.write(f"**📁 File:** {report.get('filename', 'N/A')}")
                    st.write(f"**🆔 ID:** {report['id']}")
                    st.write(f"**📅 Uploaded:** {uploaded_time}")
                    st.write(f"**👤 Session:** `{report.get('session_id', 'N/A')}`")
                    
                    # ✅ SỬA: Hiển thị detected objects từ database
                    if 'detected_objects' in report and report['detected_objects']:
                        st.write("**🎯 Detected Objects:**")
                        for i, det in enumerate(report['detected_objects']):
                            class_name = det.get('class', 'Unknown')
                            confidence = det.get('confidence', 0)
                            bbox = det.get('bbox', [])
                            
                            col_det1, col_det2 = st.columns([2, 3])
                            with col_det1:
                                st.write(f"- **{class_name}**")
                            with col_det2:
                                st.write(f"({confidence:.2%})")
                            
                            if bbox and len(bbox) == 4:
                                st.write(f"  📍 BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                    else:
                        st.info("📭 No objects detected")
                
                with col2:
                    # Nút xem ảnh từ database
                    if st.button(f"👁️ View Image with BBox", key=f"view_bbox_{report['id']}", use_container_width=True):
                        with st.spinner("Đang tải và vẽ bounding boxes..."):
                            try:
                                headers = get_auth_headers()
                                response = requests.get(f"{API_URL}/admin/detections/{report['id']}", headers=headers)
                                
                                if response.status_code == 200:
                                    image_bytes = response.content
                                    image = Image.open(io.BytesIO(image_bytes))
                                    
                                    # ✅ VẼ BOUNDING BOXES
                                    if 'detected_objects' in report and report['detected_objects']:
                                        draw = ImageDraw.Draw(image)
                                        
                                        for det in report['detected_objects']:
                                            bbox = det.get('bbox', [])
                                            if len(bbox) == 4:
                                                x1, y1, x2, y2 = bbox
                                                
                                                # Vẽ rectangle
                                                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                                                
                                                # Vẽ label
                                                class_name = det.get('class', 'Unknown')
                                                confidence = det.get('confidence', 0)
                                                label = f"{class_name}: {confidence:.2f}"
                                                
                                                # Tính kích thước text
                                                text_bbox = draw.textbbox((x1, y1), label)
                                                text_width = text_bbox[2] - text_bbox[0]
                                                text_height = text_bbox[3] - text_bbox[1]
                                                
                                                # Vẽ background cho text
                                                draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill="red")
                                                
                                                # Vẽ text
                                                draw.text((x1 + 2, y1 - text_height - 3), label, fill="white")
                                    
                                    st.image(image, 
                                            caption=f"Image: {report.get('filename', 'Unknown')} (with BBox)", 
                                            use_container_width=True)
                                    
                                    st.success("✅ Đã tải ảnh và vẽ bounding boxes")
                                    st.write(f"📏 Kích thước: {image.size}")
                                    st.write(f"🎯 Objects: {len(report['detected_objects']) if report.get('detected_objects') else 0}")
                                    
                                else:
                                    st.error(f"❌ Lỗi API: {response.status_code}")
                                    
                            except Exception as e:
                                st.error(f"❌ Lỗi tải ảnh: {e}")
                    
                    # Hiển thị thông tin nhanh từ database thực tế
                    if report.get('detected_objects'):
                        detections_count = len(report['detected_objects'])
                        st.metric("Objects Detected", detections_count)
                    else:
                        st.metric("Objects Detected", 0)
        
        # Hiển thị thông tin phân trang
        st.write(f"**Hiển thị {start_idx + 1}-{end_idx} của {len(detection_reports)} reports**")
        
    else:
        st.info(UI_TEXT["no_detection_history"][st.session_state.language])
        
        # Hướng dẫn sử dụng
        st.markdown("---")
        st.info("""
        **📖 Hướng dẫn sử dụng Detection History:**
        
        - **Refresh**: Tải lại danh sách mới nhất từ database
        - **Tìm kiếm**: Tìm report theo tên file
        - **Lọc**: Lọc theo kết quả detection
        - **View Image**: Xem ảnh gốc từ database
        - **Phân trang**: Duyệt qua các trang khi có nhiều reports
        
        **💡 Tính năng:**
        - Hiển thị tất cả detection reports từ tất cả users
        - Thống kê tổng quan về số lượng reports và objects
        - Tìm kiếm và lọc nâng cao
        - Xem ảnh gốc từ database
        """)
