import streamlit as st
import requests
import uuid
from PIL import Image
import io

# Cấu hình trang
st.set_page_config(
    page_title="Chatbot Nhận thức UXO",
    page_icon="⚠️",
    layout="wide"
)

# Khởi tạo session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "language" not in st.session_state:
    st.session_state.language = "vi"

# API endpoint
API_URL = "http://localhost:8000"

# Giao diện sidebar
with st.sidebar:
    st.title("⚠️ Chatbot UXO")
    st.markdown("""
    Chatbot hỗ trợ nhận thức về vật nổ chưa nổ (UXO) tại Việt Nam.
    """)
    
    # Chọn ngôn ngữ
    language = st.radio(
        "Ngôn ngữ:",
        ["Tiếng Việt", "English"],
        index=0 if st.session_state.language == "vi" else 1
    )
    st.session_state.language = "vi" if language == "Tiếng Việt" else "en"
    
    # Tải lên ảnh để phân tích
    st.subheader("Phân tích ảnh")
    uploaded_image = st.file_uploader(
        "Tải lên ảnh vật nghi ngờ", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_image is not None:
        # Hiển thị ảnh
        image = Image.open(uploaded_image)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
        
        # Gửi ảnh đến API phân tích
        if st.button("Phân tích ảnh"):
            files = {"file": uploaded_image.getvalue()}
            response = requests.post(f"{API_URL}/detect-uxo/", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.warning(result["message"])
                
                if result["detections"]:
                    st.subheader("Kết quả phát hiện:")
                    for detection in result["detections"]:
                        st.write(f"- {detection['class']} (độ tin cậy: {detection['confidence']:.2f})")
                else:
                    st.info("Không phát hiện vật thể nghi ngờ nào.")
            else:
                st.error("Lỗi phân tích ảnh.")

# Giao diện chat chính
st.title("🤖 Chatbot Nhận thức UXO")
st.markdown("Hỏi tôi về bom mìn, vật nổ và an toàn UXO tại Việt Nam")

# Hiển thị lịch sử chat
for message in st.session_state.chat_history:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["content"])

# Input chat
prompt = st.chat_input("Nhập câu hỏi của bạn...")

if prompt:
    # Thêm vào lịch sử chat
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gọi API chatbot
    with st.spinner("Đang xử lý..."):
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
                
                # Thêm vào lịch sử chat
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": bot_response
                })
                
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
            
            else:
                st.error("Lỗi kết nối đến chatbot.")
        
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")

# Hiển thị thông tin hotline
st.sidebar.markdown("---")
st.sidebar.subheader("📞 Hotline khẩn cấp")
st.sidebar.info("""
**MAG Vietnam:** 0914 555 247 /0913 888 27  
**Quân đội địa phương:** 113  
**Công an:** 113  
**Cấp cứu:** 115  

Không chạm vào vật nghi ngờ và gọi ngay hotline!
""")