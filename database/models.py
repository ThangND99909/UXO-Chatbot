from sqlalchemy import Column, Integer, String, Text, JSON, Float, DateTime, ForeignKey, LargeBinary
from sqlalchemy.sql import func
from .connection import Base
from datetime import datetime

# ============================
# BẢNG NGƯỜI DÙNG (USERS)
# ============================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<User id={self.id}>"

# ============================
# BẢNG QUẢN TRỊ VIÊN (ADMINS)
# ============================
class Admin(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String, unique=True, index=True, nullable=False)
    # Mật khẩu đã được băm (hashed)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Admin id={self.id} email={self.email}>"

# ============================
# BẢNG LỊCH SỬ CHAT (CHAT LOGS)
# ============================
class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    # Liên kết với bảng users (mỗi log thuộc về 1 người dùng)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    # session_id giúp theo dõi hội thoại (mỗi lần trò chuyện có thể sinh session_id khác)
    session_id = Column(String, index=True, nullable=False)  # Thêm session_id để tracking
    # Tin nhắn của người dùng
    message = Column(Text, nullable=False)
    # Phản hồi của chatbot
    response = Column(Text, nullable=False)
    # Ý định (intent) được mô hình NLU phát hiện
    intent = Column(String, nullable=True)
    # Thực thể (entities) được trích xuất từ câu hỏi
    entities = Column(JSON, nullable=True, default=dict)
    # Độ tin cậy của dự đoán intent (0 - 1)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ChatLog id={self.id} session={self.session_id}>"

# ============================
# BẢNG LỊCH SỬ HỎI ĐÁP QA (QA LOGS)
# ============================
class QALog(Base):
    __tablename__ = "qa_logs"
    id = Column(Integer, primary_key=True, index=True)
    # Liên kết với session_id hội thoại
    session_id = Column(String, index=True, nullable=False)
    # Câu hỏi gốc của người dùng
    question = Column(Text, nullable=False)
    # Câu trả lời từ chatbot hoặc RAG
    answer = Column(Text, nullable=False)
    # Dữ liệu NLU (intent, entities, etc.)
    nlu = Column(JSON, nullable=True, default=dict)
    # Số lượng câu trong bộ nhớ hội thoại (Conversation Memory Window)
    memory_length = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<QALog id={self.id} session={self.session_id}>"


# ============================
# CƠ SỞ KIẾN THỨC UXO (UXO KNOWLEDGE BASE)
# ============================
class UXOKnowledge(Base):
    __tablename__ = "uxo_knowledge_base"
    id = Column(Integer, primary_key=True, index=True)
    # Tên loại vật nổ (VD: Bom bi, Mìn sát thương, ...)
    name = Column(String, index=True, nullable=False)
    # Mô tả chi tiết về loại vật nổ          
    description = Column(Text, nullable=False)  
    # Mức độ nguy hiểm (VD: thấp, trung bình, cao)               
    danger_level = Column(String, nullable=False) 
    # Quy trình xử lý an toàn             
    handling_procedure = Column(Text, nullable=False)  
    # Số hotline liên hệ khẩn cấp (nếu có)        
    hotline = Column(String, nullable=True)                    

    def __repr__(self):
        return f"<UXOKnowledge id={self.id} name={self.name}>"
    
# ============================
# BẢNG BÁO CÁO UXO (UXO REPORTS)
# ============================
class UXOReport(Base):
    __tablename__ = "uxo_reports"
    id = Column(Integer, primary_key=True, index=True)
    # Tọa độ vị trí nghi có vật nổ
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    # Mô tả do người dùng gửi lên (VD: phát hiện vật thể lạ gần bãi đất trống)
    description = Column(String, nullable=True)
    # Trạng thái xử lý (pending, reviewed, resolved)
    status = Column(String, default="pending")  
    # Người dùng đã gửi báo cáo (nếu có)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# ============================
# BẢNG KẾT QUẢ PHÁT HIỆN ẢNH UXO (UXO DETECTIONS)
# ============================
class UXODetection(Base):
    __tablename__ = "uxo_detections"
    id = Column(Integer, primary_key=True, index=True)
    # Liên kết đến bảng báo cáo UXO_REPORTS
    report_id = Column(Integer, ForeignKey("uxo_reports.id", ondelete="CASCADE"), nullable=True)
    # Tên file ảnh gốc do người dùng tải lên
    filename = Column(String, nullable=True) # Tên file ảnh
    # Liên kết session hội thoại hoặc phiên phát hiện
    session_id = Column(String, nullable=True)
    # Kết quả phát hiện vật thể (dạng JSON, gồm label, tọa độ bbox, confidence)
    detected_objects = Column(JSON, nullable=True) 
    # Ảnh gốc lưu dạng nhị phân (binary)
    image_data = Column(LargeBinary, nullable=True)  # Lưu binary image
    # URL ảnh đã được lưu (ví dụ: static/uploads/uxo123.jpg)
    image_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<UXODetection id={self.id} report_id={self.report_id}>"

# ============================
# BẢNG LOG CẢNH BÁO PHÁT HIỆN ẢNH (IMAGE DETECTION LOGS)
# ============================
class ImageDetectionLog(Base):
    __tablename__ = "image_detection_logs"
    id = Column(Integer, primary_key=True, index=True)
    # Liên kết với kết quả phát hiện ảnh
    detection_id = Column(Integer, ForeignKey("uxo_detections.id", ondelete="CASCADE"), nullable=False)
    # Session của người dùng (để truy xuất nhanh)
    session_id = Column(String, index=True, nullable=False)
    # Thông báo cảnh báo (VD: “Vật thể nghi là bom bi, không lại gần”)
    warning_message = Column(Text, nullable=True)
    # Mức độ tin cậy của phát hiện (0 - 1)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ImageDetectionLog id={self.id} session={self.detection_id}>"