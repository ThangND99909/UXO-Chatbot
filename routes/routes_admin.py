from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from typing import Optional
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import json

# Import model, CRUD, DB connection
from database import models, crud, connection
# Import chức năng xác thực admin
from utils.auth import create_access_token, get_current_admin
# Import schema cho request/response
from app.schemas import AdminLoginRequest, AdminLoginResponse, UXOReportCreate, UXOReportResponse, UXODetectionResponse, UXODetectionCreate

# import YOLO detector
from computer_vision.yolov8_detector import UXODetector

# Khởi tạo router cho nhóm API /admin
router = APIRouter(prefix="/admin", tags=["Admin"])

# Khởi tạo detector 1 lần (load model khi start server, không load lại mỗi request)
detector = UXODetector("computer_vision/weights/best.pt")

# ================================
# Login Admin
# ================================
@router.post("/login", response_model=AdminLoginResponse)
def login_admin(req: AdminLoginRequest, db: Session = Depends(connection.get_db)):
    """
    Đăng nhập Admin → kiểm tra email + password
    Nếu đúng → tạo JWT token để frontend dùng xác thực.
    """
    # Kiểm tra thông tin đăng nhập trong database
    admin = crud.authenticate_admin(db, email=req.email, password=req.password)
    if not admin:
        raise HTTPException(status_code=401, detail="❌ Email hoặc mật khẩu không đúng")
    # Tạo JWT token có payload là ID admin
    token = create_access_token(data={"sub": str(admin.id)})
    # Trả về access token cho client
    return AdminLoginResponse(access_token=token)

# ================================
# YOLOv8 UXO Detection API
# ================================
@router.post("/detect-uxo/")
async def detect_uxo_api(
    file: UploadFile = File(...),
    session_id: str = Form("default_session"),
    confidence_threshold: float = Form(0.3),
    db: Session = Depends(connection.get_db)
):
    """
    Nhận ảnh từ frontend, chạy YOLOv8 detect, TRẢ VỀ VÀ LƯU KẾT QUẢ VÀO DATABASE
    """
    try:
        # Đọc nội dung ảnh dưới dạng bytes
        image_bytes = await file.read()
        
        # Chạy detection
        detections = detector.detect_from_bytes(image_bytes, confidence_threshold=confidence_threshold)
        
        # LƯU KẾT QUẢ VÀO DATABASE
        detection_record = models.UXODetection(
            filename=file.filename,
            session_id=session_id,
            detected_objects=detections,  # Lưu toàn bộ kết quả detection
            image_data=image_bytes,  # Lưu ảnh gốc
            created_at=datetime.utcnow()
        )
        
        db.add(detection_record)
        db.commit()
        db.refresh(detection_record)
        
        # Tạo message cảnh báo dựa trên kết quả
        if detections:
            warning_message = "⚠️ Cảnh báo: Có vật thể nghi ngờ UXO!"
            
            # Nếu có detection có confidence > 0.5 → tạo log cảnh báo
            if any(detection['confidence'] > 0.5 for detection in detections):
                detection_log = models.ImageDetectionLog(
                    detection_id=detection_record.id,
                    session_id=session_id,
                    warning_message=warning_message,
                    confidence=max(detection['confidence'] for detection in detections),
                    created_at=datetime.utcnow()
                )
                db.add(detection_log)
                db.commit()
        else:
            warning_message = "✅ Không phát hiện vật thể nguy hiểm."
        # -----------------------------
        # Trả kết quả cho frontend
        # -----------------------------
        return {
            "detection_id": detection_record.id,  # Trả về ID của detection record
            "detections": detections,
            "warning_message": warning_message,
            "saved_to_database": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {e}")

# ================================
# View chatlogs (Admin only)
# ================================
@router.get("/chatlogs")
def view_all_chatlogs(skip: int = 0, limit: int = 100,
                      db: Session = Depends(connection.get_db),
                      current_admin=Depends(get_current_admin)):
    """
    Chỉ Admin được xem toàn bộ chat logs (đã được xác thực bằng JWT).
    """
    logs = crud.get_all_chatlogs(db, skip=skip, limit=limit)
    return logs

# ================================
# Log chat message từ frontend
# ================================
@router.post("/log-chat")
async def log_chat_message(
    request: Request,
    db: Session = Depends(connection.get_db)
):
    """
    Frontend gửi log chat để lưu vào database
    body JSON: { "session_id": str, "message": str, "response": str }
    """
    try:
        data = await request.json()  # <-- thêm await
    except Exception:
        raise HTTPException(status_code=400, detail="❌ Dữ liệu không hợp lệ")

    session_id = data.get("session_id")
    message = data.get("message")
    response_text = data.get("response")  # tránh trùng tên với hàm response()

    if not session_id or not message or not response_text:
        raise HTTPException(status_code=400, detail="❌ Thiếu trường dữ liệu cần thiết")

    # Ghi vào bảng ChatLog
    db_chat = models.ChatLog(
        session_id=session_id,
        message=message,
        response=response_text,
        created_at=datetime.utcnow()
    )
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)

    return {"message": "✅ Chat log đã được lưu", "id": db_chat.id}

# ========================
# USER: Gửi báo cáo UXO
# ========================
@router.post("/report-uxo", response_model=UXOReportResponse)
def create_report(
    req: UXOReportCreate,
    db: Session = Depends(connection.get_db)
):
    """
    Người dùng gửi báo cáo vị trí nghi có UXO.
    """
    db_report = models.UXOReport(
        latitude=req.latitude,
        longitude=req.longitude,
        description=req.description,
        created_at=datetime.utcnow()
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

# ========================
# ADMIN: Xem toàn bộ báo cáo UXO
# ========================
@router.get("/uxo-reports", response_model=List[UXOReportResponse])
def get_all_reports(
    db: Session = Depends(connection.get_db),
    current_admin=Depends(get_current_admin)  # chỉ admin mới được xem
):
    """Lấy toàn bộ kết quả phát hiện ảnh (chỉ admin có quyền)."""
    return db.query(models.UXOReport).all()

# ========================
# ADMIN: Image detection logs
# ========================


@router.get("/all-detections", response_model=List[UXODetectionResponse])
def read_all_detections_admin(
    db: Session = Depends(connection.get_db),
    current_admin=Depends(get_current_admin)  # Yêu cầu admin
):
    """Lấy tất cả detections - chỉ admin"""
    return db.query(models.UXODetection).all()

# ================================
# LẤY ẢNH DETECTION GỐC
# ================================
@router.get("/detections/{report_id}")
def get_detection_image(
    report_id: int,
    db: Session = Depends(connection.get_db),
    current_admin=Depends(get_current_admin)
):
    """
    Trả lại ảnh binary của bản ghi detection theo ID.
    """
    detection = db.query(models.UXODetection).filter(models.UXODetection.id == report_id).first()
    if not detection or not detection.image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    from fastapi.responses import Response
    # Xác định media type từ filename
    media_type = "image/jpeg"
    if detection.filename and detection.filename.lower().endswith('.png'):
        media_type = "image/png"  
    return Response(content=detection.image_data, media_type=media_type)

# ================================
# USER: LẤY DANH SÁCH DETECTION CỦA MÌNH
# ================================
@router.get("/detections/", response_model=List[UXODetectionResponse])
def read_user_detections(
    session_id: str,
    db: Session = Depends(connection.get_db)
):
    """Lấy detections của user cụ thể - không yêu cầu auth"""
    return db.query(models.UXODetection).filter(
        models.UXODetection.session_id == session_id
    ).all()