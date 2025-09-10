from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime


from database import models, crud, connection
from utils.auth import create_access_token, get_current_admin
from app.schemas import AdminLoginRequest, AdminLoginResponse, UXOReportCreate, UXOReportResponse

router = APIRouter(prefix="/admin", tags=["Admin"])

# ================================
# Login Admin
# ================================
@router.post("/login", response_model=AdminLoginResponse)
def login_admin(req: AdminLoginRequest, db: Session = Depends(connection.get_db)):
    """
    Đăng nhập Admin, trả về access token
    """
    admin = crud.authenticate_admin(db, email=req.email, password=req.password)
    if not admin:
        raise HTTPException(status_code=401, detail="❌ Email hoặc mật khẩu không đúng")

    token = create_access_token(data={"sub": str(admin.id)})
    return AdminLoginResponse(access_token=token)

# ================================
# View chatlogs (Admin only)
# ================================
@router.get("/chatlogs")
def view_all_chatlogs(skip: int = 0, limit: int = 100,
                      db: Session = Depends(connection.get_db),
                      current_admin=Depends(get_current_admin)):
    """
    Lấy danh sách chat logs cho admin
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

    # Lưu vào DB
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
    return db.query(models.UXOReport).all()