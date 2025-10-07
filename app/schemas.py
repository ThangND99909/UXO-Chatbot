from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # ✅ Cho phép None để linh hoạt
    language: str = "vi"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    entities: Dict[str, Any]

class QAResponse(BaseModel):
    question: str
    answer: str
    nlu: Dict[str, Any]
    session_id: str
    memory_length: int

class ImageDetectionRequest(BaseModel):
    session_id: Optional[str] = None  # ✅ Cho phép None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

class ImageDetectionResponse(BaseModel):
    detections: List[Dict[str, Any]]
    warning_message: str
    session_id: str
    confidence: float

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None


# ===== Admin schemas =====
class AdminRegisterRequest(BaseModel):
    email: EmailStr
    password: str

class AdminRegisterResponse(BaseModel):
    message: str
    admin_id: int
    email: EmailStr

class AdminLoginRequest(BaseModel):
    email: EmailStr
    password: str

class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UXOReportCreate(BaseModel):
    latitude: float
    longitude: float
    description: str | None = None

class UXOReportResponse(UXOReportCreate):
    id: int
    status: str = "pending"
    created_at: Optional[datetime] = None
    user_id: Optional[int] = None

    class Config:
        orm_mode = True

# ================================
# Schemas cho update/delete
# ================================
class ChatUpdateRequest(BaseModel):
    new_content: str

class ChatDeleteResponse(BaseModel):
    message: str
    chat_id: int

# ================================
# Schemas cho image detection
# ================================
# Schema cho việc tạo detection (upload ảnh)
class UXODetectionCreate(BaseModel):
    filename: str
    session_id: Optional[str] = None
    


class UXODetectionResponse(BaseModel):
    id: int
    filename: str
    session_id: Optional[str] = None
    detected_objects: Optional[List[Dict[str, Any]]] = []
    created_at: datetime

    class Config:
        from_attributes = True