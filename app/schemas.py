from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    session_id: str
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
    session_id: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None  # Thêm option base64

class ImageDetectionResponse(BaseModel):
    detections: List[Dict[str, Any]]
    warning_message: str
    session_id: str
    confidence: float

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None