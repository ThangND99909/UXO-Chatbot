from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
from . import models
from .models import UXOReport
from utils.auth import verify_password, hash_password

# =======================
# User CRUD
# =======================
def create_user(db: Session, name: Optional[str] = None) -> models.User:
    db_user = models.User()
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db: Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_all_users(db: Session, skip: int = 0, limit: int = 100) -> List[models.User]:
    return db.query(models.User).offset(skip).limit(limit).all()

# =======================
# Admin CRUD
# =======================
# Chỉ giữ những hàm cần thiết cho login và truy vấn admin
def authenticate_admin(db: Session, email: str, password: str) -> Optional[models.Admin]:
    """Xác thực admin bằng email + password"""
    admin = db.query(models.Admin).filter(models.Admin.email == email).first()
    if admin and verify_password(password, admin.hashed_password):
        return admin
    return None

def get_admin_by_email(db: Session, email: str) -> Optional[models.Admin]:
    return db.query(models.Admin).filter(models.Admin.email == email).first()

def get_admin(db: Session, admin_id: int) -> Optional[models.Admin]:
    return db.query(models.Admin).filter(models.Admin.id == admin_id).first()

def get_all_admins(db: Session) -> List[models.Admin]:
    return db.query(models.Admin).all()

# =======================
# ChatLog CRUD
# =======================
def create_chat_log(db: Session, session_id: str, message: str, response: str,
                    intent: Optional[str] = None, entities: Optional[Dict[str, Any]] = None,
                    confidence: Optional[float] = None, user_id: Optional[int] = None):
    """Tạo log chat mới"""
    chat_log = models.ChatLog(
        user_id=user_id,
        session_id=session_id,
        message=message,
        response=response,
        intent=intent,
        entities=entities,
        confidence=confidence,
        created_at=datetime.utcnow()
    )
    db.add(chat_log)
    db.commit()
    db.refresh(chat_log)
    return chat_log

def get_all_chatlogs(db: Session, skip: int = 0, limit: int = 100) -> List[models.ChatLog]:
    """Lấy tất cả chat logs, sắp xếp theo thời gian giảm dần"""
    return db.query(models.ChatLog).order_by(models.ChatLog.created_at.desc()).offset(skip).limit(limit).all()

def get_chat_logs_by_session(db: Session, session_id: str, limit: int = 50) -> List[models.ChatLog]:
    """Lấy chat logs theo session"""
    return db.query(models.ChatLog)\
             .filter(models.ChatLog.session_id == session_id)\
             .order_by(models.ChatLog.created_at.desc())\
             .limit(limit).all()

# =======================
# QALog CRUD
# =======================
def create_qa_log(db: Session, session_id: str, question: str, answer: str,
                  nlu: Optional[Dict[str, Any]] = None, memory_length: Optional[int] = None):
    qa_log = models.QALog(
        session_id=session_id,
        question=question,
        answer=answer,
        nlu=nlu,
        memory_length=memory_length,
        created_at=datetime.utcnow()
    )
    db.add(qa_log)
    db.commit()
    db.refresh(qa_log)
    return qa_log

def get_qa_logs(db: Session, session_id: str, limit: int = 50) -> List[models.QALog]:
    return db.query(models.QALog)\
             .filter(models.QALog.session_id == session_id)\
             .order_by(models.QALog.created_at.desc())\
             .limit(limit).all()

# =======================
# ImageDetectionLog CRUD
# =======================
def create_image_detection_log(db: Session, session_id: str, image_url: Optional[str],
                               detections: List[Dict[str, Any]], warning_message: str,
                               confidence: float):
    detection_log = models.ImageDetectionLog(
        session_id=session_id,
        image_url=image_url,
        detections=detections,
        warning_message=warning_message,
        confidence=confidence,
        created_at=datetime.utcnow()
    )
    db.add(detection_log)
    db.commit()
    db.refresh(detection_log)
    return detection_log

def get_image_detections(db: Session, session_id: str, limit: int = 20) -> List[models.ImageDetectionLog]:
    return db.query(models.ImageDetectionLog)\
             .filter(models.ImageDetectionLog.session_id == session_id)\
             .order_by(models.ImageDetectionLog.created_at.desc())\
             .limit(limit).all()

# =======================
# UXOKnowledge CRUD
# =======================
def create_uxo_entry(db: Session, name: str, description: str,
                     danger_level: str, handling_procedure: str,
                     hotline: Optional[str] = None):
    entry = models.UXOKnowledge(
        name=name,
        description=description,
        danger_level=danger_level,
        handling_procedure=handling_procedure,
        hotline=hotline
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry

def get_uxo_entry(db: Session, uxo_id: int) -> Optional[models.UXOKnowledge]:
    return db.query(models.UXOKnowledge).filter(models.UXOKnowledge.id == uxo_id).first()

def get_uxo_by_name(db: Session, name: str) -> Optional[models.UXOKnowledge]:
    return db.query(models.UXOKnowledge).filter(models.UXOKnowledge.name.ilike(f"%{name}%")).first()

def get_all_uxo_entries(db: Session, skip: int = 0, limit: int = 100) -> List[models.UXOKnowledge]:
    return db.query(models.UXOKnowledge).offset(skip).limit(limit).all()

def update_uxo_entry(db: Session, uxo_id: int, update_data: Dict[str, Any]) -> Optional[models.UXOKnowledge]:
    entry = db.query(models.UXOKnowledge).filter(models.UXOKnowledge.id == uxo_id).first()
    if entry:
        for key, value in update_data.items():
            setattr(entry, key, value)
        db.commit()
        db.refresh(entry)
    return entry

def delete_uxo_entry(db: Session, uxo_id: int) -> Optional[models.UXOKnowledge]:
    entry = db.query(models.UXOKnowledge).filter(models.UXOKnowledge.id == uxo_id).first()
    if entry:
        db.delete(entry)
        db.commit()
    return entry

# =======================
# UXOReport CRUD
# =======================
def create_uxo_report(db: Session, latitude: float, longitude: float, description: str = None):
    report = UXOReport(latitude=latitude, longitude=longitude, description=description)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report
