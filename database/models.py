from sqlalchemy import Column, Integer, String, Text, JSON, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from .connection import Base
from datetime import datetime

# ============================
# Users
# ============================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<User id={self.id}>"

# ============================
# Admins
# ============================
class Admin(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Admin id={self.id} email={self.email}>"

# ============================
# Chat logs
# ============================
class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String, index=True, nullable=False)  # Thêm session_id để tracking
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    intent = Column(String, nullable=True)
    entities = Column(JSON, nullable=True, default=dict)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ChatLog id={self.id} session={self.session_id}>"

# ============================
# QA logs
# ============================
class QALog(Base):
    __tablename__ = "qa_logs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    nlu = Column(JSON, nullable=True, default=dict)
    memory_length = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<QALog id={self.id} session={self.session_id}>"

# ============================
# Image Detection logs
# ============================
class ImageDetectionLog(Base):
    __tablename__ = "image_detection_logs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    image_url = Column(Text, nullable=True)
    detections = Column(JSON, nullable=True, default=list)
    warning_message = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ImageDetectionLog id={self.id} session={self.session_id}>"

# ============================
# UXO Knowledge Base
# ============================
class UXOKnowledge(Base):
    __tablename__ = "uxo_knowledge_base"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)          
    description = Column(Text, nullable=False)                 
    danger_level = Column(String, nullable=False)              
    handling_procedure = Column(Text, nullable=False)          
    hotline = Column(String, nullable=True)                    

    def __repr__(self):
        return f"<UXOKnowledge id={self.id} name={self.name}>"
    
# ============================
# UXO Reports
# ============================
class UXOReport(Base):
    __tablename__ = "uxo_reports"
    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    description = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, reviewed, resolved
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  
    created_at = Column(DateTime(timezone=True), server_default=func.now())
