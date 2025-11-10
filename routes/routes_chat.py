from fastapi import APIRouter, Depends, HTTPException, Header, Cookie
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

router = APIRouter()

# ====== Import dependencies ======
from database import connection, models
from app.schemas import AdminLoginRequest, AdminLoginResponse, UXOReportCreate, UXOReportResponse, UXODetectionResponse, UXODetectionCreate
from app.schemas import ChatRequest, QAResponse, ErrorResponse


# ====== Import từ các module khác ======
# Các dependencies này sẽ được inject từ main.py
def get_ai_components():
    """Lấy AI components từ main.py"""
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent
    sys.path.append(str(root_dir))
    
    from ai_core.nlu_processor import NLUProcessor
    from ai_core.retrieval_qa import UXORetrievalQA
    from ai_core.llm_chain import GeminiLLM
    from data_layer.vector_store import vector_store_manager
    
    # Khởi tạo AI components (giống trong main.py)
    llm = GeminiLLM()
    nlu = NLUProcessor(llm=llm)
    
    try:
        vector_store_instance = vector_store_manager.load_vector_store()
    except Exception:
        vector_store_instance = vector_store_manager
        
    qa = UXORetrievalQA(llm=llm, vector_store=vector_store_instance)
    
    return {
        "nlu": nlu,
        "qa": qa
    }

def get_session_manager():
    """Lấy session manager từ core"""
    from session.session_manager import user_sessions, get_or_create_session, get_session_id_from_multiple_sources
    return {
        "sessions": user_sessions,
        "get_or_create_session": get_or_create_session,
        "get_session_id_from_multiple_sources": get_session_id_from_multiple_sources
    }

# ====== Chat Endpoints ======
@router.post("/ask", response_model=QAResponse, responses={500: {"model": ErrorResponse}})
def ask_question(
    req: ChatRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    session_id_cookie: Optional[str] = Cookie(None, alias="session_id"),
    db: Session = Depends(connection.get_db),
    ai_components=Depends(get_ai_components),
    session_manager=Depends(get_session_manager)
):
    """
    Endpoint chính cho hội thoại người dùng
    """
    try:
        # Lấy components
        nlu_processor = ai_components["nlu"]
        qa_processor = ai_components["qa"]
        sessions = session_manager["sessions"]
        get_or_create_session_func = session_manager["get_or_create_session"]
        get_session_id_func = session_manager["get_session_id_from_multiple_sources"]
        
        # Lấy hoặc tạo session mới
        session_id_from_sources = get_session_id_func(
            header_session_id=x_session_id,
            cookie_session_id=session_id_cookie,
            body_session_id=req.session_id
        )
        session_id = get_or_create_session_func(session_id_from_sources)
        
        # Phân tích ngôn ngữ tự nhiên
        nlu_result = nlu_processor.process_nlu(req.message, req.language)
        intent = nlu_result["intent"]
        
        # Sinh câu trả lời
        answer = qa_processor.get_response(
            question=req.message,
            intent=intent,
            session_id=session_id,
            language=req.language
        )

        # LƯU CHAT VÀO DATABASE
        db_chat = models.ChatLog(
            session_id=session_id,
            message=req.message,
            response=answer,
            intent=nlu_result.get("intent"),
            entities=nlu_result.get("entities", {}),
            confidence=nlu_result.get("confidence"),
            created_at=datetime.utcnow()
        )
        db.add(db_chat)
        db.commit()
        db.refresh(db_chat)

        return {
            "question": req.message,
            "answer": answer,
            "nlu": nlu_result,
            "session_id": session_id,
            "memory_length": len(qa_processor.memory_manager.get_messages(session_id)) if hasattr(qa_processor, 'memory_manager') else 0
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý câu hỏi: {str(e)}")

@router.get("/session/{session_id}")
def get_session_info(
    session_id: str,
    session_manager=Depends(get_session_manager)
):
    """Lấy thông tin chi tiết của 1 session"""
    sessions = session_manager["sessions"]
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session không tồn tại")
    
    session_info = sessions[session_id].copy()
    return session_info

@router.delete("/memory/{session_id}")
def clear_session_memory(
    session_id: str,
    ai_components=Depends(get_ai_components)
):
    """Xóa toàn bộ hội thoại trong memory"""
    qa_processor = ai_components["qa"]
    if hasattr(qa_processor, 'memory_manager'):
        qa_processor.memory_manager.clear_memory(session_id)
        return {"message": f"Memory của session {session_id} đã được xóa."}
    return {"message": "Memory manager không khả dụng"}

@router.delete("/session/{session_id}")
def delete_session(
    session_id: str,
    session_manager=Depends(get_session_manager),
    ai_components=Depends(get_ai_components)
):
    """Xóa hoàn toàn một session"""
    sessions = session_manager["sessions"]
    qa_processor = ai_components["qa"]
    
    if hasattr(qa_processor, 'memory_manager'):
        qa_processor.memory_manager.clear_memory(session_id)
    if session_id in sessions:
        del sessions[session_id]
    return {"message": f"Session {session_id} đã được xóa hoàn toàn."}