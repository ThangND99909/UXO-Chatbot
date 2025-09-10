import uvicorn
import logging
import sys
from pathlib import Path
from typing import Optional
import asyncio
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Header, Cookie
from fastapi.middleware.cors import CORSMiddleware

# ====== Logging ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Thêm thư mục gốc vào sys.path ======
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

# ====== Import AI modules ======
from ai_core.nlu_processor import NLUProcessor
from ai_core.retrieval_qa import UXORetrievalQA
from ai_core.llm_chain import GeminiLLM

# ====== Import database & routes ======
from database import connection, models, crud
from routes.routes_admin import router as admin_router

# ====== Import vector store ======
from data_layer.vector_store import vector_store_manager

# ====== Import schemas ======
try:
    from schemas import ChatRequest, ChatResponse, QAResponse, ErrorResponse
except ImportError:
    from app.schemas import ChatRequest, ChatResponse, QAResponse, ErrorResponse

# ====== Khởi tạo FastAPI ======
app = FastAPI(
    title="UXO Chatbot API",
    description="API for UXO (Unexploded Ordnance) Chatbot with Gemini AI",
    version="1.0.0"
)

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Include router admin ======
app.include_router(admin_router)

# ====== Khởi tạo database ======
connection.create_db_tables(models)

# ====== AI module initialization ======
try:
    llm = GeminiLLM()
    nlu = NLUProcessor(llm=llm)

    # Load vector store trực tiếp từ data_layer
    try:
        vector_store_instance = vector_store_manager.load_vector_store()
        logger.info("✅ Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not load vector store: {e}. Using empty store.")
        vector_store_instance = vector_store_manager

    qa = UXORetrievalQA(llm=llm, vector_store=vector_store_instance)
    logger.info("✅ AI modules initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize AI modules: {e}")
    raise

# ====== Session Management ======
user_sessions = {}

def get_or_create_session(session_id: Optional[str] = None) -> str:
    import uuid
    if not session_id or session_id not in user_sessions:
        new_session_id = str(uuid.uuid4())
        user_sessions[new_session_id] = {
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0
        }
        logger.info(f"🆕 Created new session: {new_session_id}")
        return new_session_id
    
    user_sessions[session_id]["last_activity"] = datetime.now()
    user_sessions[session_id]["message_count"] += 1
    return session_id

def get_session_id_from_multiple_sources(
    header_session_id: Optional[str] = None,
    cookie_session_id: Optional[str] = None,
    body_session_id: Optional[str] = None
) -> Optional[str]:
    return header_session_id or cookie_session_id or body_session_id

# ====== Endpoints ======
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "UXO Chatbot API"}

@app.get("/health")
def health_detail():
    vector_store_status = "not_initialized"
    if hasattr(vector_store_instance, 'health_check'):
        try:
            vector_store_status = vector_store_instance.health_check().get("status", "unknown")
        except:
            vector_store_status = "error"
    
    return {
        "status": "healthy",
        "llm_ready": hasattr(llm, 'invoke'),
        "vector_store_ready": vector_store_status,
        "nlu_ready": hasattr(nlu, 'process_nlu'),
        "active_sessions": len(user_sessions),
        "vector_store_document_count": vector_store_instance.get_document_count() if hasattr(vector_store_instance, 'get_document_count') else 0
    }

@app.post("/ask", response_model=QAResponse, responses={500: {"model": ErrorResponse}})
def ask_question(
    req: ChatRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    session_id_cookie: Optional[str] = Cookie(None, alias="session_id")
):
    try:
        session_id_from_sources = get_session_id_from_multiple_sources(
            header_session_id=x_session_id,
            cookie_session_id=session_id_cookie,
            body_session_id=req.session_id
        )
        session_id = get_or_create_session(session_id_from_sources)
        logger.info(f"📥 Question from session {session_id}: {req.message}")

        nlu_result = nlu.process_nlu(req.message, req.language)
        intent = nlu_result["intent"]
        logger.info(f"🧠 Intent detected: {intent}")

        answer = qa.get_response(
            question=req.message,
            intent=intent,
            session_id=session_id,
            language=req.language
        )
        logger.info(f"💬 Answer generated: {answer[:100]}...")

        return {
            "question": req.message,
            "answer": answer,
            "nlu": nlu_result,
            "session_id": session_id,
            "memory_length": len(qa.memory_manager.get_messages(session_id)) if hasattr(qa, 'memory_manager') else 0
        }
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý câu hỏi: {str(e)}")

@app.get("/session/{session_id}")
def get_session_info(session_id: str):
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session không tồn tại")
    session_info = user_sessions[session_id].copy()
    if hasattr(qa, 'memory_manager'):
        try:
            memory_messages = qa.memory_manager.get_messages(session_id)
            session_info["memory_message_count"] = len(memory_messages)
        except:
            session_info["memory_message_count"] = 0
    return session_info

@app.delete("/memory/{session_id}")
def clear_session_memory(session_id: str):
    try:
        if hasattr(qa, 'memory_manager'):
            qa.memory_manager.clear_memory(session_id)
            return {"message": f"Memory của session {session_id} đã được xóa."}
        else:
            return {"message": "Memory manager không khả dụng"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xóa memory: {str(e)}")

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    try:
        if hasattr(qa, 'memory_manager'):
            qa.memory_manager.clear_memory(session_id)
        if session_id in user_sessions:
            del user_sessions[session_id]
        return {"message": f"Session {session_id} đã được xóa hoàn toàn."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xóa session: {str(e)}")

# ====== Cleanup task ======
async def cleanup_old_sessions():
    while True:
        try:
            now = datetime.now()
            to_delete = [sid for sid, data in user_sessions.items() if now - data["last_activity"] > timedelta(hours=24)]
            for sid in to_delete:
                if hasattr(qa, 'memory_manager'):
                    qa.memory_manager.clear_memory(sid)
                del user_sessions[sid]
                logger.info(f"🧹 Cleaned up old session: {sid}")
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"❌ Error in cleanup task: {e}")
            await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_sessions())
    logger.info("✅ Cleanup task started")

# ====== Chạy server ======
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
