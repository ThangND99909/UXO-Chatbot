import uvicorn
from fastapi import FastAPI, HTTPException
import logging
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để fix import issues
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from ai_core.nlu_processor import NLUProcessor
from ai_core.retrieval_qa import UXORetrievalQA
from ai_core.memory_manager import UXOMemoryManager
from ai_core.llm_chain import GeminiLLM

# Import đúng cách từ data_layer.vector_store
try:
    from data_layer.vector_store import vector_store_manager
except ImportError:
    # Fallback nếu không import được
    from data_layer.vector_store import vector_store as vector_store_manager

# Import schemas từ file riêng
try:
    from schemas import ChatRequest, ChatResponse, QAResponse, ErrorResponse
except ImportError:
    # Fallback cho import schemas
    try:
        from app.schemas import ChatRequest, ChatResponse, QAResponse, ErrorResponse
    except ImportError:
        print("❌ Cannot import schemas. Please check the file structure.")
        raise

# ====== Logging ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== FastAPI setup ======
app = FastAPI(
    title="UXO Chatbot API",
    description="API for UXO (Unexploded Ordnance) Chatbot with Gemini AI",
    version="1.0.0"
)

# ====== AI module ======
try:
    llm = GeminiLLM()
    nlu = NLUProcessor(llm=llm)
    
    # Khởi tạo vector store trước khi tạo QA
    try:
        # Load vector store nếu đã tồn tại
        vector_store_instance = vector_store_manager.load_vector_store()
        logger.info("✅ Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not load vector store: {e}. Using empty store.")
        vector_store_instance = vector_store_manager
    
    qa = UXORetrievalQA(llm=llm, vector_store=vector_store_instance)  # Sửa thành instance
    memory_manager = UXOMemoryManager(k=5)
    logger.info("✅ AI modules initialized successfully")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize AI modules: {e}")
    raise

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
        "memory_ready": hasattr(memory_manager, 'save_context'),
        "vector_store_document_count": vector_store_instance.get_document_count() if hasattr(vector_store_instance, 'get_document_count') else 0
    }

# Sửa endpoint để dùng ChatRequest từ schemas
@app.post("/ask", response_model=QAResponse, responses={500: {"model": ErrorResponse}})
def ask_question(req: ChatRequest):  # Sửa thành ChatRequest
    try:
        logger.info(f"📥 Question from session {req.session_id}: {req.message}")
        
        # 1️⃣ NLU
        nlu_result = nlu.process_nlu(req.message, req.language)
        intent = nlu_result["intent"]
        logger.info(f"🧠 Intent detected: {intent}")

        # 2️⃣ QA theo intent
        answer = qa.get_response(req.message, intent, req.language)
        logger.info(f"💬 Answer generated: {answer[:100]}...")

        # 3️⃣ Lưu memory
        memory_manager.save_context(
            req.session_id, 
            {"question": req.message}, 
            {"answer": answer}
        )

        return {
            "question": req.message,  # Sửa thành req.message
            "answer": answer,
            "nlu": nlu_result,
            "session_id": req.session_id,
            "memory_length": len(memory_manager.get_messages(req.session_id))
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý câu hỏi: {str(e)}")

@app.delete("/memory/{session_id}")
def clear_session_memory(session_id: str):
    try:
        memory_manager.clear_memory(session_id)
        return {"message": f"Memory của session {session_id} đã được xóa."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xóa memory: {str(e)}")

# ====== Chạy server ======
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)