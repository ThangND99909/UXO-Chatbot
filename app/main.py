import uvicorn
import logging
import sys
from pathlib import Path
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ====== Logging ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Thêm thư mục gốc vào sys.path ======
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

# ====== Import database & routes ======
from database import connection, models
from routes.routes_admin import router as admin_router
from routes.routes_chat import router as chat_router  # ✅ MỚI
from session.session_manager import cleanup_old_sessions  # ✅ MỚI

# ====== Import AI modules ======
from ai_core.nlu_processor import NLUProcessor
from ai_core.retrieval_qa import UXORetrievalQA
from ai_core.llm_chain import GeminiLLM
from data_layer.vector_store import vector_store_manager

# ====== Khởi tạo FastAPI ======
app = FastAPI(
    title="UXO Chatbot API",
    description="API for UXO (Unexploded Ordnance) Chatbot with Gemini AI",
    version="1.0.0"
)

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Khởi tạo database ======
connection.create_db_tables(models)

# ====== Khởi tạo các mô-đun AI ======
try:
    llm = GeminiLLM()
    nlu = NLUProcessor(llm=llm)

    try:
        vector_store_instance = vector_store_manager.load_vector_store()
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load vector store: {e}. Using empty store.")
        vector_store_instance = vector_store_manager

    qa = UXORetrievalQA(llm=llm, vector_store=vector_store_instance)
    logger.info("AI modules initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI modules: {e}")
    raise

# ====== Include routers ======
app.include_router(admin_router)
app.include_router(chat_router)  # ✅ CHAT ROUTES

# ====== Health endpoints (GIỮ LẠI) ======
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
        "vector_store_document_count": vector_store_instance.get_document_count() if hasattr(vector_store_instance, 'get_document_count') else 0
    }

# ====== Startup event ======
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_sessions(qa))  # ✅ TRUYỀN qa instance
    logger.info("Cleanup task started")

# ====== Chạy server ======
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)