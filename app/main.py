# app/main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from ai_core.nlu_processor import NLUProcessor
from ai_core.retrieval_qa import UXORetrievalQA
from ai_core.memory_manager import UXOMemoryManager
from ai_core.llm_chain import GeminiLLM

# ====== Vector store load ======
from data_layer import vector_store 

# ====== FastAPI setup ======
app = FastAPI(title="UXO Chatbot API")

# ====== AI module ======
llm = GeminiLLM()
nlu = NLUProcessor(llm=llm)
qa = UXORetrievalQA(llm=llm, vector_store=vector_store)
memory_manager = UXOMemoryManager(k=5)

# ====== Request schema ======
class QuestionRequest(BaseModel):
    session_id: str
    question: str
    language: Optional[str] = "vi"

# ====== Endpoint /ask ======
@app.post("/ask")
def ask_question(req: QuestionRequest):
    # 1️⃣ NLU
    nlu_result = nlu.process_nlu(req.question, req.language)
    intent = nlu_result["intent"]

    # 2️⃣ QA theo intent
    answer = qa.get_response(req.question, intent, req.language)

    # 3️⃣ Lưu memory
    memory_manager.save_context(req.session_id, {"question": req.question}, {"answer": answer})

    return {
        "question": req.question,
        "answer": answer,
        "nlu": nlu_result,
        "session_id": req.session_id,
        "memory_length": len(memory_manager.get_messages(req.session_id))
    }

# ====== Endpoint xóa memory nếu muốn ======
@app.delete("/memory/{session_id}")
def clear_session_memory(session_id: str):
    memory_manager.clear_memory(session_id)
    return {"message": f"Memory của session {session_id} đã được xóa."}

# ====== Chạy server ======
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
