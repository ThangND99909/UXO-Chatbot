# run_ai_core.py
import os
from ai_core.nlu_processor import NLUProcessor
from ai_core.retrieval_qa import UXORetrievalQA
from ai_core.memory_manager import UXOMemoryManager
from ai_core.llm_chain import GeminiLLM
from data_layer.vector_store import VectorStoreManager

def main():
    # ================= 1. Load vector store từ data layer =================
    print("🔹 Loading vector store...")
    vs_manager = VectorStoreManager()
    try:
        vector_store = vs_manager.load_vector_store("./chroma_db")
        print(f"✅ Loaded vector store with {len(vector_store.get())} documents")
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        return

    # ================= 2. Khởi tạo LLM (Gemini) =================
    print("🔹 Initializing GeminiLLM...")
    llm = GeminiLLM()  # Sử dụng model mặc định "gemini-1.5-t"

    # ================= 3. Khởi tạo NLU Processor =================
    print("🔹 Initializing NLUProcessor...")
    nlu = NLUProcessor(llm=llm)

    # ================= 4. Khởi tạo QA =================
    print("🔹 Initializing UXORetrievalQA...")
    qa = UXORetrievalQA(llm=llm, vector_store=vector_store)

    # ================= 5. Khởi tạo Memory Manager =================
    memory_mgr = UXOMemoryManager(k=5)
    session_id = "test_session"

    # ================= 6. Test pipeline =================
    test_questions = [
        "Tôi tìm thấy một quả bom ở sân sau, nên làm gì?",
        "Bom mìn chưa nổ nguy hiểm như thế nào?",
        "Số hotline để báo UXO là gì?"
    ]
    language = "vi"

    for q in test_questions:
        print("\n============================")
        print(f"❓ Question: {q}")

        # --- NLU ---
        nlu_result = nlu.process_nlu(q, language)
        print("🧠 NLU result:", nlu_result)

        # --- QA dựa trên intent ---
        intent = nlu_result.get("intent", "definition")
        response = qa.get_response(q, intent, language)
        print("💬 QA response:", response)

        # --- Lưu vào memory ---
        memory_mgr.save_context(session_id, {"question": q}, {"answer": response})
        memory = memory_mgr.get_memory(session_id)
        print("📝 Memory buffer:", [m.content for m in memory_mgr.get_messages(session_id)])

if __name__ == "__main__":
    main()
