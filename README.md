setting lib: pip install -r requirements.txt

data_layer
crawler.py (thu thập dữ liệu từ web, file, API)
     ↓
preprocessor.py (làm sạch, chuẩn hóa, chia nhỏ văn bản)
     ↓
vector_store.py (tạo embedding, lưu vào FAISS/Qdrant/Chroma) -> run file to test: python -m data_layer.run
     ↓
retrieval_qa.py (dùng để truy vấn RAG + LLM)

chroma_db: thư mục lưu trữ cơ sở dữ liệu vector (vector database)

chroma.sqlite3 → database chính (lưu metadata, collections, mappings giữa doc-id và embedding).

Thư mục có tên như cf687ce7-0cb1-4842-841c-a884505e6b9d → chính là faiss/HNSW index được Chroma tạo ra để lưu vector embeddings.

data_level0.bin, header.bin, length.bin, link_lists.bin → là các file nhị phân lưu đồ thị HNSW index.

index_metadata.pickle → metadata cho index.

######################################
ai_core
1. nlu_processor.py

Mục đích: Xử lý NLU (Natural Language Understanding) cho chatbot.

Chức năng chính:

Phát hiện intent của người dùng (ví dụ: hỏi định nghĩa, hướng dẫn an toàn, báo cáo UXO…).

Trích xuất entities (thực thể) từ câu hỏi (ví dụ: location, loại UXO, hành động).

Kết hợp intent detection và entity extraction thành một hàm duy nhất process_nlu.

2. memory_manager.py

Mục đích: Quản lý bộ nhớ hội thoại cho từng session người dùng.

Chức năng chính:

Lưu giữ ngữ cảnh các tin nhắn gần đây (ConversationBufferWindowMemory) để hỗ trợ multi-turn conversation.

Lấy, lưu và xóa memory theo session_id.

3. retrieval_qa.py

Mục đích: Xử lý QA dựa trên tài liệu (retrieval-based QA) cho các intent khác nhau.

Chức năng chính:

Khởi tạo các chain QA với prompt template khác nhau cho từng intent (ví dụ: definition, safety_advice).

Tích hợp vector store để truy xuất thông tin từ tài liệu UXO đã index.

Cung cấp hàm get_response trả lời câu hỏi dựa trên intent và ngôn ngữ.

4. llm_chain.py

Mục đích: Khởi tạo và cấu hình LLM (Language Model) cho chatbot.

Chức năng chính:

Tạo instance của GeminiLLM (hoặc OpenAI/LLM khác) để sử dụng trong các chain NLU, QA.

Có thể đặt các tham số như temperature, max tokens, model name để điều chỉnh hành vi LLM.

run: python -m tests.run_ai_core

setup version: python3.10 -m venv venv310
kích hoạt: .\venv310\Scripts\Activate.ps1

## 📂 API Module

- **`api/schemas.py`**  
  Định nghĩa các **Pydantic models** cho request/response của API (ví dụ: `ChatRequest`, `QAResponse`, `ErrorResponse`).  
  → Giúp chuẩn hóa dữ liệu trao đổi giữa client ↔ server, hỗ trợ validation và tự động sinh tài liệu API.

- **`api/main.py`**  
  File **entrypoint FastAPI** của hệ thống.  
  → Khởi tạo ứng dụng, load các module AI (LLM, NLU, QA, memory) và định nghĩa các **endpoint** chính:  
    - `/` – Health check nhanh  
    - `/health` – Trạng thái chi tiết các module  
    - `/ask` – Đặt câu hỏi và nhận câu trả lời từ chatbot  
    - `/memory/{session_id}` – Xóa bộ nhớ hội thoại theo session
    
app/main.py
 app/
│   │── main.py          # FastAPI entrypoint
│   │── schema.py        # Pydantic models
run fastapi: uvicorn app.main:app --reload

Truy cập API

Health check: http://localhost:8000/

API docs (Swagger UI): http://localhost:8000/docs

Alternative docs (ReDoc): http://localhost:8000/redoc

frontend
before run frontend, we need to run backend first
run frontend: cd: frontend/streamlit run app.py