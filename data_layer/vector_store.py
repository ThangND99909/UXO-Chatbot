from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import BaseRetriever
from typing import List, Dict, Any
import json
import numpy as np
import os
from data_layer.preprocessor import UXOPreprocessor

class VectorStoreManager:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.vector_store = None
        self.persist_directory = None

    # ✅ Mới: check vector store đã init chưa
    def is_initialized(self) -> bool:
        return self.vector_store is not None

    # ================== CÁC HÀM CŨ ==================
    def create_vector_store(self, documents, persist_directory="./chroma_db",
                            json_path="data/uxo_full_documents.json",
                            npz_path="data/uxo_embeddings.npz"):
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=persist_directory
        )
        self.vector_store.persist()
        self.persist_directory = persist_directory

        # Lưu JSON
        data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Lưu NPZ
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        np.savez_compressed(npz_path, embeddings=embeddings, metadata=[doc.metadata for doc in documents])

        return self.vector_store

    def load_vector_store(self, persist_directory="./chroma_db"):
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )
        self.persist_directory = persist_directory
        return self.vector_store

    def load_or_create_vector_store(self, persist_directory="./chroma_db", force_create=False):
        """
        Nếu DB tồn tại → load
        Nếu chưa tồn tại hoặc force_create=True → tạo mới
        """
        self.persist_directory = persist_directory
        if os.path.exists(persist_directory) and not force_create:
            try:
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embedding_model
                )
                print(f"✅ Vector store đã load từ {persist_directory}")
            except Exception as e:
                print(f"⚠️ Không thể load vector store, sẽ tạo mới. Lỗi: {e}")
                self.vector_store = None
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_model
            )
            print(f"✅ Tạo vector store mới tại {persist_directory}")
        return self.vector_store

    def search_similar_documents(self, query, k=5):
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo")
        return self.vector_store.similarity_search(query, k=k)

    def as_retriever(self, search_type: str = "similarity", k: int = 5, **kwargs) -> BaseRetriever:
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo. Hãy load hoặc create vector store trước.")
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k, **kwargs}
        )

    def get_retriever(self, **kwargs) -> BaseRetriever:
        return self.as_retriever(**kwargs)

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo")
        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_document_count(self) -> int:
        if self.vector_store is None:
            return 0
        try:
            collection = self.vector_store._collection
            return collection.count() if collection else 0
        except:
            return len(self.vector_store.get()['documents']) if self.vector_store.get() else 0

    def get_collection_info(self) -> Dict[str, Any]:
        if self.vector_store is None:
            return {"status": "not_initialized"}
        info = {
            "document_count": self.get_document_count(),
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model.model_name
        }
        return info

    def add_documents(self, documents: List[Any], persist: bool = True) -> List[str]:
        if not self.is_initialized():
            print("⚠️ Vector store chưa khởi tạo, sẽ tạo mới.")
            self.load_or_create_vector_store()
        ids = self.vector_store.add_documents(documents)
        if persist:
            self.vector_store.persist()
        return ids

    def delete_documents(self, ids: List[str], persist: bool = True) -> None:
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo")
        self.vector_store.delete(ids)
        if persist:
            self.vector_store.persist()

    def clear_vector_store(self) -> None:
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo")
        try:
            collection = self.vector_store._collection
            if collection:
                collection.delete(where={})
                self.vector_store.persist()
        except Exception as e:
            print(f"Warning: Could not clear vector store: {e}")

    def health_check(self) -> Dict[str, Any]:
        return {
            "initialized": self.vector_store is not None,
            "document_count": self.get_document_count(),
            "persist_directory": self.persist_directory,
            "status": "healthy" if self.vector_store else "not_initialized"
        }

    def load_documents_from_json(self, filename):
        from langchain.schema import Document
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = []
        for item in data:
            doc = Document(
                page_content=item["content"],
                metadata=item["metadata"]
            )
            documents.append(doc)
        return documents

    def load_documents_from_jsonl(self, filename):
        from langchain.schema import Document
        documents = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                documents.append(Document(
                    page_content=obj["content"],
                    metadata=obj["metadata"]
                ))
        return documents

    def index_from_jsonl(self, jsonl_file, persist_directory="./chroma_db",
                         json_path="data/uxo_full_documents.json",
                         npz_path="data/uxo_embeddings.npz"):
        documents = self.load_documents_from_jsonl(jsonl_file)
        return self.create_vector_store(
            documents,
            persist_directory=persist_directory,
            json_path=json_path,
            npz_path=npz_path
        )

    # ================== HÀM IMPORT FILE ==================
    def import_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Import trực tiếp file PDF/TXT/DOCX vào vector store mà không ghi đè dữ liệu cũ
        """
        preprocessor = UXOPreprocessor()
        from langchain.schema import Document
        text = ""

        if file_path.lower().endswith(".pdf"):
            text = preprocessor.read_pdf(file_path)
        elif file_path.lower().endswith(".txt"):
            text = preprocessor.read_txt(file_path)
        elif file_path.lower().endswith(".docx"):
            text = preprocessor.read_docx(file_path)
        else:
            raise ValueError(f"⚠️ Không hỗ trợ định dạng: {file_path}")

        if not text.strip():
            print(f"⚠️ File rỗng hoặc không đọc được: {file_path}")
            return

        doc = Document(page_content=text, metadata={"source": file_path})
        chunks = preprocessor.split_documents([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not self.is_initialized():
            self.load_or_create_vector_store()

        ids = self.vector_store.add_documents(chunks)
        self.vector_store.persist()
        print(f"✅ Đã import {len(chunks)} chunks từ {file_path}")
        return ids

# ================== GLOBAL INSTANCE ==================
vector_store_manager = VectorStoreManager()
vector_store = vector_store_manager
