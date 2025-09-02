from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import BaseRetriever
from typing import List, Dict, Any, Optional, Union
import json
import numpy as np
import os

class VectorStoreManager:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.vector_store = None
        self.persist_directory = None
    
    def create_vector_store(self, documents, persist_directory="./chroma_db",
                            json_path="data/uxo_full_documents.json",
                            npz_path="data/uxo_embeddings.npz"):
        """
        Tạo vector store và đồng thời lưu ra:
        - Chroma DB (SQLite + npz)
        - File JSON (text + metadata)
        - File NPZ (embeddings + metadata)
        """
        # 1. Tạo vector store và persist
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=persist_directory
        )
        self.vector_store.persist()
        self.persist_directory = persist_directory

        # 2. Lưu documents ra JSON
        data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 3. Xuất embeddings ra NPZ
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
    
    def search_similar_documents(self, query, k=5):
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo")
        return self.vector_store.similarity_search(query, k=k)
    
    def as_retriever(self, search_type: str = "similarity", k: int = 5, **kwargs) -> BaseRetriever:
        """
        Chuyển đổi vector store thành retriever để sử dụng với LangChain chains
        """
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo. Hãy load hoặc create vector store trước.")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k, **kwargs}
        )
    
    def get_retriever(self, **kwargs) -> BaseRetriever:
        """
        Alias cho as_retriever để tương thích với code cũ
        """
        return self.as_retriever(**kwargs)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Tìm kiếm tương đồng với điểm số confidence
        """
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo")
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_document_count(self) -> int:
        """
        Lấy tổng số documents trong vector store
        """
        if self.vector_store is None:
            return 0
        # Chroma specific way to get count
        try:
            collection = self.vector_store._collection
            return collection.count() if collection else 0
        except:
            return len(self.vector_store.get()['documents']) if self.vector_store.get() else 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về collection
        """
        if self.vector_store is None:
            return {"status": "not_initialized"}
        
        info = {
            "document_count": self.get_document_count(),
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model.model_name
        }
        
        return info
    
    def add_documents(self, documents: List[Any], persist: bool = True) -> List[str]:
        """
        Thêm documents mới vào vector store
        """
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo")
        
        ids = self.vector_store.add_documents(documents)
        if persist:
            self.vector_store.persist()
        return ids
    
    def delete_documents(self, ids: List[str], persist: bool = True) -> None:
        """
        Xóa documents khỏi vector store
        """
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo")
        
        self.vector_store.delete(ids)
        if persist:
            self.vector_store.persist()
    
    def clear_vector_store(self) -> None:
        """
        Xóa toàn bộ dữ liệu trong vector store
        """
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo")
        
        try:
            # Chroma specific deletion
            collection = self.vector_store._collection
            if collection:
                collection.delete(where={})  # Delete all documents
                self.vector_store.persist()
        except Exception as e:
            print(f"Warning: Could not clear vector store: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Kiểm tra tình trạng của vector store
        """
        return {
            "initialized": self.vector_store is not None,
            "document_count": self.get_document_count(),
            "persist_directory": self.persist_directory,
            "status": "healthy" if self.vector_store else "not_initialized"
        }
    
    def load_documents_from_json(self, filename):
        """Load dữ liệu từ JSON chuẩn (list object)."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        from langchain.schema import Document
        documents = []
        for item in data:
            doc = Document(
                page_content=item["content"],
                metadata=item["metadata"]
            )
            documents.append(doc)
        return documents

    def load_documents_from_jsonl(self, filename):
        """Load dữ liệu từ JSONL (mỗi dòng 1 object)."""
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
        """Tạo vector store trực tiếp từ JSONL và lưu thêm JSON/NPZ"""
        documents = self.load_documents_from_jsonl(jsonl_file)
        return self.create_vector_store(
            documents,
            persist_directory=persist_directory,
            json_path=json_path,
            npz_path=npz_path
        )


# Tạo global instance để import
vector_store_manager = VectorStoreManager()

# Alias cho tương thích với code cũ
vector_store = vector_store_manager