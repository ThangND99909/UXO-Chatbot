from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
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
        return self.vector_store
    
    def search_similar_documents(self, query, k=5):
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo")
        return self.vector_store.similarity_search(query, k=k)
    
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
