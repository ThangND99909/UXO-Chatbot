from data_layer.crawler import UXOCrawler
from data_layer.preprocessor import UXOPreprocessor
from data_layer.vector_store import VectorStoreManager

if __name__ == "__main__":
    # 1. Crawl dữ liệu
    crawler = UXOCrawler()
    raw_docs = []
    for key, url in crawler.sources.items():
        raw_docs.extend(crawler.crawl_domain(key, url, limit=30))

    # 2. Tiền xử lý: clean + chunk
    preprocessor = UXOPreprocessor()
    processed_docs = preprocessor.clean_and_chunk(raw_docs)

    # 3. Lưu dữ liệu thô đã xử lý ra file JSONL
    jsonl_file = "data/uxo_full_documents.jsonl"
    preprocessor.save_to_jsonl(processed_docs, jsonl_file)

    # 4. Load lại dữ liệu từ JSONL + index vào VectorStore
    vector_manager = VectorStoreManager()
    vector_store = vector_manager.index_from_jsonl(
        jsonl_file,
        persist_directory="./chroma_db",
        json_path="data/uxo_full_documents.json",
        npz_path="data/uxo_embeddings.npz"
    )

    print("✅ Pipeline completed: Crawl → Preprocess → JSONL + JSON + NPZ + ChromaDB saved")
