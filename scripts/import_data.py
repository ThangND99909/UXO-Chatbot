import os
import argparse
from data_layer.vector_store import vector_store
from data_layer.preprocessor import UXOPreprocessor
from langchain.schema import Document

def import_files(input_dir: str, file_types: list[str], chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Import tất cả file trong thư mục vào vector store mà không ghi đè dữ liệu cũ
    """
    total_chunks = 0

    # Check/load vector store
    if not vector_store.is_initialized():
        vector_store.load_or_create_vector_store()
        print(f"✅ Vector store đã sẵn sàng!")

    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_types):
                file_path = os.path.join(root, file)
                print(f"\n📄 Đang xử lý: {file_path}")

                try:
                    ids = vector_store.import_file(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    if ids:
                        total_chunks += len(ids)
                except Exception as e:
                    print(f"⚠️ Lỗi khi import {file}: {e}")

    print(f"\n🎯 Hoàn tất! Tổng cộng {total_chunks} chunks được thêm vào vector DB.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import PDF/TXT/DOCX vào Vector Database mà không mất dữ liệu cũ")
    parser.add_argument("--dir", type=str, required=True, help="Thư mục chứa dữ liệu (PDF/TXT/DOCX)")
    parser.add_argument("--types", type=str, default="pdf,txt,docx", help="Loại file (vd: pdf,txt,docx)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Kích thước chunk văn bản")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap giữa các chunk")
    args = parser.parse_args()

    file_types = [ext.strip().lower() for ext in args.types.split(",")]
    import_files(args.dir, file_types, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
