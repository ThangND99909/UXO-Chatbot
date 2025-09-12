import re
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import numpy as np

# ✅ Bổ sung import
from PyPDF2 import PdfReader
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
from docx import Document as DocxDocument
from pdf2image import convert_from_path
import pytesseract


class UXOPreprocessor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'(javascript:|window\.|var\s+)', '', text, flags=re.IGNORECASE)
        noise_patterns = [
            r'Follow us.*', r'Subscribe.*', r'Contact us.*', r'©.*\d{4}.*',
            r'Terms of Use.*', r'Privacy Policy.*', r'All rights reserved.*',
            r'Sitemap.*', r'Search.*',
        ]
        for pat in noise_patterns:
            text = re.sub(pat, '', text, flags=re.IGNORECASE)
        html_entities = {
            '&nbsp;': ' ', '&amp;': '&', '&quot;': '"',
            '&apos;': "'", '&lt;': '<', '&gt;': '>',
            '\u2013': '-', '\u2014': '-', '\u2022': '•',
        }
        for k, v in html_entities.items():
            text = text.replace(k, v)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_documents(self, documents):
        processed_docs = []
        for doc in documents:
            content = self.clean_text(doc.page_content)
            metadata = doc.metadata
            doc_type = "general"
            if re.search(r"\b(safety|an toàn|hướng dẫn)\b", content, re.IGNORECASE):
                doc_type = "safety_guidelines"
            elif re.search(r"\b(hotline|liên hệ|contact)\b", content, re.IGNORECASE):
                doc_type = "contact_info"
            elif re.search(r"\b(bom|mìn|uxo|ordnance)\b", content, re.IGNORECASE):
                doc_type = "uxo_info"
            processed_doc = Document(
                page_content=content,
                metadata={**metadata, "type": doc_type}
            )
            processed_docs.append(processed_doc)
        return processed_docs

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return splitter.split_documents(documents)

    def embed_documents(self, documents):
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(
            texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True
        )
        return embeddings

    def save_to_jsonl(self, documents, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps({"content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False) + "\n")

    def save_embeddings(self, embeddings, documents, out_path="data/uxo_embeddings.npz"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, embeddings=embeddings, metadata=[doc.metadata for doc in documents])

    # ✅ Thêm hàm clean_and_chunk để chạy trong run.py
    def clean_and_chunk(self, raw_docs, chunk_size=1000, chunk_overlap=200):
        """Làm sạch → phân loại → chunk văn bản"""
        processed = self.process_documents(raw_docs)
        chunks = self.split_documents(processed, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunks

    # ✅ Nâng cấp read_pdf: text + OCR nếu page rỗng
    def read_pdf(self, file_path: str) -> str:
        """Đọc PDF, ưu tiên text, fallback OCR nếu cần và có poppler"""
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"⚠️ Lỗi đọc PDF trực tiếp: {e}")

        # Nếu không đọc được text nào, thử OCR
        if not text.strip():
            if not OCR_AVAILABLE:
                print(f"⚠️ Không thể đọc text và OCR không khả dụng: {file_path}")
                return ""
            try:
                # OCR bằng pdf2image + pytesseract
                poppler_path = r"E:\Poppler\poppler-24.07.0\Library\bin"  # Cập nhật đường dẫn poppler nếu cần
                images = convert_from_path(file_path, poppler_path=poppler_path)
                for i, img in enumerate(images):
                    ocr_text = pytesseract.image_to_string(img, lang='vie+eng')
                    text += ocr_text + "\n"
            except Exception as e:
                print(f"⚠️ OCR thất bại cho file {file_path}: {e}")
                return ""

        return text

    # ✅ Bổ sung: đọc TXT
    def read_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # ✅ Bổ sung: đọc DOCX
    def read_docx(self, file_path: str) -> str:
        doc = DocxDocument(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])
