# ai_core/llm_chain.py
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict, Union
import google.generativeai as genai
import os
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import Runnable

from dotenv import load_dotenv
load_dotenv()  # nạp file .env

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

logger = logging.getLogger(__name__)

# ================= Gemini Wrapper cho LangChain =================
class GeminiLLM(Runnable):
    """
    Wrapper Gemini LLM tương thích LangChain 2.x
    """
    def __init__(self, model: str = "gemini-1.5-flash", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        if "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    def invoke(self, inputs: Union[str, Dict], config=None, **kwargs) -> str:
        if "stop" in kwargs:
            kwargs.pop("stop")

        if isinstance(inputs, dict):
            prompt = "\n".join(f"{k}: {v}" for k, v in inputs.items())
        else:
            prompt = str(inputs)

        if not hasattr(self, "model_instance"):
            self.model_instance = genai.GenerativeModel(self.model)

        response = self.model_instance.generate_content(
            prompt,
            generation_config={"temperature": self.temperature}
        )

        # Lấy text sạch
        try:
            return response.candidates[0].content.parts[0].text.strip()
        except Exception:
            return response.text.strip()

# ================= LLMChain Manager =================
class LLMChainManager:
    """
    Quản lý nhiều LLMChain dùng Gemini
    """
    def __init__(self, model: str = "gemini-1.5-t", temperature: float = 0.2):
        self.llm = GeminiLLM(model=model, temperature=temperature)
        self.chains: Dict[str, LLMChain] = {}

    def create_chain(self, name: str, prompt_template: str, input_vars: list, parser=None):
        prompt = PromptTemplate(template=prompt_template, input_variables=input_vars)
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=parser,
            output_key="answer"  # ✅ đồng bộ với memory
        )
        self.chains[name] = chain
        logger.info(f"✅ Created Gemini chain: {name}")
        return chain

    def get_chain(self, name: str) -> LLMChain:
        """Lấy chain đã tạo theo tên"""
        return self.chains.get(name, None)

    def run_chain(self, name: str, inputs: dict) -> str:
        """Chạy chain với input dictionary"""
        chain = self.get_chain(name)
        if not chain:
            raise ValueError(f"❌ Chain '{name}' chưa được tạo.")
        # Dùng invoke thay run
        return chain.invoke(inputs)

# ================= Ví dụ tạo các chain mặc định =================
def build_default_gemini_chains():
    from ai_core.nlu_processor import NLUOutputParser

    manager = LLMChainManager()

    # Intent detection
    intent_prompt = """
    Phân tích câu hỏi sau và xác định ý định (intent) của người dùng.
    Các intent có thể là:
    - definition: hỏi về định nghĩa, khái niệm
    - safety_advice: hỏi về hướng dẫn an toàn
    - location_info: hỏi về thông tin địa điểm
    - report_uxo: báo cáo vật nổ
    - ask_hotline: hỏi số hotline
    - general: câu hỏi chung khác

    Câu hỏi: {question}
    Ngôn ngữ: {language}

    Trả về JSON với cấu trúc:
    {{"intent": "tên_intent", "confidence": độ_tin_cậy (0-1)}}
    """
    manager.create_chain("intent", intent_prompt, ["question", "language"], parser=NLUOutputParser())

    # Entity extraction
    entity_prompt = """
    Trích xuất thực thể (entities) từ câu hỏi sau:

    Câu hỏi: {question}
    Ngôn ngữ: {language}

    Các loại thực thể cần trích xuất:
    - location: địa điểm, tỉnh thành
    - uxo_type: loại vật nổ (bom, mìn, lựu đạn, etc.)
    - action: hành động

    Trả về JSON với cấu trúc:
    {{"entities": {{"location": ["địa điểm 1", "địa điểm 2"],
                    "uxo_type": ["loại vật nổ 1", "loại vật nổ 2"],
                    "action": ["hành động 1", "hành động 2"]}}}}
    """
    manager.create_chain("entity", entity_prompt, ["question", "language"], parser=NLUOutputParser())

    return manager
