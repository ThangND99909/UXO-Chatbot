# ai_core/llm_chain.py
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from ai_core.prompts import intent_prompt, entity_prompt
from ai_core.parsers import NLUOutputParser
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
    - Runnable: giao diện chuẩn để dùng invoke() thay cho run() + Chuẩn hóa mọi thành phần có thể “chạy” được
    - Hỗ trợ input dạng str hoặc dict
    """
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        if "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    def invoke(self, inputs: Union[str, Dict], config=None, **kwargs) -> str:
        """
        Gọi mô hình Gemini sinh phản hồi dựa trên prompt đầu vào.
        """
        if "stop" in kwargs:
            kwargs.pop("stop")
        
        # Nếu input là dict → nối thành chuỗi prompt (dễ debug hơn)
        if isinstance(inputs, dict):
            prompt = "\n".join(f"{k}: {v}" for k, v in inputs.items())
        else:
            prompt = str(inputs)

        #Chỉ khởi tạo mô hình Gemini một lần
        if not hasattr(self, "model_instance"):
            self.model_instance = genai.GenerativeModel(self.model)

        # Gọi API sinh nội dung từ Gemini
        response = self.model_instance.generate_content(
            prompt,
            generation_config={"temperature": self.temperature}
        )

        # Trả về text sạch (xử lý fallback phòng lỗi)
        try:
            return response.candidates[0].content.parts[0].text.strip()
        except Exception:
            return response.text.strip()

# ================= LLMChain Manager =================
# Quản lý nhiều LLMChain dùng Gemini
class LLMChainManager:
    """
    Quản lý nhiều LLMChain dùng Gemini
    """
    def __init__(self, model: str = "gemini-1.5-t", temperature: float = 0.2):
        #Tạo 1 instance GeminiLLM dùng chung cho các chain
        self.llm = GeminiLLM(model=model, temperature=temperature)
        #Lưu trữ các chain đã tạo trong dictionary
        self.chains: Dict[str, LLMChain] = {}

    def create_chain(self, name: str, prompt_template: str, input_vars: list, parser=None):
        """
        Tạo mới một LLMChain:
        - name: tên chain (ví dụ "intent", "entity")
        - prompt_template: nội dung prompt
        - input_vars: danh sách biến cần trong prompt
        - parser: bộ phân tích đầu ra (output parser)
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=input_vars)
        #Kết hợp GeminiLLM + Prompt + Parser thành 1 chain hoàn chỉnh
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=parser,
            output_key="answer"  # ✅ đồng bộ với memory
        )
        #Lưu chain vào dictionary
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
    """
    Khởi tạo sẵn 2 chain mặc định:
    - intent: xác định ý định của người dùng
    - entity: trích xuất thực thể trong câu hỏi
    """
    from ai_core.nlu_processor import NLUOutputParser

    manager = LLMChainManager()

    # Chain 1: Intent detection (phát hiện ý định)
    manager.create_chain("intent", intent_prompt, ["question", "language"], parser=NLUOutputParser())
    # Chain 2: Entity extraction (trích xuất thực thể)(đâu là thông tin quan trong trong câu hỏi)
    manager.create_chain("entity", entity_prompt, ["question", "language"], parser=NLUOutputParser())

    return manager
