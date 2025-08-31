# ai_core/nlu_processor.py
import json
import re
import logging
from typing import Dict, Any
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser

from .llm_chain import GeminiLLM  # Wrapper LLM tuỳ chỉnh

# ========================
# Logging setup
# ========================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ========================
# Output Parser
# ========================
class NLUOutputParser(BaseOutputParser):
    """Parser an toàn cho output từ LLM (JSON -> Dict)"""

    def parse(self, text: str) -> Dict[str, Any]:
        try:
            logger.debug(f"🔹 Raw LLM output: {text}")
            # Tìm JSON trong text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                logger.debug(f"✅ Parsed JSON: {parsed}")
                return parsed
            else:
                logger.warning("⚠️ Không tìm thấy JSON trong output")
                return {}
        except Exception as e:
            logger.error(f"❌ Parse lỗi: {e}")
            return {}

    def get_format_instructions(self) -> str:
        """Hướng dẫn format JSON cho LLM (có thể dùng trong prompt)."""
        return "Trả lời dưới dạng JSON hợp lệ."


# ========================
# NLU Processor
# ========================
class NLUProcessor:
    def __init__(self, llm=None):
        """
        llm: object LLM, nếu None sẽ tự khởi tạo GeminiLLM mặc định
        """
        self.llm = llm or GeminiLLM()
        self.setup_intent_detection()
        self.setup_entity_extraction()

    # ----------------------------
    # Intent Detection
    # ----------------------------
    def setup_intent_detection(self):
        intent_template = """
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

        Trả lời dưới dạng JSON với cấu trúc:
        {{
            "intent": "tên_intent",
            "confidence": số_thập_phân_từ_0_đến_1
        }}
        """

        self.intent_prompt = PromptTemplate(
            template=intent_template,
            input_variables=["question", "language"],
        )

        self.intent_chain = LLMChain(
            llm=self.llm,
            prompt=self.intent_prompt,
            output_parser=NLUOutputParser(),
            output_key="answer"
        )

    # ----------------------------
    # Entity Extraction
    # ----------------------------
    def setup_entity_extraction(self):
        entity_template = """
        Trích xuất thực thể (entities) từ câu hỏi sau:
        Câu hỏi: {question}
        Ngôn ngữ: {language}

        Các loại thực thể cần trích xuất:
        - location: địa điểm, tỉnh thành
        - uxo_type: loại vật nổ (bom, mìn, lựu đạn, etc.)
        - action: hành động

        Trả lời dưới dạng JSON với cấu trúc:
        {{
            "entities": {{
                "location": [],
                "uxo_type": [],
                "action": []
            }}
        }}
        """

        self.entity_prompt = PromptTemplate(
            template=entity_template,
            input_variables=["question", "language"],
        )

        self.entity_chain = LLMChain(
            llm=self.llm,
            prompt=self.entity_prompt,
            output_parser=NLUOutputParser(),
            output_key="answer"
        )

    # ----------------------------
    # API: Detect Intent
    # ----------------------------
    def detect_intent(self, question: str, language: str = "vi") -> Dict[str, Any]:
        try:
            raw = self.intent_chain.invoke({"question": question, "language": language})
            # Nếu raw là dict, lấy trường 'answer'
            if isinstance(raw, dict) and "answer" in raw:
                output_text = raw["answer"]
            else:
                output_text = raw
            parsed = self.intent_chain.output_parser.parse(output_text)
            return {
                "intent": parsed.get("intent", "unknown"),
                "confidence": float(parsed.get("confidence", 0.0)),
            }
        except Exception as e:
            logger.error(f"❌ Intent detection lỗi: {e}")
            return {"intent": "unknown", "confidence": 0.0}

    # ----------------------------
    # API: Extract Entities
    # ----------------------------
    def extract_entities(self, question: str, language: str = "vi") -> Dict[str, Any]:
        try:
            result = self.entity_chain.invoke({"question": question, "language": language})
            if isinstance(result, dict) and "answer" in result:
                output_text = result["answer"]
            else:
                output_text = result
            parsed = self.entity_chain.output_parser.parse(output_text)
            return {
                "entities": parsed.get(
                    "entities", {"location": [], "uxo_type": [], "action": []}
                )
            }
        except Exception as e:
            logger.error(f"❌ Entity extraction lỗi: {e}")
            return {"entities": {"location": [], "uxo_type": [], "action": []}}
    # ----------------------------
    # API: Full NLU Pipeline
    # ----------------------------
    def process_nlu(self, question: str, language: str = "vi") -> Dict[str, Any]:
        intent_result = self.detect_intent(question, language)
        entity_result = self.extract_entities(question, language)

        merged = {
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "entities": entity_result["entities"],
        }
        logger.debug(f"✅ Final NLU output: {merged}")
        return merged
