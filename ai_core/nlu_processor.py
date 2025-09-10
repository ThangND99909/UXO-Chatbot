# ai_core/nlu_processor.py
import json
import re
import logging
import unicodedata
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
# Context Memory (simple)
# ========================
class ContextMemory:
    def __init__(self):
        self.last_intent = None
        self.last_entities = {}

    def update(self, intent: str, entities: Dict[str, Any]):
        if intent and intent != "unknown":
            self.last_intent = intent
        if entities:
            self.last_entities = entities

    def get_context(self) -> Dict[str, Any]:
        return {
            "last_intent": self.last_intent,
            "last_entities": self.last_entities,
        }
    
# ========================
# Helpers (NEW)
# ========================
def _strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt để so khớp keyword dễ hơn."""
    if not isinstance(s, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _contains_any(haystack: str, needles) -> bool:
    return any(n in haystack for n in needles)

# Các từ gợi ý câu hỏi KHÔNG phải hotline (để không ép về hotline)
QUESTION_TRIGGERS = [
    "ở đâu", "o dau", "là gì", "la gi", "giới thiệu", "gioi thieu",
    "thông tin", "thong tin", "bao nhiêu", "bao nhieu", "vì sao", "vi sao", "?"
]

# Từ khoá hotline
HOTLINE_KEYWORDS = [
    "hotline", "số điện thoại", "so dien thoai", "điện thoại", "dien thoai",
    "đường dây nóng", "duong day nong", "gọi", "goi", "liên hệ", "lien he"
]

# Danh sách địa danh phổ biến (có cả có dấu & không dấu, để match nhanh)
LOCATION_TOKENS = [
    "quảng bình","quang binh","qb",
    "quảng trị","quang tri","qt",
    "thừa thiên huế","thua thien hue","huế","hue","tth",
    "đà nẵng","da nang","dn",
    "quảng nam","quang nam","qn",
    "nghệ an","nghe an","na",
    "hà tĩnh","ha tinh","ht",
    "thanh hóa","thanh hoa","th",
    # có thể bổ sung thêm...
]
# ========================
# NLU Processor
# ========================
class NLUProcessor:
    def __init__(self, llm=None, memory_manager=None):
        """
        llm: object LLM, nếu None sẽ tự khởi tạo GeminiLLM mặc định
        memory_manager: để truy cập last_intent, last_question, chat_history
        """
        self.llm = llm or GeminiLLM()
        self.memory_manager = memory_manager
        self.memory = ContextMemory()  # ✅ thêm bộ nhớ ngữ cảnh
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
        - location_info: hỏi về thông tin địa điểm (ví dụ: "Quảng Trị có gì đặc biệt?")
        - report_uxo: báo cáo vật nổ
        - ask_hotline: hỏi số hotline (ví dụ: "số điện thoại Quảng Trị", "hotline ở đâu?")
        - general: câu hỏi chung khác

        PHÂN BIỆT QUAN TRỌNG:
        - "Quảng Trị" → location_info (nếu chỉ là tên địa điểm không ngữ cảnh)
        - "số điện thoại Quảng Trị" → ask_hotline
        - "hotline Quảng Trị" → ask_hotline

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

    def _get_last_assistant_message(self, session_id: str) -> str:
        """🔹 NEW: lấy tin nhắn assistant gần nhất (nếu có) để biết có đang hỏi xin địa danh hotline không."""
        if not self.memory_manager:
            return ""
        try:
            msgs = self.memory_manager.get_messages(session_id)
            for m in reversed(msgs):
                # Trong memory_manager của bạn, msg.type == "human"/"ai"
                if hasattr(m, "type") and m.type != "human":
                    return getattr(m, "content", "") or ""
        except Exception as e:
            logger.debug(f"⚠️ Không lấy được last assistant message: {e}")
        return ""

    def detect_intent(self, question: str, language: str = "vi", session_id: str = "default") -> Dict[str, Any]:
        try:
            raw = self.intent_chain.invoke({"question": question, "language": language})
            output_text = raw["answer"] if isinstance(raw, dict) and "answer" in raw else raw
            if isinstance(output_text, dict):
                output_text = json.dumps(output_text, ensure_ascii=False)
            parsed = self.intent_chain.output_parser.parse(output_text)

            intent = parsed.get("intent", "unknown")
            confidence = float(parsed.get("confidence", 0.0))
            enriched_text = None

            # 🔹 Context-based refinement (STRICT)
            last_intent, last_question = "", ""
            awaiting_hotline_location = False

            if self.memory_manager:
                last_intent = self.memory_manager.get_last_intent(session_id)
                last_question = self.memory_manager.get_last_question(session_id)

                # Nhận diện xem bot có đang hỏi người dùng "hotline ở khu vực nào?" không
                last_bot = self._get_last_assistant_message(session_id)
                last_bot_lc = _strip_accents((last_bot or "").lower())
                if "hotline o khu vuc nao" in last_bot_lc or "ban muon hoi so hotline" in last_bot_lc:
                    awaiting_hotline_location = True

            # Chuẩn hoá text để so khớp
            t_lc = question.strip().lower()
            t_ascii = _strip_accents(t_lc)
            words = t_ascii.split()
            is_short = len(words) <= 4

            has_question_word = _contains_any(t_lc, QUESTION_TRIGGERS) or _contains_any(t_ascii, QUESTION_TRIGGERS)
            has_hotline_kw = _contains_any(t_lc, HOTLINE_KEYWORDS) or _contains_any(t_ascii, HOTLINE_KEYWORDS)
            has_location = any(tok in t_lc or tok in t_ascii for tok in LOCATION_TOKENS)

            # 🎯 QUY TẮC MỚI (fix bug bạn gặp):
            # Chỉ ép về ask_hotline khi:
            #  - ĐANG CHỜ địa danh cho hotline (bot vừa hỏi khu vực) HOẶC last_intent là ask_hotline
            #  - Câu rất ngắn & CHỈ là địa danh (không có từ nghi vấn/miêu tả)
            #  - Không chứa từ khoá hotline (vì khi đó intent đã là ask_hotline tự nhiên)
            if (
                has_location
                and is_short
                and not has_question_word
                and not has_hotline_kw
                and (awaiting_hotline_location or last_intent == "ask_hotline")
            ):
                logger.debug("⚡ Follow-up hotline hợp lệ: ép intent = ask_hotline")
                intent = "ask_hotline"
                enriched_text = f"{last_question} {question}" if last_question else f"hotline {question}"
            else:
                # Nếu câu hỏi có từ nghi vấn như 'ở đâu', 'là gì'... thì KHÔNG ép hotline
                logger.debug("ℹ️ Không ép intent về hotline (giữ theo LLM hoặc suy luận thường).")

            # Trường hợp câu cực ngắn (<=2 từ) → enrich text để RAG hiểu hơn
            if not enriched_text and last_question and len(words) <= 2:
                enriched_text = f"{last_question} {question}"
                logger.debug(f"🧩 Enriched text (câu cực ngắn): {enriched_text}")

            return {
                "intent": intent,
                "confidence": confidence,
                "last_intent": last_intent,
                "last_question": last_question,
                "awaiting_hotline_location": awaiting_hotline_location,
                "enriched_text": enriched_text
            }

        except Exception as e:
            logger.error(f"❌ Intent detection lỗi: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "last_intent": "",
                "last_question": "",
                "awaiting_hotline_location": False,
                "enriched_text": None
            }
    # ----------------------------
    # API: Extract Entities
    # ----------------------------
    def extract_entities(self, question: str, language: str = "vi") -> Dict[str, Any]:
        try:
            result = self.entity_chain.invoke({"question": question, "language": language})
            output_text = result["answer"] if isinstance(result, dict) and "answer" in result else result
            if isinstance(output_text, dict):
                output_text = json.dumps(output_text, ensure_ascii=False)
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
    def process_nlu(self, question: str, language: str = "vi", session_id: str = "default") -> Dict[str, Any]:
        intent_result = self.detect_intent(question, language, session_id)
        entity_result = self.extract_entities(question, language)

        merged = {
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "entities": entity_result["entities"],
            "last_intent": intent_result.get("last_intent", ""),
            "last_question": intent_result.get("last_question", ""),
            "awaiting_hotline_location": intent_result.get("awaiting_hotline_location", False),
            "enriched_text": intent_result.get("enriched_text")
        }
        logger.debug(f"✅ Final NLU output: {merged}")
        return merged
