# ai_core/parsers.py
import json, re, logging
from typing import Dict, Any
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)

class NLUOutputParser(BaseOutputParser):
    """Parser an toàn cho output từ LLM (JSON -> Dict)"""

    def parse(self, text: str) -> Dict[str, Any]:
        try:
            logger.debug(f"🔹 Raw LLM output: {text}")
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
        return "Trả lời dưới dạng JSON hợp lệ."
