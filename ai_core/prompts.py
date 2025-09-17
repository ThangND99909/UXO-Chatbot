from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Prompt cho Intent Detection
intent_system = """
Bạn là một trợ lý AI chuyên phân tích ý định người dùng.
Các intent có thể là:
- definition: hỏi về định nghĩa, khái niệm
- safety_advice: hỏi về hướng dẫn an toàn
- location_info: hỏi về thông tin địa điểm
- report_uxo: báo cáo vật nổ
- ask_hotline: hỏi số hotline
- general: câu hỏi chung khác
"""

intent_human = """
Câu hỏi: {question}
Ngôn ngữ: {language}

Trả về JSON với cấu trúc:
{{"intent": "tên_intent", "confidence": độ_tin_cậy (0-1)}}
"""

intent_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(intent_system),
    HumanMessagePromptTemplate.from_template(intent_human)
])

# Prompt cho Entity Extraction
entity_system = """
Bạn là một trợ lý AI chuyên trích xuất thực thể từ câu hỏi.
Các loại thực thể cần trích xuất:
- location: địa điểm, tỉnh thành
- uxo_type: loại vật nổ (bom, mìn, lựu đạn, etc.)
- action: hành động
"""

entity_human = """
Câu hỏi: {question}
Ngôn ngữ: {language}

Trả về JSON với cấu trúc:
{{"entities": {{"location": ["..."], "uxo_type": ["..."], "action": ["..."]}}}}
"""

entity_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(entity_system),
    HumanMessagePromptTemplate.from_template(entity_human)
])
