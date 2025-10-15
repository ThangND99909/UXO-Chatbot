from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Prompt cho Intent Detection
# Phần "System prompt" mô tả vai trò và hướng dẫn hành vi của LLM
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

# Phần "Human prompt" chứa dữ liệu đầu vào động (question + language)
intent_human = """
Câu hỏi: {question}
Ngôn ngữ: {language}

Trả về JSON với cấu trúc:
{{"intent": "tên_intent", "confidence": độ_tin_cậy (0-1)}}
"""

# Gộp 2 phần trên thành ChatPromptTemplate (dạng hội thoại system + user)
intent_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(intent_system), # Hướng dẫn cho model
    HumanMessagePromptTemplate.from_template(intent_human) # Câu hỏi thực tế từ người dùng
])

# Prompt cho Entity Extraction
# Phần "System prompt" cho model biết nhiệm vụ là trích xuất thực thể
entity_system = """
Bạn là một trợ lý AI chuyên trích xuất thực thể từ câu hỏi.
Các loại thực thể cần trích xuất:
- location: địa điểm, tỉnh thành
- uxo_type: loại vật nổ (bom, mìn, lựu đạn, etc.)
- action: hành động
"""

# "Human prompt" mô tả input và định dạng output JSON mong muốn
entity_human = """
Câu hỏi: {question}
Ngôn ngữ: {language}

Trả về JSON với cấu trúc:
{{"entities": {{"location": ["..."], "uxo_type": ["..."], "action": ["..."]}}}}
"""

# Gộp system + human prompt thành 1 template hội thoại
entity_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(entity_system),
    HumanMessagePromptTemplate.from_template(entity_human)
])
