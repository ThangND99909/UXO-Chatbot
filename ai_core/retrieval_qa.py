from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import Dict, Any
from data_layer.hotline_manager import HotlineManager

class UXORetrievalQA:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.hotline_manager = HotlineManager()
        self.setup_qa_chains()
    
    def setup_qa_chains(self):
        # ================= DEFINITION PROMPT =================
        definition_template = """
            Bạn là trợ lý ảo chuyên gia về bom mìn và vật nổ chưa nổ (UXO) tại Việt Nam.
            Dựa trên ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng ngôn ngữ {language}.

            Ngữ cảnh:
            {context}

            Câu hỏi: {question}

            Hãy trả lời ngắn gọn, chính xác và hữu ích. Nếu không biết câu trả lời, hãy nói không biết.
            Trả lời bằng ngôn ngữ {language}:
            """
        self.definition_prompt = PromptTemplate(
            template=definition_template,
            input_variables=["context", "question", "language"]
        )

        # ================= SAFETY PROMPT =================
        safety_template = """
        Bạn là chuyên gia hướng dẫn an toàn về bom mìn và vật nổ chưa nổ (UXO).
        Dựa trên ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng ngôn ngữ {language}.
        
        ⚠️ QUAN TRỌNG: 
        - Luôn nhấn mạnh vào việc KHÔNG CHẠM vào vật nghi ngờ.
        - Gọi ngay hotline cơ quan chức năng tại địa phương.
        
        Ngữ cảnh:
        {context}
        
        Câu hỏi: {question}
        
        Hãy trả lời rõ ràng, từng bước và an toàn. 
        Luôn cung cấp số hotline nếu có.
        Trả lời bằng ngôn ngữ {language}:
        """
        self.safety_prompt = PromptTemplate(
            template=safety_template,
            input_variables=["context", "question", "language"]
        )

        # ================= LOCATION PROMPT =================
        location_template = """
        Bạn là chuyên gia về thông tin địa điểm liên quan đến bom mìn và UXO tại Việt Nam.
        Dựa trên ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng ngôn ngữ {language}.

        Ngữ cảnh:
        {context}

        Câu hỏi: {question}

        Hãy cung cấp thông tin chính xác về địa điểm, khu vực, và các thông tin liên quan.
        Nếu có số hotline cụ thể cho khu vực, hãy cung cấp.
        Trả lời bằng ngôn ngữ {language}:
        """
        self.location_prompt = PromptTemplate(
            template=location_template,
            input_variables=["context", "question", "language"]
        )

        # Tạo retriever
        self.retriever = self.vector_store.as_retriever()

    # ================= AI-PROMPT SELECTION =================
    def get_response(self, question: str, intent: str, language: str = "vi") -> str:
        try:
            # ✅ Nếu là hotline thì xử lý riêng, không cần retriever
            if intent == "ask_hotline":
                from ai_core.nlu_processor import NLUProcessor
                nlu = NLUProcessor(self.llm)
                nlu_result = nlu.extract_entities(question, language)
                locations = nlu_result["entities"].get("location", [])
                if locations:
                    return self.hotline_manager.get_hotline(locations[0])
                else:
                    return "Bạn muốn hỏi số hotline ở khu vực nào?"

            docs = self.retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Prompt để LLM tự chọn template phù hợp
            selector_prompt = f"""
            Phân tích intent và chọn template phù hợp:
            Intent: {intent}
            Câu hỏi: {question}
            
            Các template có sẵn:
            - definition_prompt: cho câu hỏi về định nghĩa, thông tin chung, khái niệm
            - safety_prompt: cho câu hỏi an toàn, khẩn cấp, hotline, hướng dẫn hành động
            - location_prompt: cho câu hỏi về địa điểm, khu vực, địa bàn
            
            Chỉ trả về tên template (definition_prompt, safety_prompt, hoặc location_prompt)
            Không thêm bất kỳ text nào khác.
            """
            
            # LLM chọn template
            selected_template = self.llm.invoke(selector_prompt).strip().lower()
            print(f"🔍 AI selected template: {selected_template}")  # Debug
            
            # Map đến prompt thực tế - THÊM TẤT CẢ PROMPTS
            prompt_mapping = {
                "definition_prompt": self.definition_prompt,
                "safety_prompt": self.safety_prompt,
                "location_prompt": self.location_prompt,
            }
            
            # Fallback nếu template không tồn tại
            prompt = prompt_mapping.get(selected_template, self.safety_prompt)
            
            formatted_prompt = prompt.format(
                context=context,
                question=question,
                language=language
            )
            
            response = self.llm.invoke(formatted_prompt)
            return response.strip()
            
        except Exception as e:
            return f"❌ Lỗi khi xử lý QA: {e}"

    # ================= PHƯƠNG THỨC DỰ PHÒNG =================
    def get_response_fallback(self, question: str, intent: str, language: str = "vi") -> str:
        """Phương thức fallback đơn giản nếu AI selection gặp lỗi"""
        try:
            docs = self.retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Logic chọn prompt đơn giản
            if "hotline" in intent or "report" in intent or "emergency" in intent or "safety" in intent:
                prompt = self.safety_prompt
            elif "location" in intent or "area" in intent or "where" in intent:
                prompt = self.location_prompt
            else:
                prompt = self.definition_prompt
            
            formatted_prompt = prompt.format(
                context=context,
                question=question,
                language=language
            )
            
            response = self.llm.invoke(formatted_prompt)
            return response.strip()
            
        except Exception as e:
            return f"❌ Lỗi khi xử lý QA: {e}"