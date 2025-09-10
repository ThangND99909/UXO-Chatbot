from langchain.prompts import PromptTemplate
from typing import Dict, Any, List
from data_layer.hotline_manager import HotlineManager
from ai_core.nlu_processor import NLUProcessor
from ai_core.memory_manager import UXOMemoryManager
import traceback

class UXORetrievalQA:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.hotline_manager = HotlineManager()
        self.memory_manager = UXOMemoryManager()
        # ✅ Nối memory_manager với NLU
        self.nlu_processor = NLUProcessor(llm, memory_manager=self.memory_manager)
        self.setup_qa_chains()
    
    def setup_qa_chains(self):
        # ================= DEFINITION PROMPT =================
        definition_template = """
            Bạn là trợ lý ảo chuyên gia về bom mìn và vật nổ chưa nổ (UXO) tại Việt Nam.
            Dựa trên ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng ngôn ngữ {language}.

            LỊCH SỬ CHAT GẦN ĐÂY:
            {chat_history}

            THÔNG TIN TRA CỨU:
            {context}

            Câu hỏi: {question}

            Hãy trả lời ngắn gọn, chính xác và hữu ích. Nếu không biết câu trả lời, hãy nói không biết.
            Trả lời bằng ngôn ngữ {language}:
            """
        self.definition_prompt = PromptTemplate(
            template=definition_template,
            input_variables=["context", "question", "language", "chat_history"]
        )

        # ================= SAFETY PROMPT =================
        safety_template = """
        Bạn là chuyên gia hướng dẫn an toàn về bom mìn và vật nổ chưa nổ (UXO).
        Dựa trên ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng ngôn ngữ {language}.
        
        LỊCH SỬ CHAT GẦN ĐÂY:
        {chat_history}
        
        THÔNG TIN TRA CỨU:
        {context}
        
        ⚠️ QUAN TRỌNG: 
        - Luôn nhấn mạnh vào việc KHÔNG CHẠM vào vật nghi ngờ.
        - Gọi ngay hotline cơ quan chức năng tại địa phương.
        
        Câu hỏi: {question}
        
        Hãy trả lời rõ ràng, từng bước và an toàn. 
        Luôn cung cấp số hotline nếu có.
        Trả lời bằng ngôn ngữ {language}:
        """
        self.safety_prompt = PromptTemplate(
            template=safety_template,
            input_variables=["context", "question", "language", "chat_history"]
        )

        # ================= LOCATION PROMPT =================
        location_template = """
        Bạn là chuyên gia về thông tin địa điểm liên quan đến bom mìn và UXO tại Việt Nam.
        Dựa trên ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng ngôn ngữ {language}.

        LỊCH SỬ CHAT GẦN ĐÂY:
        {chat_history}

        THÔNG TIN TRA CỨU:
        {context}

        Câu hỏi: {question}

        Hãy cung cấp thông tin chính xác về địa điểm, khu vực, và các thông tin liên quan.
        Nếu có số hotline cụ thể cho khu vực, hãy cung cấp.
        Trả lời bằng ngôn ngữ {language}:
        """
        self.location_prompt = PromptTemplate(
            template=location_template,
            input_variables=["context", "question", "language", "chat_history"]
        )

        # Tạo retriever
        self.retriever = self.vector_store.as_retriever()

    # ================= AI-PROMPT SELECTION =================
    def get_response(self, question: str, intent: str, session_id: str = "default",
                 language: str = "vi", enriched_text: str = None) -> str:
        try:
            chat_history = self.memory_manager.get_chat_history(session_id)
            last_intent = self.memory_manager.get_last_intent(session_id)
            last_question = self.memory_manager.get_last_question(session_id)
            effective_query = enriched_text if enriched_text else question

            print(f"🧠 CONTEXT AWARE: last_intent='{last_intent}', current_intent='{intent}', "
                f"question='{question}', effective_query='{effective_query}'")

            # Lấy tin nhắn cuối của assistant
            last_assistant_msg = ""
            try:
                msgs = self.memory_manager.get_messages(session_id)
                for m in reversed(msgs):
                    if getattr(m, "type", "") != "human":
                        last_assistant_msg = getattr(m, "content", "")
                        break
            except Exception:
                pass

            last_assistant_lc = (last_assistant_msg or "").lower()
            awaiting_hotline = (
                "bạn muốn hỏi số hotline" in last_assistant_lc
                or "số hotline ở khu vực nào" in last_assistant_lc
            )

            # ✅ Case 1: user hỏi trực tiếp
            if intent == "ask_hotline" or self._is_hotline_question(effective_query):
                print("🔍 Hotline request (direct)")
                response = self.process_hotline_request(effective_query, language, session_id)
                self.memory_manager.save_context(session_id, question, response, "ask_hotline")
                return response

            # ✅ Case 2: user trả lời theo ngữ cảnh (bot vừa hỏi tỉnh)
            if last_intent == "ask_hotline" or awaiting_hotline:
                print("⚡ Hotline follow-up (context aware)")
                full_query = f"{last_question} {question}"
                response = self.process_hotline_request(full_query, language, session_id)
                self.memory_manager.save_context(session_id, question, response, "ask_hotline")
                return response

            # ✅ Các intent khác → dùng RAG
            print("🔍 Processing with RAG for non-hotline intent")
            response = self._process_rag_intent(effective_query, intent, session_id, language, chat_history)
            effective_intent = intent or "general"
            self.memory_manager.save_context(session_id, question, response, effective_intent)
            return response

        except Exception as e:
            print(f"❌ Lỗi khi xử lý QA: {str(e)}")
            self.memory_manager.save_context(session_id, question, "Lỗi hệ thống", "error")
            return "Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau."

    def _is_hotline_follow_up(self, question: str) -> bool:
        question_lower = question.lower().strip()
        hotline_keywords = ["hotline", "số điện thoại", "liên hệ", "số máy", "điện thoại", "phone", "gọi"]
        if any(keyword in question_lower for keyword in hotline_keywords):
            return False
        location_keywords = ["quảng bình", "quang binh", "qb", 
                             "quảng trị", "quang tri", "qt",
                             "thừa thiên huế", "thua thien hue", "huế", "hue", "tth",
                             "đà nẵng", "da nang", "dn",
                             "quảng nam", "quang nam", "qn",
                             "nghệ an", "nghe an", "na",
                             "hà tĩnh", "ha tinh", "ht",
                             "thanh hóa", "thanh hoa", "th"]
        has_location = any(loc in question_lower for loc in location_keywords)
        is_short = len(question_lower.split()) <= 5
        return has_location and is_short

    def _is_hotline_question(self, question: str) -> bool:
        question_lower = question.lower()
        hotline_keywords = ["hotline", "số điện thoại", "liên hệ", "số máy", "điện thoại", "phone", "gọi", "đường dây nóng"]
        return any(keyword in question_lower for keyword in hotline_keywords)

    def _process_rag_intent(self, question: str, intent: str, session_id: str, language: str, chat_history: str) -> str:
        try:
            # ✅ enrich cho câu hỏi "ở đâu"
            enriched_query = f"Địa điểm: {question}" if "ở đâu" in question.lower() else question
            docs = self.retriever.get_relevant_documents(enriched_query)
            if not docs:
                return "❌ Tôi không tìm thấy thông tin liên quan trong dữ liệu. Bạn có muốn hỏi lại chi tiết hơn không?"
            context = "\n".join([doc.page_content for doc in docs])

            prompt_mapping = {
                "definition": self.definition_prompt,
                "safety_advice": self.safety_prompt,
                "location_info": self.location_prompt,
                "report_uxo": self.safety_prompt,
                "general": self.definition_prompt
            }
            effective_intent = intent or "general"
            prompt = prompt_mapping.get(effective_intent, self.definition_prompt)

            formatted_prompt = prompt.format(
                context=context,
                question=question,
                language=language,
                chat_history=chat_history
            )

            # ✅ Fix invoke → fallback predict
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(formatted_prompt).strip()
            else:
                response = self.llm.predict(formatted_prompt).strip()
            return response

        except Exception as e:
            print(f"❌ Lỗi khi xử lý RAG: {str(e)}")
            print(traceback.format_exc())
            return "Xin lỗi, tôi gặp sự cố khi tìm thông tin. Vui lòng thử lại sau."

    def extract_location_manual(self, question: str) -> List[str]:
        question_lower = question.lower()
        location_mapping = {
            "quảng bình": "quang_binh", "quang binh": "quang_binh", "qb": "quang_binh",
            "quảng trị": "quang_tri", "quang tri": "quang_tri", "qt": "quang_tri",
            "thừa thiên huế": "thua_thien_hue", "thua thien hue": "thua_thien_hue", 
            "huế": "thua_thien_hue", "hue": "thua_thien_hue", "tth": "thua_thien_hue",
            "đà nẵng": "da_nang", "da nang": "da_nang", "dn": "da_nang",
            "quảng nam": "quang_nam", "quang nam": "quang_nam", "qn": "quang_nam",
            "nghệ an": "nghe_an", "nghe an": "nghe_an", "na": "nghe_an",
            "hà tĩnh": "ha_tinh", "ha tinh": "ha_tinh", "ht": "ha_tinh",
            "thanh hóa": "thanh_hoa", "thanh hoa": "thanh_hoa", "th": "thanh_hoa"
        }
        return [loc for key, loc in location_mapping.items() if key in question_lower]

    def process_hotline_request(self, question: str, language: str, session_id: str = "default") -> str:
        print(f"🔍 Processing hotline request: '{question}'")
        try:
            nlu_result = self.nlu_processor.extract_entities(question, language)
            locations = nlu_result["entities"].get("location", [])
            if not locations:
                locations = self.extract_location_manual(question)

            for location in locations:
                hotline = self.hotline_manager.get_hotline(location)
                if hotline and "Xin lỗi" not in hotline and "không có" not in hotline.lower():
                    return f"📞 Số hotline xử lý bom mìn tại {location.replace('_', ' ').title()} là: {hotline}"

            if not locations:
                return ("❓ Bạn muốn hỏi số hotline ở khu vực nào? "
                        "(Ví dụ: Quảng Bình, Quảng Trị, Huế, Đà Nẵng, Quảng Nam, Nghệ An)")
            return f"❌ Xin lỗi, tôi không có thông tin hotline cho khu vực {locations[0]}."
        except Exception as e:
            print(f"❌ Lỗi khi xử lý hotline: {str(e)}")
            print(traceback.format_exc())
            return "Xin lỗi, tôi gặp sự cố khi tìm số hotline. Vui lòng thử lại sau."
