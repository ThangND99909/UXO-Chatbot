from langchain.prompts import PromptTemplate
from typing import Dict, Any, List
from data_layer.hotline_manager import HotlineManager
from ai_core.nlu_processor import NLUProcessor
from ai_core.memory_manager import UXOMemoryManager
import traceback

class UXORetrievalQA:
    """
    Lớp chính cho mô-đun Hỏi-Đáp (QA) trong chatbot UXO.
    - Gồm 3 tầng chính: NLU → RAG → Memory
    - Kết hợp vector store (retriever) + LLM để sinh câu trả lời.
    - Có xử lý đặc biệt cho các intent như "ask_hotline".
    """
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        # Quản lý hotline theo địa phương
        self.hotline_manager = HotlineManager()
         # Bộ nhớ hội thoại (lưu lịch sử chat, intent, context,...)
        self.memory_manager = UXOMemoryManager()
        # Nối memory_manager với NLU để giữ ngữ cảnh
        self.nlu_processor = NLUProcessor(llm, memory_manager=self.memory_manager)
         # Thiết lập các Prompt Template khác nhau
        self.setup_qa_chains()

    # ========================== TẠO PROMPT CHO TỪNG INTENT ==========================
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

        # ================= LOCATION_INFO PROMPT =================
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

        #Chuyển vector store thành retriever để lấy dữ liệu phù hợp
        self.retriever = self.vector_store.as_retriever()

    # ================= HÀM XỬ LÝ CÂU HỎI =================
    def get_response(self, question: str, intent: str, session_id: str = "default",
                 language: str = "vi", enriched_text: str = None) -> str:
        """
        Chọn cách xử lý câu hỏi tùy intent:
        - ask_hotline → xử lý đặc biệt (truy hotline)
        - các intent khác → dùng RAG để trả lời
        """
        try:
            # Lấy dữ liệu hội thoại từ bộ nhớ
            chat_history = self.memory_manager.get_chat_history(session_id)
            last_intent = self.memory_manager.get_last_intent(session_id)
            last_question = self.memory_manager.get_last_question(session_id)
            effective_query = enriched_text if enriched_text else question

            print(f"CONTEXT AWARE: last_intent='{last_intent}', current_intent='{intent}', "
                f"question='{question}', effective_query='{effective_query}'")

             #Xác định xem user đang hỏi hotline theo ngữ cảnh hay không
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

            # Case 1: user hỏi trực tiếp
            if intent == "ask_hotline" or self._is_hotline_question(effective_query):
                print("Hotline request (direct)")
                response = self.process_hotline_request(effective_query, language, session_id)
                self.memory_manager.save_context(session_id, question, response, "ask_hotline")
                return response

            # Case 2: user trả lời theo ngữ cảnh (bot vừa hỏi tỉnh)
            if last_intent == "ask_hotline" or awaiting_hotline:
                print("Hotline follow-up (context aware)")
                full_query = f"{last_question} {question}"
                response = self.process_hotline_request(full_query, language, session_id)
                self.memory_manager.save_context(session_id, question, response, "ask_hotline")
                return response
            
            # user báo cáo phát hiện bom (report_bomb)
            is_location_desc = self._is_location_description(question)
            has_bomb_reference = self._contains_bomb_reference(question)
            
            print(f"DEBUG: is_location_desc={is_location_desc}, has_bomb_reference={has_bomb_reference}, intent={intent}")

            # Xử lý mọi câu mô tả địa điểm CÓ hoặc KHÔNG có tham chiếu bom
            if is_location_desc:
                print("📍 Location description detected - checking context")
                
                # TRƯỜNG HỢP 1: Địa điểm CÓ tham chiếu bom → cảm ơn và xác nhận
                if has_bomb_reference:
                    print("📍 User providing bomb location - sending confirmation")
                    if language == "vi":
                        response = f"""Cảm ơn bạn đã cung cấp thông tin địa điểm!

                            Chúng tôi đã ghi nhận địa điểm: **"{question}"**

                            Đội ngũ chuyên gia sẽ đến địa điểm bạn cung cấp để xác minh và xử lý. Vui lòng giữ khoảng cách an toàn và không tự ý xử lý.

                            📞 Nếu có thông tin khẩn cấp, hãy gọi ngay:
                            • MAG Vietnam: 0914 555 247 / 0913 888 27
                            • Quân đội địa phương: 113
                            • Công an: 113"""
                    else:
                        response = f"""Thank you for providing the location information!

                            We have recorded the location: **"{question}"**

                            Our expert team will come to the location you provided for verification and handling. Please maintain a safe distance and do not handle it yourself.

                            📞 For emergencies, call immediately:
                            • MAG Vietnam: 0914 555 247 / 0913 888 27  
                            • Local Army: 113
                            • Police: 113"""
                    
                    self.memory_manager.save_context(session_id, question, response, "location_confirmation")
                    return response
                
                # TRƯỜNG HỢP 2: Địa điểm KHÔNG có tham chiếu bom, nhưng intent là report_uxo → vẫn cảm ơn
                elif intent in ["report_uxo", "report_bomb"]:
                    print("📍 Location with report intent - sending confirmation")
                    if language == "vi":
                        response = f"""Cảm ơn bạn đã cung cấp thông tin địa điểm!

                            Chúng tôi đã ghi nhận địa điểm: **"{question}"**

                            Đội ngũ chuyên gia sẽ đến địa điểm bạn cung cấp để xác minh và xử lý. Vui lòng giữ khoảng cách an toàn và không tự ý xử lý.

                            📞 Nếu có thông tin khẩn cấp, hãy gọi ngay:
                            • MAG Vietnam: 0914 555 247 / 0913 888 27
                            • Quân đội địa phương: 113
                            • Công an: 113"""
                    else:
                        response = f"""Thank you for providing the location information!

                            We have recorded the location: **"{question}"**

                            Our expert team will come to the location you provided for verification and handling. Please maintain a safe distance and do not handle it yourself.

                            📞 For emergencies, call immediately:
                            • MAG Vietnam: 0914 555 247 / 0913 888 27  
                            • Local Army: 113
                            • Police: 113"""
                    
                    self.memory_manager.save_context(session_id, question, response, "location_confirmation")
                    return response
            if intent == "report_bomb" and not is_location_desc:
                print("🚨 Report bomb intent detected - returning special response")
                if language == "vi":
                    response = """Nếu bạn nhìn thấy một quả bom trên đường, tuyệt đối không lại gần, không chạm vào và giữ khoảng cách an toàn.

                        Hãy báo ngay cho chính quyền địa phương gần nhất (công an, quân đội hoặc ủy ban nhân dân xã/phường) để họ có biện pháp xử lý kịp thời và đảm bảo an toàn cho mọi người.

                        📍 **Bạn có thể gửi địa điểm chính xác bằng cách:**
                        • Click vào bản đồ bên dưới để đánh dấu vị trí bạn nhìn thấy quả bom
                        • Hoặc mô tả địa điểm chi tiết trong phần chat"""
                else:
                    response = """If you see a bomb on the road, absolutely do not approach, do not touch it, and maintain a safe distance.

                        Report immediately to the nearest local authorities (police, military, or local people's committee) so they can take timely measures and ensure everyone's safety.

                        📍 **You can report the exact location by:**
                        • Clicking on the map below to mark the location where you saw the bomb
                        • Or describing the location in detail in the chat section"""
                
                self.memory_manager.save_context(session_id, question, response, "report_bomb")
                return response

            # Các intent khác → dùng RAG
            print("Processing with RAG for non-hotline intent")
            response = self._process_rag_intent(effective_query, intent, session_id, language, chat_history)
            effective_intent = intent or "general"
            self.memory_manager.save_context(session_id, question, response, effective_intent)
            return response

        except Exception as e:
            print(f"Lỗi khi xử lý QA: {str(e)}")
            self.memory_manager.save_context(session_id, question, "Lỗi hệ thống", "error")
            return "Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau."

    # ========================== HÀM KIỂM TRA NGỮ CẢNH HỎI HOTLINE ==========================
    def _is_hotline_follow_up(self, question: str) -> bool:
        """
        Kiểm tra xem người dùng có đang hỏi *tiếp theo* về hotline hay không.
        """
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

    # ========================== HÀM NHẬN DIỆN HOTLINE ==========================
    def _is_hotline_question(self, question: str) -> bool:
        question_lower = question.lower()
        hotline_keywords = ["hotline", "số điện thoại", "liên hệ", "số máy", "điện thoại", "phone", "gọi", "đường dây nóng"]
        return any(keyword in question_lower for keyword in hotline_keywords)

    # ========================== HÀM XỬ LÝ INTENT BẰNG RAG ==========================
    def _process_rag_intent(self, question: str, intent: str, session_id: str, language: str, chat_history: str) -> str:
        try:
            # enrich cho câu hỏi "ở đâu"
            enriched_query = f"Địa điểm: {question}" if "ở đâu" in question.lower() else question
            docs = self.retriever.get_relevant_documents(enriched_query)
            if not docs:
                return "Tôi không tìm thấy thông tin liên quan trong dữ liệu. Bạn có muốn hỏi lại chi tiết hơn không?"
            # Gom nội dung context từ các document
            context = "\n".join([doc.page_content for doc in docs])

            # Ánh xạ intent → prompt tương ứng
            prompt_mapping = {
                "definition": self.definition_prompt,
                "safety_advice": self.safety_prompt,
                "location_info": self.location_prompt,
                "report_uxo": self.safety_prompt,
                "report_bomb": self.safety_prompt,
                "general": self.definition_prompt
            }
            effective_intent = intent or "general"
            prompt = prompt_mapping.get(effective_intent, self.definition_prompt)

            # Format prompt với dữ liệu thật
            formatted_prompt = prompt.format(
                context=context,
                question=question,
                language=language,
                chat_history=chat_history
            )

            # Fix invoke → fallback predict
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(formatted_prompt).strip()
            else:
                response = self.llm.predict(formatted_prompt).strip()
            return response

        except Exception as e:
            print(f"Lỗi khi xử lý RAG: {str(e)}")
            print(traceback.format_exc())
            return "Xin lỗi, tôi gặp sự cố khi tìm thông tin. Vui lòng thử lại sau."

    # ========================== HÀM TRÍCH XUẤT ĐỊA ĐIỂM THỦ CÔNG ==========================
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

    # ========================== HÀM XỬ LÝ HOTLINE ==========================
    def process_hotline_request(self, question: str, language: str, session_id: str = "default") -> str:
        """
        Gọi NLU để trích xuất location từ câu hỏi → lấy số hotline tương ứng.
        """
        print(f"🔍 Processing hotline request: '{question}'")
        try:
            # Bước 1: Trích xuất thực thể địa điểm bằng NLU
            nlu_result = self.nlu_processor.extract_entities(question, language)
            locations = nlu_result["entities"].get("location", [])
            if not locations:
                locations = self.extract_location_manual(question)
            # Bước 2: Tra hotline theo từng địa phương
            for location in locations:
                hotline = self.hotline_manager.get_hotline(location)
                if hotline and "Xin lỗi" not in hotline and "không có" not in hotline.lower():
                    return f"📞 Số hotline xử lý bom mìn tại {location.replace('_', ' ').title()} là: {hotline}"
            # Bước 3: Nếu không xác định được địa phương
            if not locations:
                return ("❓ Bạn muốn hỏi số hotline ở khu vực nào? "
                        "(Ví dụ: Quảng Bình, Quảng Trị, Huế, Đà Nẵng, Quảng Nam, Nghệ An)")
            return f"❌ Xin lỗi, tôi không có thông tin hotline cho khu vực {locations[0]}."
        except Exception as e:
            print(f"❌ Lỗi khi xử lý hotline: {str(e)}")
            print(traceback.format_exc())
            return "Xin lỗi, tôi gặp sự cố khi tìm số hotline. Vui lòng thử lại sau."
    # ========================== HÀM KIỂM TRA MÔ TẢ ĐỊA ĐIỂM ==========================    
    def _is_location_description(self, question: str) -> bool:
        """
        Kiểm tra xem câu có phải là mô tả địa điểm chi tiết không
        """
        question_lower = question.lower().strip()
        
        # Từ khóa cho thấy đây là mô tả địa điểm
        location_keywords = [
            "vị trí", "vi tri", "ở", "o", "tại", "địa điểm", "dia diem",
            "địa chỉ", "dia chi", "cánh đồng", "canh dong", "cạnh", "canh",
            "gần", "gan", "trước", "truoc", "sau", "bên", "ben", 
            "đường", "duong", "phố", "pho", "xã", "xa", "thôn", "thon",
            "ấp", "ap", "khu vực", "khu vuc", "khu", "trạm", "tram",
            "cầu", "cau", "chợ", "cho", "trường", "truong", "bệnh viện", "benh vien",
            "đồi", "doi", "ruộng", "ruong", "núi", "nui", "sông", "song"
        ]
        
        # Từ khóa cho thấy đây là báo cáo mới
        report_keywords = [
            "thấy", "thay", "nhìn thấy", "nhin thay", "phát hiện", "phat hien",
            "gặp", "gap", "có", "co", "một", "mot", "quả", "qua", "trái", "trai",
            "tôi", "toi", "tui", "em", "tớ", "to", "mình", "minh"
        ]
        
        # THÊM: Từ khóa cho thấy đây là câu hỏi định nghĩa (KHÔNG phải location)
        question_keywords = [
            "là gì", "la gi", "gì", "gi", "gì vậy", "gi vay", "?", "bao nhiêu", "bao nhieu",
            "thế nào", "the nao", "tại sao", "tai sao", "vì sao", "vi sao"
        ]
        
        has_location = any(keyword in question_lower for keyword in location_keywords)
        has_report = any(keyword in question_lower for keyword in report_keywords)
        has_question = any(keyword in question_lower for keyword in question_keywords)
        
        print(f"DEBUG LOCATION: has_location={has_location}, has_report={has_report}, has_question={has_question}")
        
        # Nếu có từ nghi vấn → KHÔNG phải location description
        if has_question:
            print("🔍 DEBUG: Contains question word - NOT location")
            return False
        
        # Nếu có từ địa điểm và KHÔNG có từ báo cáo → đây là mô tả địa điểm
        if has_location and not has_report:
            print("🔍 DEBUG: Pure location description")
            return True
        
        # Nếu có cả từ địa điểm và từ báo cáo
        if has_location and has_report:
            # Kiểm tra xem có phải là cấu trúc "vị trí + tên cụ thể" không
            import re
            location_patterns = [
                r"vị trí\s+\w+", r"vi tri\s+\w+", r"ở\s+\w+", r"o\s+\w+", 
                r"tại\s+\w+", r"địa điểm\s+\w+", r"dia diem\s+\w+",
                r"đồi\s+\w+", r"doi\s+\w+", r"ruộng\s+\w+", r"ruong\s+\w+"
            ]
            
            has_specific_location = any(re.search(pattern, question_lower) for pattern in location_patterns)
            print(f"DEBUG: has_specific_location={has_specific_location}")
            
            # Nếu có địa điểm cụ thể → coi như đã cung cấp địa điểm
            if has_specific_location:
                return True
        
        # Nếu câu bắt đầu bằng từ địa điểm
        starts_with_location = any(question_lower.startswith(prefix) for prefix in 
                                ["vị trí", "vi tri", "ở", "o", "tại", "địa điểm", "dia diem"])
        
        result = (has_location and not has_report) or starts_with_location
        print(f"DEBUG LOCATION RESULT: {result}")
        return result
    # ========================== HÀM KIỂM TRA THAM CHIẾU BOM ==========================
    def _contains_bomb_reference(self, question: str) -> bool:
        """
        Kiểm tra xem câu có tham chiếu đến bom/mìn không
        """
        question_lower = question.lower()
        bomb_keywords = [
            "bom", "mìn", "min", "uxo", "vật nổ", "vat no", 
            "quả nổ", "qua no", "trái nổ", "trai no",
            "quả bom", "qua bom", "trái bom", "trai bom",
            "đạn", "dan", "lựu đạn", "luu dan", "mìn", "min"
        ]
        
        # LOẠI TRỪ: Nếu "bom" đứng cùng từ nghi vấn → không phải tham chiếu bom
        question_patterns = [r"bom.*là gì", r"bom.*la gi", r"bom.*gì", r"bom.*gi", r"bom\?"]
        import re
        is_question_about_bomb = any(re.search(pattern, question_lower) for pattern in question_patterns)
        
        if is_question_about_bomb:
            print("DEBUG: Question about bomb definition - NOT bomb reference")
            return False
        
        return any(keyword in question_lower for keyword in bomb_keywords)
