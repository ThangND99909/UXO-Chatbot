from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class UXORetrievalQA:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.setup_qa_chains()
    
    def setup_qa_chains(self):
        definition_template = """
            Bạn là trợ lý ảo chuyên gia về bom mìn và vật nổ chưa nổ (UXO) tại Việt Nam.
            Dựa trên ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng ngôn ngữ {language}.

            Ngữ cảnh:
            {context}

            Câu hỏi: {query}

            Hãy trả lời ngắn gọn, chính xác và hữu ích. Nếu không biết câu trả lời, hãy nói không biết.
            Trả lời bằng ngôn ngữ {language}:
            """
        self.definition_prompt = PromptTemplate(
            template=definition_template,
            input_variables=["context", "query", "language"]
        )

        safety_template = """
        Bạn là chuyên gia hướng dẫn an toàn về bom mìn và vật nổ chưa nổ (UXO).
        Dựa trên ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng ngôn ngữ {language}.
        
        ⚠️ QUAN TRỌNG: 
        - Luôn nhấn mạnh vào việc KHÔNG CHẠM vào vật nghi ngờ.
        - Gọi ngay hotline cơ quan chức năng tại địa phương.
        
        Ngữ cảnh:
        {context}
        
        Câu hỏi: {query}
        
        Hãy trả lời rõ ràng, từng bước và an toàn. 
        Luôn cung cấp số hotline nếu có.
        Trả lời bằng ngôn ngữ {language}:
        """
        self.safety_prompt = PromptTemplate(
            template=safety_template,
            input_variables=["context", "query", "language"]
        )

        # Chain
        self.definition_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": self.definition_prompt},
            return_source_documents=False
        )

        self.safety_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": self.safety_prompt},
            return_source_documents=False
        )

    # ----------------------------
    # Get response theo intent
    # ----------------------------
    def get_response(self, question: str, intent: str, language: str = "vi") -> str:
        inputs = {"query": question, "language": language}

        # Map intent sang chain
        intent_to_chain = {
            "definition": self.definition_qa,
            "safety_advice": self.safety_qa,
            "ask_hotline": self.safety_qa,  # map tạm để trả lời hotline
        }

        qa_chain = intent_to_chain.get(intent)
        if not qa_chain:
            return "❌ Không hiểu ý định của bạn. Vui lòng thử lại."

        try:
            result = qa_chain(inputs)
            answer = (
                result.get("result")
                or result.get("answer")
                or result.get("output")
            )
            if answer:
                return answer.strip()
            else:
                return f"⚠️ Không tìm thấy câu trả lời hợp lệ. Debug outputs: {result}"
        except Exception as e:
            return f"❌ Lỗi khi xử lý QA: {e}"
