from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from typing import Dict, List

class UXOMemoryManager:
    def __init__(self, k=3):
        self.memories: Dict[str, ConversationBufferWindowMemory] = {}  # Lưu trữ memory cho từng session
        self.last_intents: Dict[str, str] = {}  # 🔹 Lưu intent cuối cùng riêng
        self.last_questions: Dict[str, str] = {}   # 🔹 NEW: Lưu câu hỏi cuối
        self.k = k  # Số lượng tin nhắn lưu trong memory
    
    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Lấy memory cho session, tạo mới nếu chưa có"""
        if session_id not in self.memories: 
            memory = ConversationBufferWindowMemory( 
                k=self.k, return_messages=True, memory_key="chat_history", output_key="output" ) 
            self.memories[session_id] = memory 
        return self.memories[session_id]
    
    def save_context(self, session_id: str, user_input: str, assistant_output: str, intent: str = None):
        """Lưu ngữ cảnh hội thoại kèm intent"""
        # Lấy bộ nhớ (ConversationBufferWindowMemory) cho session hiện tại
        # Nếu chưa tồn tại thì get_memory sẽ tạo mới và lưu trong self.memories
        memory = self.get_memory(session_id)

        # Gọi hàm save_context của memory để lưu lại cặp (input, output) vào RAM
        # - {"input": user_input}  : câu hỏi/đầu vào của User
        # - {"output": assistant_output} : câu trả lời/đầu ra của Assistant
        #memory.save_context không phải là đệ quy chỉ gọi tiếp memory.save_context
        #đây là method của một object khác (ConversationBufferWindowMemory), không phải chính nó.
        memory.save_context(
            {"input": user_input},      # ✅ input format chuẩn (dict)
            {"output": assistant_output}  # ✅ output format chuẩn (dict)
        )

        # Nếu có intent (ý định của người dùng) thì lưu lại riêng
        # Dùng session_id làm key để mỗi phiên có intent riêng
        if intent:
            self.last_intents[session_id] = intent

        # Luôn luôn lưu câu hỏi cuối cùng của User để tiện xử lý về sau
        self.last_questions[session_id] = user_input   # 🔹 Lưu câu hỏi cuối
        print(f"💾 Saved context: {user_input[:50]}... -> {assistant_output[:50]}... | intent={intent}")

    def get_chat_history(self, session_id: str) -> str:
        """Lấy lịch sử chat dạng text"""
        """Lấy memory từ self.memories[session_id].
        Gọi memory.load_memory_variables({}).
        Trả về list message trong RAM.
        Toàn bộ hội thoại (input, output) được lưu trong RAM → cụ thể là trong list memory.chat_memory.messages."""
        try:
            memory = self.get_memory(session_id)
            memory_vars = memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            
            if not chat_history:  # ✅ tránh lỗi khi chưa có gì
                return ""
            history_text = ""
            for msg in chat_history:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = "User" if msg.type == "human" else "Assistant"
                    history_text += f"{role}: {msg.content}\n"
            
            print(f"📝 Chat history for {session_id}: {history_text}")
            return history_text
            
        except Exception as e:
            print(f"❌ Error getting chat history: {e}")
            return ""
    
    def clear_memory(self, session_id: str):
        """Xóa memory của session"""
        if session_id in self.memories:
            del self.memories[session_id]
        if session_id in self.last_intents:
            del self.last_intents[session_id]

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """Trả về danh sách message trong session"""
        memory = self.get_memory(session_id)
        return memory.chat_memory.messages  

    def get_last_intent(self, session_id: str) -> str:
        return self.last_intents.get(session_id, "general")
    
    def get_last_question(self, session_id: str) -> str:
        """Lấy câu hỏi cuối cùng của user"""
        return self.last_questions.get(session_id, "")
    

'''User input (câu hỏi) ────────────────┐
                                      │
Assistant output (trả lời) ───────────┤
                                      ▼
        save_context(session_id, user_input, assistant_output)
                                      │
                                      ▼
                        get_memory(session_id)
                                      │
                                      ▼
          self.memories[session_id]  (dict lưu các memory theo session)
                │
                └── Nếu chưa có:
                        Tạo ConversationBufferWindowMemory
                        gán vào self.memories[session_id]
                                      │
                                      ▼
                memory.save_context({"input": ...}, {"output": ...})
                                      │
                                      ▼
                memory.chat_memory.messages (list trong RAM)
                     ┌─────────────────────────────────┐
                     │ [HumanMessage(content="..."),    │
                     │  AIMessage(content="..."), ...]  │'''
