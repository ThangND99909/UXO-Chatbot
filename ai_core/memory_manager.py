from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from typing import Dict, List

class UXOMemoryManager:
    def __init__(self, k=5):
        self.memories = {}  # Lưu trữ memory cho từng session
        self.k = k  # Số lượng tin nhắn lưu trong memory
    
    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Lấy memory cho session, tạo mới nếu chưa có"""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferWindowMemory(
                k=self.k,
                return_messages=True,
                memory_key="chat_history",
                output_key="answer"   # ✅ giữ nguyên
            )
        return self.memories[session_id]
    
    def save_context(self, session_id: str, inputs: Dict, outputs: Dict):
        """Lưu ngữ cảnh hội thoại"""
        memory = self.get_memory(session_id)
        print("DEBUG inputs:", inputs)
        print("DEBUG outputs:", outputs)
        memory.save_context(inputs, outputs)
    
    def clear_memory(self, session_id: str):
        """Xóa memory của session"""
        if session_id in self.memories:
            del self.memories[session_id]

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """Trả về danh sách message trong session"""
        memory = self.get_memory(session_id)
        return memory.chat_memory.messages  # ✅ API mới
