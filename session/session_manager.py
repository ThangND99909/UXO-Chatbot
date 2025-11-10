import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ====== QUẢN LÝ PHIÊN HỘI THOẠI ======
user_sessions = {}

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """
    Lấy session ID hiện có hoặc tạo mới nếu chưa có.
    Mỗi session lưu:
    - Thời điểm tạo
    - Thời điểm hoạt động cuối
    - Số tin nhắn đã gửi
    """
    if not session_id or session_id not in user_sessions:
        new_session_id = str(uuid.uuid4())
        user_sessions[new_session_id] = {
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0
        }
        logger.info(f"Created new session: {new_session_id}")
        return new_session_id
    
    user_sessions[session_id]["last_activity"] = datetime.now()
    user_sessions[session_id]["message_count"] += 1
    return session_id

def get_session_id_from_multiple_sources(
    header_session_id: Optional[str] = None,
    cookie_session_id: Optional[str] = None,
    body_session_id: Optional[str] = None
) -> Optional[str]:
    """Lấy session_id từ header, cookie, hoặc body (ưu tiên theo thứ tự)."""
    return header_session_id or cookie_session_id or body_session_id

async def cleanup_old_sessions(qa_instance=None):
    """
    Nhiệm vụ nền chạy định kỳ mỗi giờ.
    - Xóa các session không hoạt động quá 24h.
    - Dọn dẹp memory tương ứng.
    """
    import asyncio
    while True:
        try:
            now = datetime.now()
            to_delete = [sid for sid, data in user_sessions.items() if now - data["last_activity"] > timedelta(hours=24)]
            for sid in to_delete:
                if qa_instance and hasattr(qa_instance, 'memory_manager'):
                    qa_instance.memory_manager.clear_memory(sid)
                del user_sessions[sid]
                logger.info(f"Cleaned up old session: {sid}")
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(300)