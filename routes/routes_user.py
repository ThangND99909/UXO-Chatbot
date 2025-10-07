# routers/user.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette import status

from database import crud, connection
from utils.auth import get_current_user
from app.schemas import ChatUpdateRequest, ChatDeleteResponse

router = APIRouter(prefix="/user", tags=["User"])

@router.patch("/chat/{chat_id}")
def update_user_chat(chat_id: int,
                     req: ChatUpdateRequest,
                     db: Session = Depends(connection.get_db),
                     current_user=Depends(get_current_user)):
    updated_chat = crud.update_chat_log(db, chat_id, req.new_content, current_user.id)
    if not updated_chat:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="❌ Không có quyền sửa tin nhắn này hoặc tin nhắn không tồn tại")
    return updated_chat

@router.delete("/chat/{chat_id}", response_model=ChatDeleteResponse)
def delete_user_chat(chat_id: int,
                     db: Session = Depends(connection.get_db),
                     current_user=Depends(get_current_user)):
    deleted_chat = crud.soft_delete_chat_log(db, chat_id, current_user.id)
    if not deleted_chat:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="❌ Không có quyền xóa tin nhắn này hoặc tin nhắn không tồn tại")
    return ChatDeleteResponse(message="✅ Tin nhắn đã được xóa", chat_id=chat_id)
