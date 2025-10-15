from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from database import connection, crud
from .load_env import SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES  # ✅ import tự động


# ============================================================
# Cấu hình HASH MẬT KHẨU & XÁC THỰC BẰNG JWT TOKEN
# ============================================================

# Tạo context để hash mật khẩu với thuật toán bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Cấu hình scheme cho OAuth2 (dùng để lấy token từ request)
# Khi người dùng gửi request có header Authorization: Bearer <token>,
# FastAPI sẽ tự động lấy token này để xác thực.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login")

# ============================================================
# HÀM XỬ LÝ MẬT KHẨU
# ============================================================
def hash_password(password: str) -> str:
    """Hash mật khẩu trước khi lưu vào database"""
    return pwd_context.hash(password)

def verify_password(password: str, hashed_password: str) -> bool:
    """So sánh mật khẩu người dùng nhập với hash trong database"""
    return pwd_context.verify(password, hashed_password)

# ============================================================
# HÀM TẠO JWT TOKEN
# ============================================================
def create_access_token(data: dict, expires_delta: timedelta = None):
    """
    Tạo JWT token từ payload (ví dụ: {"sub": "admin_id"})
    - expires_delta: thời gian hết hạn (mặc định từ file .env)
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire}) # thêm thời gian hết hạn vào payload
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256") # tạo token

# ============================================================
# HÀM XÁC THỰC ADMIN TỪ TOKEN
# ============================================================

def get_current_admin(token: str = Depends(oauth2_scheme), db: Session = Depends(connection.get_db)):
    """
    Giải mã JWT token để xác định admin hiện tại.
    - Nếu token không hợp lệ hoặc admin không tồn tại => raise 401
    """
    credentials_exception = HTTPException(status_code=401, detail="Token không hợp lệ")
    try:
        # Giải mã token
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        admin_id: int = int(payload.get("sub")) # lấy ID từ payload
        if admin_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Tìm admin trong database
    admin = crud.get_admin(db, admin_id=admin_id)
    if admin is None:
        raise credentials_exception
    return admin
