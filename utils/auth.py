from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from database import connection, crud
from .load_env import SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES  # ✅ import tự động

# ===== Password hashing =====
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)

# ===== JWT =====
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

def get_current_admin(token: str = Depends(oauth2_scheme), db: Session = Depends(connection.get_db)):
    credentials_exception = HTTPException(status_code=401, detail="❌ Token không hợp lệ")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        admin_id: int = int(payload.get("sub"))
        if admin_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    admin = crud.get_admin(db, admin_id=admin_id)
    if admin is None:
        raise credentials_exception
    return admin
