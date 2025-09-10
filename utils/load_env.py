import os
from dotenv import load_dotenv, set_key, dotenv_values
import secrets

# ===== Load env hiện tại =====
env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_file)

# ===== Check SECRET_KEY =====
config = dotenv_values(env_file)
if "SECRET_KEY" not in config or not config["SECRET_KEY"]:
    # Generate key mới
    new_key = secrets.token_urlsafe(32)
    # Ghi vào .env
    set_key(env_file, "SECRET_KEY", new_key)
    print(f"✅ SECRET_KEY mới được tạo và lưu vào .env: {new_key}")

# Load lại .env sau khi có SECRET_KEY
load_dotenv(env_file)

# Lấy giá trị SECRET_KEY
SECRET_KEY = os.getenv("SECRET_KEY")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
