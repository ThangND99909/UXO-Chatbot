
import sqlite3
import bcrypt

# Đường dẫn database SQLite của bạn
DB_PATH = "sql_app.db"

# Thông tin admin test
ADMIN_EMAIL = "test@gmail.com"
ADMIN_PASSWORD = "123456"

def create_admin():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Tạo bảng admins nếu chưa tồn tại
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Kiểm tra xem admin đã tồn tại chưa
    cursor.execute("SELECT * FROM admins WHERE email = ?", (ADMIN_EMAIL,))
    if cursor.fetchone():
        print(f"⚠️ Admin đã tồn tại: {ADMIN_EMAIL}")
    else:
        # Hash password bằng bcrypt
        hashed = bcrypt.hashpw(ADMIN_PASSWORD.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO admins (email, hashed_password) VALUES (?, ?)", 
                       (ADMIN_EMAIL, hashed.decode('utf-8')))
        conn.commit()
        print(f"✅ Admin tạo thành công: {ADMIN_EMAIL}")

    conn.close()

if __name__ == "__main__":
    create_admin()
