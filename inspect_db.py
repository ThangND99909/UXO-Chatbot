####CHECK DATABASE STRUCTURE####
# inspect_db.py

import sqlite3

#python inspect_db.py
# Kết nối database
conn = sqlite3.connect("sql_app.db")
cursor = conn.cursor()

# Lấy danh sách bảng
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("📌 Danh sách bảng trong sql_app.db:")
for t in tables:
    print("-", t[0])

# Nếu muốn xem cột chi tiết từng bảng
for t in tables:
    print(f"\n🔹 Schema bảng {t[0]}:")
    cursor.execute(f"PRAGMA table_info({t[0]});")
    for col in cursor.fetchall():
        print(col)

conn.close()
