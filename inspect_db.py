####CHECK DATABASE STRUCTURE####
# inspect_db.py

import sqlite3
import json

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

print("🚀 KIỂM TRA NHANH UXO_DETECTIONS")

# Kiểm tra uxo_detections
cursor.execute("SELECT id, filename, session_id, created_at, detected_objects FROM uxo_detections;")
detections = cursor.fetchall()

print(f"📊 Tổng records: {len(detections)}")

for row in detections:
    id, filename, session_id, created_at, detected_objects = row
    print(f"\n🎯 ID: {id}")
    print(f"   File: {filename}")
    print(f"   Session: {session_id}")
    print(f"   Time: {created_at}")
    if detected_objects:
        try:
            objects = json.loads(detected_objects)
            print(f"   Detections: {len(objects)} objects")
            for obj in objects:
                print(f"     - {obj.get('class', 'Unknown')} ({obj.get('confidence', 0):.2f})")
        except:
            print(f"   Detections: [Invalid JSON]")
    else:
        print(f"   Detections: None")

print(f"\n🔍 KIỂM TRA IMAGE_DETECTION_LOGS")
cursor.execute("SELECT id, detection_id, session_id, confidence FROM image_detection_logs;")
logs = cursor.fetchall()

print(f"📊 Tổng logs: {len(logs)}")
for log in logs:
    print(f"   Log ID: {log[0]}, Detection ID: {log[1]}, Session: {log[2]}, Conf: {log[3]}")

conn.close()
