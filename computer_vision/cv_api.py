from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from yolov8_detector import UXODetector
import tempfile
import os
from datetime import datetime
import uuid

app = FastAPI(title="UXO Detection API", version="1.0.0")

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = UXODetector()

def cleanup_temp_file(file_path: str):
    """Xóa file tạm sau khi xử lý"""
    if os.path.exists(file_path):
        os.unlink(file_path)

@app.post("/detect-uxo/")
async def detect_uxo(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    """API endpoint để phát hiện UXO trong ảnh"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Lưu file tạm
        file_extension = os.path.splitext(file.filename)[1] or '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Đảm bảo xóa file tạm sau khi xử lý
        background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        # Phát hiện vật thể
        detections = detector.detect(tmp_path, confidence_threshold)
        
        # Phân loại mức độ nguy hiểm
        danger_level = "low"
        if any(detection["confidence"] > 0.8 for detection in detections):
            danger_level = "high"
        elif detections:
            danger_level = "medium"
        
        return {
            "detections": detections,
            "total_detections": len(detections),
            "danger_level": danger_level,
            "timestamp": datetime.now().isoformat(),
            "message": "Cảnh báo: Nếu phát hiện vật nghi ngờ, không chạm vào và gọi ngay hotline 113 hoặc 090 xxxx xxx"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")

@app.post("/detect-uxo-with-image/")
async def detect_uxo_with_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    """API trả về cả detection results và ảnh đã được đánh dấu"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Lưu file tạm
        file_extension = os.path.splitext(file.filename)[1] or '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Tạo file output tạm
        output_filename = f"detected_{uuid.uuid4().hex}{file_extension}"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        # Xử lý ảnh
        detections = detector.draw_detections(tmp_path, output_path, confidence_threshold)
        
        # Đảm bảo xóa file tạm
        background_tasks.add_task(cleanup_temp_file, tmp_path)
        background_tasks.add_task(cleanup_temp_file, output_path)
        
        # Phân loại mức độ nguy hiểm
        danger_level = "low"
        if any(detection["confidence"] > 0.8 for detection in detections):
            danger_level = "high"
        elif detections:
            danger_level = "medium"
        
        return {
            "detections": detections,
            "total_detections": len(detections),
            "danger_level": danger_level,
            "annotated_image_url": f"/download-result/{output_filename}",
            "timestamp": datetime.now().isoformat(),
            "message": "Cảnh báo: Nếu phát hiện vật nghi ngờ, không chạm vào và gọi ngay hotline 113"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")

@app.get("/download-result/{filename}")
async def download_result(filename: str):
    """Download ảnh đã được đánh dấu"""
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="image/jpeg")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}