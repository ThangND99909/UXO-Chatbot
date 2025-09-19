from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from yolov8_detector import UXODetector
from pathlib import Path
import tempfile, os, uuid, logging, aiofiles
from datetime import datetime

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)  # Logger để in thông tin debug, error, info

# ---------------- FastAPI app ----------------
app = FastAPI(title="UXO Detection API", version="2.2.0")  # Khởi tạo FastAPI app

# CORS middleware để cho phép truy cập từ bất kỳ domain nào (phù hợp cho frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------- Model load ----------------
MODEL_PATH = Path(__file__).resolve().parent / "weights" / "best.pt"  # Đường dẫn model YOLOv8
detector = UXODetector(model_path=str(MODEL_PATH))  # Load model UXO
logger.info(f"📦 Loaded UXO model from: {MODEL_PATH}")

# ---------------- Schemas ----------------
class DetectionResult(BaseModel):
    class_: str
    confidence: float
    bbox: list
    area: int

class DetectionResponse(BaseModel):
    detections: list[DetectionResult]
    total_detections: int
    danger_level: str
    timestamp: str
    message: str
    annotated_image_url: str | None = None  # Link tới ảnh đã vẽ detection (nếu có)

# ---------------- Utils ----------------
def cleanup_temp_file(file_path: str):
    """Delete temp files safely"""
    if os.path.exists(file_path):
        try:
            os.unlink(file_path)  # Xóa file tạm
            logger.info(f"🗑️ Deleted temp file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not delete {file_path}: {e}")

# Ngưỡng đánh giá mức độ nguy hiểm dựa trên confidence
DANGER_THRESHOLDS = {"high": 0.8, "medium": 0.5}

def calculate_danger_level(detections):
    """Decide danger level based on confidence thresholds"""
    if any(d["confidence"] > DANGER_THRESHOLDS["high"] for d in detections):
        return "high"
    elif any(d["confidence"] > DANGER_THRESHOLDS["medium"] for d in detections):
        return "medium"
    return "low"

# ---------------- Routes ----------------
@app.post("/detect-uxo/", response_model=DetectionResponse)
async def detect_uxo(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    # Kiểm tra file upload có phải ảnh không
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        content = await file.read()  # Đọc bytes của ảnh

        # ✅ detect trực tiếp từ bytes (không cần file tạm)
        detections = detector.detect_from_bytes(content, confidence_threshold)
        detection_objects = [DetectionResult(**{
            "class_": d["class"],
            "confidence": d["confidence"],
            "bbox": d["bbox"],
            "area": d["area"]
        }) for d in detections]  # Chuyển detections sang schema

        danger_level = calculate_danger_level(detections)  # Tính mức nguy hiểm
        logger.info(f"📷 {file.filename} → {len(detections)} detections, danger={danger_level}")

        # Trả về kết quả detection
        return DetectionResponse(
            detections=detection_objects,
            total_detections=len(detections),
            danger_level=danger_level,
            timestamp=datetime.now().isoformat(),
            message="⚠️ Nếu phát hiện vật nghi ngờ, không chạm vào và gọi ngay hotline 113"
        )

    except Exception as e:
        logger.error(f"❌ Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


"""@app.post("/detect-uxo-with-image/", response_model=DetectionResponse)
async def detect_uxo_with_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    # Kiểm tra file upload có phải ảnh không
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        content = await file.read()  # Đọc bytes của ảnh
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Chuẩn bị tên file annotated tạm thời
        file_extension = os.path.splitext(file.filename)[1] or ".jpg"
        output_filename = f"detected_{uuid.uuid4().hex}{file_extension}"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        # Dò và vẽ detection trực tiếp từ bytes, lưu file annotated
        detections = detector.draw_detections_from_bytes(
            image_bytes=content,
            output_path=output_path,
            confidence_threshold=confidence_threshold
        )

        # Chuyển detections thành schema
        detection_objects = [
            DetectionResult(**{
                "class_": d["class"],
                "confidence": d["confidence"],
                "bbox": d["bbox"],
                "area": d["area"]
            }) for d in detections
        ]

        danger_level = calculate_danger_level(detections)
        logger.info(f"📷 {file.filename} → {len(detections)} detections, danger={danger_level} (with image)")

        # Thêm task xóa file tạm sau khi download
        background_tasks.add_task(cleanup_temp_file, output_path)

        return DetectionResponse(
            detections=detection_objects,
            total_detections=len(detections),
            danger_level=danger_level,
            annotated_image_url=f"/download-result/{output_filename}",
            timestamp=datetime.now().isoformat(),
            message="⚠️ Nếu phát hiện vật nghi ngờ, không chạm vào và gọi ngay hotline 113"
        )

    except Exception as e:
        logger.error(f"❌ Error during detection with image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-result/{filename}")
async def download_result(filename: str, background_tasks: BackgroundTasks):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Xóa file tạm sau khi download
    background_tasks.add_task(cleanup_temp_file, file_path)
    return FileResponse(file_path, media_type="image/jpeg")  # Trả về file ảnh đã annotate"""


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}  # Route kiểm tra server còn sống
