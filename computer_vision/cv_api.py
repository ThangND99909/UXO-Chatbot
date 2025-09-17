from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from yolov8_detector import UXODetector
import tempfile, os, uuid, logging
from datetime import datetime
import aiofiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UXO Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

detector = UXODetector()

class DetectionResult(BaseModel):
    class_: str
    confidence: float
    bbox: list
    area: int

class DetectionResponse(BaseModel):
    detections: list
    total_detections: int
    danger_level: str
    timestamp: str
    message: str
    annotated_image_url: str | None = None

def cleanup_temp_file(file_path: str):
    if os.path.exists(file_path):
        try:
            os.unlink(file_path)
            logger.info(f"🗑️ Deleted temp file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not delete {file_path}: {e}")

@app.post("/detect-uxo/", response_model=DetectionResponse)
async def detect_uxo(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        file_extension = os.path.splitext(file.filename)[1] or '.jpg'
        tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{file_extension}")

        async with aiofiles.open(tmp_path, "wb") as tmp_file:
            content = await file.read()
            await tmp_file.write(content)

        background_tasks.add_task(cleanup_temp_file, tmp_path)
        detections = detector.detect(tmp_path, confidence_threshold)

        danger_level = "low"
        if any(d["confidence"] > 0.8 for d in detections):
            danger_level = "high"
        elif detections:
            danger_level = "medium"

        return DetectionResponse(
            detections=detections,
            total_detections=len(detections),
            danger_level=danger_level,
            timestamp=datetime.now().isoformat(),
            message="⚠️ Nếu phát hiện vật nghi ngờ, không chạm vào và gọi ngay hotline 113"
        )

    except Exception as e:
        logger.error(f"❌ Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-uxo-with-image/", response_model=DetectionResponse)
async def detect_uxo_with_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        file_extension = os.path.splitext(file.filename)[1] or '.jpg'
        tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{file_extension}")

        async with aiofiles.open(tmp_path, "wb") as tmp_file:
            content = await file.read()
            await tmp_file.write(content)

        output_filename = f"detected_{uuid.uuid4().hex}{file_extension}"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        detections = detector.draw_detections(tmp_path, output_path, confidence_threshold)

        background_tasks.add_task(cleanup_temp_file, tmp_path)
        # chỉ cleanup output sau khi download xong
        # background_tasks.add_task(cleanup_temp_file, output_path)

        danger_level = "low"
        if any(d["confidence"] > 0.8 for d in detections):
            danger_level = "high"
        elif detections:
            danger_level = "medium"

        return DetectionResponse(
            detections=detections,
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
async def download_result(filename: str):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="image/jpeg")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
