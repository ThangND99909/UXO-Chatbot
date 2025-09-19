from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import os
import logging
from dotenv import load_dotenv
from io import BytesIO

# ---------------- Load biến môi trường ----------------
load_dotenv()  # Load các biến môi trường từ file .env nếu có

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UXODetector")  # Logger để in thông tin debug, info, error

# ---------------- UXODetector class ----------------
class UXODetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Khởi tạo UXODetector với YOLOv8 model.
        - Nếu không truyền model_path thì sẽ lấy từ biến môi trường MODEL_PATH
        - Nếu vẫn không có thì fallback về ./weights/best.pt
        """
        if model_path is None:
            model_path = os.getenv(
                "MODEL_PATH",
                os.path.join(os.path.dirname(__file__), "weights", "best.pt")
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")

        try:
            logger.info(f"📂 Loading YOLO model from: {model_path}")
            self.model = YOLO(model_path)  # Load model YOLOv8
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            raise

        # ---------------- Class names và màu vẽ ----------------
        self.class_names = [
            "Ammunition",
            "Bomb",
            "Mine",
            "Mortar",
            "Projectile",
            "Rocket"
        ]

        # Gán màu khác nhau cho từng class để vẽ bbox
        self.colors = {
            "Ammunition": (255, 165, 0),   # Orange
            "Bomb": (0, 255, 0),           # Green
            "Mine": (255, 0, 0),           # Blue
            "Mortar": (0, 0, 255),         # Red
            "Projectile": (255, 255, 0),   # Cyan
            "Rocket": (255, 0, 255)        # Magenta
        }

    # ---------------- Detection từ file path ----------------
    def detect(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Chạy detection trên ảnh từ đường dẫn và trả về danh sách dict kết quả."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image file not found: {image_path}")

        logger.info(f"🔍 Running detection on: {image_path}")
        results = self.model(image_path)  # Chạy YOLO detect

        return self._parse_results(results, confidence_threshold)  # Chuyển kết quả thành dict

    # ---------------- Detection từ bytes ----------------
    def detect_from_bytes(self, image_bytes: bytes, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Chạy detection trực tiếp từ bytes ảnh.
        Bao gồm kiểm tra lỗi và fallback lưu file tạm nếu decode thất bại.
        """
        try:
            if not image_bytes or len(image_bytes) == 0:
                raise ValueError("❌ Uploaded file is empty")

            # Chuyển bytes sang np.ndarray
            image_array = np.frombuffer(image_bytes, np.uint8)
            if image_array.size == 0:
                raise ValueError("❌ Image bytes are empty or invalid")

            # Giải mã ảnh bằng OpenCV
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                # Fallback: lưu tạm file và đọc lại bằng OpenCV
                import tempfile, uuid
                tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.jpg")
                with open(tmp_path, "wb") as f:
                    f.write(image_bytes)
                image = cv2.imread(tmp_path)
                if image is None:
                    raise ValueError("❌ Could not decode image even after saving temp file")
                else:
                    logger.info(f"⚠️ Decoded image via temp file: {tmp_path}")

            # Chạy YOLO detect
            results = self.model.predict(image)
            detections = self._parse_results(results, confidence_threshold)

            logger.info(f"✅ detect_from_bytes: Found {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"❌ Error in detect_from_bytes: {e}")
            return []

    # ---------------- Parse YOLO results ----------------
    def _parse_results(self, results, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Chuyển YOLO results thành list dict với class, bbox, confidence, area,..."""
        detections = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < confidence_threshold:
                    continue

                class_id = int(box.cls[0])
                class_name = (
                    self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                )
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                width, height = x2 - x1, y2 - y1

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "width": width,
                    "height": height,
                    "area": width * height
                })

        logger.info(f"✅ Found {len(detections)} detections")
        return detections

    # ---------------- Vẽ bbox từ file path ----------------
    def draw_detections(self, image_path: str, output_path: str,
                        confidence_threshold: float = 0.5,
                        save_conf: bool = True, save_crop: bool = False) -> List[Dict[str, Any]]:
        """Vẽ bounding box và lưu ảnh kết quả."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("❌ Could not read image file")

        # Chạy detect
        detections = self.detect(image_path, confidence_threshold)

        # Vẽ bbox và label
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            label = f"{detection['class']} {detection['confidence']:.2f}" if save_conf else detection['class']
            color = self.colors.get(detection["class"], (0, 255, 0))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)  # Vẽ bbox

            # Vẽ label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(image, (x1, y1 - label_height - 10),
                          (x1 + label_width, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Tuỳ chọn crop object lưu riêng
            if save_crop:
                crop = image[y1:y2, x1:x2]
                crop_dir = os.path.join(os.path.dirname(output_path), "crops")
                os.makedirs(crop_dir, exist_ok=True)
                crop_path = os.path.join(crop_dir, f"{detection['class']}_{x1}_{y1}.jpg")
                cv2.imwrite(crop_path, crop)

        # Lưu ảnh kết quả
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.info(f"💾 Saved annotated image: {output_path}")
        return detections

    # ---------------- Vẽ bbox từ bytes ----------------
    def draw_detections_from_bytes(
        self, image_bytes: bytes, output_path: str,
        confidence_threshold: float = 0.5,
        save_conf: bool = True, save_crop: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Vẽ bounding box trực tiếp từ bytes ảnh và lưu ảnh kết quả.
        Trả về danh sách detections.
        Bao gồm kiểm tra lỗi và fallback lưu file tạm nếu decode thất bại.
        """
        try:
            if not image_bytes or len(image_bytes) == 0:
                raise ValueError("❌ Uploaded file is empty")

            # Chuyển bytes sang np.ndarray
            image_array = np.frombuffer(image_bytes, np.uint8)
            if image_array.size == 0:
                raise ValueError("❌ Image bytes are empty or invalid")

            # Giải mã ảnh bằng OpenCV
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                # Fallback: lưu tạm file và đọc lại bằng OpenCV
                import tempfile, uuid
                tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.jpg")
                with open(tmp_path, "wb") as f:
                    f.write(image_bytes)
                image = cv2.imread(tmp_path)
                if image is None:
                    raise ValueError("❌ Could not decode image even after saving temp file")
                else:
                    logger.info(f"⚠️ Decoded image via temp file: {tmp_path}")

            # Chạy YOLO detect
            results = self.model.predict(image)
            detections = self._parse_results(results, confidence_threshold)

            # Vẽ bbox & label
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                label = f"{detection['class']} {detection['confidence']:.2f}" if save_conf else detection['class']
                color = self.colors.get(detection["class"], (0, 255, 0))

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)  # Vẽ bbox

                # Vẽ label background
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Tuỳ chọn crop object lưu riêng
                if save_crop:
                    crop = image[y1:y2, x1:x2]
                    crop_dir = os.path.join(os.path.dirname(output_path), "crops")
                    os.makedirs(crop_dir, exist_ok=True)
                    crop_path = os.path.join(crop_dir, f"{detection['class']}_{x1}_{y1}.jpg")
                    cv2.imwrite(crop_path, crop)

            # Lưu ảnh kết quả
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            logger.info(f"💾 Saved annotated image: {output_path}")

            return detections

        except Exception as e:
            logger.error(f"❌ Error in draw_detections_from_bytes: {e}")
            return []
