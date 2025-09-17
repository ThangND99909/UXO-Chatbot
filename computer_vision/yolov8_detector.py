from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UXODetector:
    def __init__(self, model_path: str = "models/uxo_yolov8.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")

        self.model = YOLO(model_path)

        # Config class names + colors
        self.class_names = ["bomb", "mine", "grenade", "artillery", "cluster_bomb"]
        self.colors = {
            "bomb": (0, 255, 0),        # Green
            "mine": (255, 0, 0),        # Blue
            "grenade": (0, 0, 255),     # Red
            "artillery": (255, 255, 0), # Cyan
            "cluster_bomb": (255, 0, 255) # Magenta
        }

    def detect(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image file not found: {image_path}")

        logger.info(f"🔍 Running detection on: {image_path}")
        results = self.model(image_path)

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

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "area": (x2 - x1) * (y2 - y1)
                })

        logger.info(f"✅ Found {len(detections)} detections")
        return detections

    def draw_detections(self, image_path: str, output_path: str, confidence_threshold: float = 0.5):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("❌ Could not read image file")

        detections = self.detect(image_path, confidence_threshold)

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            label = f"{detection['class']} {detection['confidence']:.2f}"
            color = self.colors.get(detection["class"], (0, 255, 0))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(image, (x1, y1 - label_height - 10),
                         (x1 + label_width, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.info(f"💾 Saved annotated image: {output_path}")
        return detections
