from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import os

class UXODetector:
    def __init__(self, model_path: str = "models/uxo_yolov8.pt"):
        # Kiểm tra file model tồn tại
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.class_names = [
            "bomb", "mine", "grenade", "artillery", "cluster_bomb"
        ]
    
    def detect(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """Nhận diện vật thể UXO trong ảnh với ngưỡng confidence"""
        # Kiểm tra file ảnh tồn tại
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        results = self.model(image_path)
        detections = []
        
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                
                # Lọc theo ngưỡng confidence
                if confidence < confidence_threshold:
                    continue
                
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "area": (x2 - x1) * (y2 - y1)  # Thêm diện tích để đánh giá kích thước
                })
        
        return detections
    
    def draw_detections(self, image_path: str, output_path: str, confidence_threshold: float = 0.5):
        """Vẽ bounding box lên ảnh và lưu kết quả"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image file")
        
        detections = self.detect(image_path, confidence_threshold)
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            label = f"{detection['class']} {detection['confidence']:.2f}"
            
            # Vẽ bounding box với màu sắc khác nhau cho các class
            colors = {
                "bomb": (0, 255, 0),        # Xanh lá
                "mine": (255, 0, 0),        # Xanh dương
                "grenade": (0, 0, 255),     # Đỏ
                "artillery": (255, 255, 0), # Xanh dương + Xanh lá
                "cluster_bomb": (255, 0, 255) # Đỏ + Xanh dương
            }
            
            color = colors.get(detection["class"], (0, 255, 0))
            
            # Vẽ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Vẽ background cho nhãn
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(image, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), color, -1)
            
            # Vẽ nhãn
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        return detections