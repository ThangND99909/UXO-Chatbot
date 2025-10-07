# evaluate_via_api.py
import requests
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import logging
from datetime import datetime
import os
import time
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIEvaluator:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.results = {}
    
    def load_test_dataset(self, test_data_path="test_data/nlu_test_dataset.json"):
        """Load dataset test từ file JSON"""
        try:
            if not os.path.isabs(test_data_path):
                test_data_path = os.path.join(os.path.dirname(__file__), test_data_path)
                
            with open(test_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File test data không tồn tại: {test_data_path}")
            return None
    
    def test_api_connection(self):
        """Test kết nối API"""
        try:
            response = requests.get(f"{self.api_url}/")
            if response.status_code == 200:
                logger.info("✅ API connection successful")
                return True
            else:
                logger.error(f"❌ API connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ API connection error: {e}")
            return False
    
    def send_chat_message(self, question: str, language: str = "vi", session_id: str = "evaluation"):
        """Gửi message đến API giống như Streamlit app"""
        try:
            response = requests.post(
                f"{self.api_url}/ask",
                json={
                    "message": question, 
                    "session_id": session_id, 
                    "language": language
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "intent": result.get("intent", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "answer": result.get("answer", "")
                }
            else:
                logger.error(f"❌ API error: {response.status_code} - {response.text}")
                return {"intent": "unknown", "confidence": 0.0, "error": response.text}
                
        except Exception as e:
            logger.error(f"❌ Request error: {e}")
            return {"intent": "unknown", "confidence": 0.0, "error": str(e)}
    
    def run_evaluation(self, test_data_path="test_data/nlu_test_dataset.json"):
        """Chạy đánh giá thông qua API"""
        logger.info("🔍 Bắt đầu đánh giá qua API...")
        
        # Test API connection
        if not self.test_api_connection():
            return None
        
        # Load dữ liệu test
        test_data = self.load_test_dataset(test_data_path)
        if not test_data:
            logger.error("Không thể load dữ liệu test")
            return None
        
        # Chuẩn bị dữ liệu cho evaluation
        true_labels = []
        predicted_labels = []
        test_cases = []
        
        total_samples = sum(len(samples) for samples in test_data.values())
        processed = 0
        
        for intent, samples in test_data.items():
            for sample in samples:
                text = sample['text']
                language = sample.get('language', 'vi')
                
                # Hiển thị tiến trình
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"📊 Đang xử lý: {processed}/{total_samples}")
                
                # Gửi request đến API (giống Streamlit)
                api_result = self.send_chat_message(text, language, f"eval-session-{processed}")
                
                predicted_intent = api_result["intent"]
                
                # Lưu kết quả
                true_labels.append(intent)
                predicted_labels.append(predicted_intent)
                test_cases.append({
                    "text": text,
                    "language": language,
                    "true_intent": intent,
                    "predicted_intent": predicted_intent,
                    "confidence": api_result["confidence"],
                    "correct": intent == predicted_intent,
                    "api_response": api_result.get("answer", ""),
                    "error": api_result.get("error")
                })
                
                # Thêm delay nhỏ để tránh quá tải API
                time.sleep(0.5)
        
        # Tính toán metrics
        self._calculate_metrics(true_labels, predicted_labels, test_cases)
        
        # Lưu kết quả chi tiết
        self._save_detailed_results(test_cases, test_data_path)
        
        return self.results
    
    def _calculate_metrics(self, y_true, y_pred, test_cases):
        """Tính toán các metrics đánh giá"""
        
        # Lấy danh sách các intent duy nhất
        labels = sorted(set(y_true + y_pred))
        
        # Tính metrics tổng thể
        precision = precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        
        # Tính metrics cho từng class
        class_report = classification_report(
            y_true, y_pred, 
            labels=labels, 
            output_dict=True,
            zero_division=0
        )
        
        # Lưu kết quả
        self.results = {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'total_samples': len(y_true),
                'accuracy': sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            },
            'per_class': {},
            'confusion_matrix': self._create_confusion_matrix(y_true, y_pred, labels),
            'test_cases_details': test_cases
        }
        
        # Extract per-class metrics
        for intent in labels:
            if intent in class_report:
                self.results['per_class'][intent] = {
                    'precision': class_report[intent]['precision'],
                    'recall': class_report[intent]['recall'],
                    'f1_score': class_report[intent]['f1-score'],
                    'support': class_report[intent]['support']
                }
    
    def _create_confusion_matrix(self, y_true, y_pred, labels):
        """Tạo confusion matrix"""
        cm = defaultdict(lambda: defaultdict(int))
        
        for true, pred in zip(y_true, y_pred):
            cm[true][pred] += 1
        
        return dict(cm)
    
    def _save_detailed_results(self, test_cases, test_data_path):
        """Lưu kết quả chi tiết ra file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = "evaluation_reports"
        
        if not os.path.isabs(report_dir):
            report_dir = os.path.join(os.path.dirname(__file__), report_dir)
        
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        # Lưu kết quả chi tiết JSON
        detailed_results = {
            'summary': self.results,
            'test_cases': test_cases,
            'evaluation_info': {
                'timestamp': timestamp,
                'test_data_source': test_data_path,
                'api_url': self.api_url,
                'total_test_cases': len(test_cases),
                'method': 'API-based evaluation (same as Streamlit)'
            }
        }
        
        report_path = os.path.join(report_dir, f"api_evaluation_{timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Tạo báo cáo markdown
        self._create_markdown_report(report_dir, timestamp)
        
        logger.info(f"📊 Kết quả đánh giá đã được lưu tại: {report_dir}")
    
    def _create_markdown_report(self, report_dir, timestamp):
        """Tạo báo cáo dạng markdown"""
        md_content = "# Kết quả huấn luyện NLU Processor (API Evaluation)\n\n"
        md_content += "*Đánh giá thông qua API endpoint `/ask` giống với Streamlit app*\n\n"
        
        md_content += "## Phân loại Intent\n\n"
        md_content += "| Intent | Precision | Recall | F1-score | Support |\n"
        md_content += "|--------|-----------|--------|----------|---------|\n"
        
        overall = self.results['overall']
        per_class = self.results['per_class']
        
        for intent, metrics in per_class.items():
            md_content += f"| {intent} | {metrics['precision']:.2f} | {metrics['recall']:.2f} | {metrics['f1_score']:.2f} | {metrics['support']} |\n"
        
        # Thêm dòng trung bình
        md_content += f"| **Trung bình** | **{overall['precision']:.2f}** | **{overall['recall']:.2f}** | **{overall['f1_score']:.2f}** | **{overall['total_samples']}** |\n\n"
        
        # Thêm thông tin tổng quan
        md_content += f"## Tổng quan\n\n"
        md_content += f"- **Độ chính xác tổng thể**: {overall['accuracy']:.3f}\n"
        md_content += f"- **Tổng số mẫu test**: {overall['total_samples']}\n"
        md_content += f"- **Phương pháp**: API-based evaluation\n"
        md_content += f"- **API Endpoint**: {self.api_url}/ask\n"
        md_content += f"- **Thời gian đánh giá**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Lưu file markdown
        md_path = os.path.join(report_dir, f"api_training_results_{timestamp}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def print_detailed_report(self):
        """In báo cáo chi tiết ra console"""
        if not self.results:
            print("Chưa có kết quả đánh giá. Hãy chạy run_evaluation() trước.")
            return
        
        print("\n" + "="*60)
        print("📊 KẾT QUẢ HUẤN LUYỆN NLU PROCESSOR (API EVALUATION)")
        print("="*60)
        
        overall = self.results['overall']
        per_class = self.results['per_class']
        
        print(f"\n📈 Kết quả tổng thể:")
        print(f"   • Độ chính xác: {overall['accuracy']:.3f}")
        print(f"   • Precision:    {overall['precision']:.3f}")
        print(f"   • Recall:       {overall['recall']:.3f}")
        print(f"   • F1-score:     {overall['f1_score']:.3f}")
        print(f"   • Tổng mẫu:     {overall['total_samples']}")
        
        print(f"\n🎯 Kết quả theo từng intent:")
        print("-" * 65)
        print(f"{'Intent':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
        print("-" * 65)
        
        for intent, metrics in per_class.items():
            print(f"{intent:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f} {metrics['support']:<10}")
        
        print("-" * 65)
        print(f"{'Trung bình':<15} {overall['precision']:<10.3f} {overall['recall']:<10.3f} {overall['f1_score']:<10.3f} {overall['total_samples']:<10}")

def main():
    """Hàm chính để chạy đánh giá qua API"""
    print("🚀 Starting NLU Evaluation via API (Streamlit method)...")
    
    evaluator = APIEvaluator(api_url="http://localhost:8000")
    results = evaluator.run_evaluation()
    
    if results:
        # Hiển thị kết quả
        evaluator.print_detailed_report()
        
        print(f"\n✅ Đánh giá hoàn tất!")
        print(f"📁 Kết quả chi tiết đã được lưu trong thư mục: evaluation_reports/")
        print(f"🔗 Method: Gọi API endpoint /ask (giống với Streamlit app)")
    else:
        print(f"\n❌ Đánh giá thất bại!")
        print(f"💡 Kiểm tra:")
        print(f"   - FastAPI server có đang chạy trên http://localhost:8000 không?")
        print(f"   - API endpoint /ask có hoạt động không?")

if __name__ == "__main__":
    main()