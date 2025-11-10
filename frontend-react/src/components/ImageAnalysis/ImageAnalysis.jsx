import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import { apiService } from '../../services/api';
import { UI_TEXT } from '../../utils/constants';

const ImageAnalysis = () => {
  const { state, dispatch } = useApp();
  const [uploadedImage, setUploadedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png' || file.type === 'image/jpg')) {
      setUploadedImage(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target.result);
      reader.readAsDataURL(file);
    } else {
      alert('Vui lòng chọn file ảnh (JPG, JPEG, PNG)');
    }
  };

  const handleAnalyzeImage = async () => {
    if (!uploadedImage) return;
    
    setIsAnalyzing(true);
    try {
      const result = await apiService.detectUXO(uploadedImage, state.sessionId);
      
      dispatch({ type: 'SET_DETECTION_RESULT', payload: result });
      dispatch({ type: 'ADD_PROCESSED_IMAGE', payload: uploadedImage.name });
      
      // Add to detection history if admin
      if (state.adminToken && result.detection_id) {
        const newDetection = {
          id: result.detection_id,
          filename: uploadedImage.name,
          detections: result.detections || [],
          created_at: new Date().toISOString(),
          session_id: state.sessionId
        };
        
        dispatch({ 
          type: 'SET_DETECTION_HISTORY', 
          payload: [...state.adminDetectionHistory, newDetection] 
        });
      }
    } catch (error) {
      alert(`Lỗi phân tích ảnh: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearImage = () => {
    setUploadedImage(null);
    setImagePreview(null);
    dispatch({ type: 'SET_DETECTION_RESULT', payload: null });
  };

  return (
    <div className="image-analysis" style={{ overflow: 'visible', minHeight: '300px' }}>
      <h3 className="sidebar-subtitle">{UI_TEXT.analyze_image[state.language]}</h3>
      
      <div className="form-group">
        <input
          type="file"
          accept="image/jpeg,image/png,image/jpg"
          onChange={handleImageUpload}
          style={{ display: 'none' }}
          id="image-upload"
        />
        <label 
          htmlFor="image-upload" 
          className="upload-area"
          style={{ cursor: 'pointer' }}
        >
          <div className="upload-text">
            <strong>{UI_TEXT.upload_image[state.language]}</strong><br />
            JPG, JPEG, PNG
          </div>
        </label>
      </div>

      {imagePreview && (
        <div className="image-preview-container">
          <img 
            src={imagePreview} 
            alt="Uploaded preview" 
            className="image-preview"
          />
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
            <button 
              onClick={handleAnalyzeImage}
              disabled={isAnalyzing}
              className="btn btn-primary"
            >
              {isAnalyzing ? 'Đang phân tích...' : UI_TEXT.analyze_image[state.language]}
            </button>
            <button 
              onClick={clearImage}
              className="btn btn-secondary"
            >
              Xóa ảnh
            </button>
          </div>
        </div>
      )}

      {state.detectionResult && (
        <div className="detection-results">
          <h4>{UI_TEXT.image_result[state.language]}</h4>
          {state.detectionResult.detections && state.detectionResult.detections.length > 0 ? (
            state.detectionResult.detections.map((det, index) => (
              <div key={index} className="detection-item">
                <strong>{det.class}</strong> - Độ tin cậy: {(det.confidence * 100).toFixed(1)}%
              </div>
            ))
          ) : (
            <div>{UI_TEXT.no_detection[state.language]}</div>
          )}
        </div>
      )}
    </div>
  );
};

export default ImageAnalysis;