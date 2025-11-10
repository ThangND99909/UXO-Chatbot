import React, { useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { UI_TEXT } from '../../utils/constants';

const ImageUploader = ({ onImageUpload }) => {
  const { state } = useApp();
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileSelect = (file) => {
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png' || file.type === 'image/jpg')) {
      onImageUpload(file);
    } else {
      alert('Vui lòng chọn file ảnh (JPG, JPEG, PNG)');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="image-uploader">
      <div
        className={`upload-area ${isDragging ? 'dragover' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        style={{ marginBottom: '1rem', cursor: 'pointer' }}
      >
        <div className="upload-text">
          <strong>Drag and drop file here</strong><br />
          Limit 200MB per file • JPG, JPEG, PNG
        </div>
        <div>
          {state.language === 'vi' ? 'Duyệt files' : 'Browse files'}
        </div>
      </div>
      
      <input
        type="file"
        ref={fileInputRef}
        accept="image/jpeg,image/png,image/jpg"
        onChange={(e) => handleFileSelect(e.target.files[0])}
        style={{ display: 'none' }}
      />
    </div>
  );
};

export default ImageUploader;