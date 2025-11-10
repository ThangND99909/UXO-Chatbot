import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';
import LocationPicker from '../components/Map/LocationPicker';
import { UI_TEXT } from '../utils/constants';

const EmergencyReport = () => {
  const { state, dispatch } = useApp();
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleLocationSelect = (latlng) => {
    setSelectedLocation(latlng);
  };

  const handleSubmitReport = async () => {
    if (!selectedLocation) {
      alert('Vui lòng chọn vị trí trên bản đồ');
      return;
    }
    
    if (!state.adminToken) {
      alert('Vui lòng đăng nhập Admin để gửi báo cáo');
      return;
    }

    setIsSubmitting(true);
    try {
      await apiService.reportUXO(
        selectedLocation.lat,
        selectedLocation.lng,
        description,
        state.adminToken
      );
      
      alert('✅ Báo cáo khẩn cấp đã được gửi thành công!');
      setSelectedLocation(null);
      setDescription('');
      
      // Quay lại trang chủ sau 2 giây
      setTimeout(() => {
        window.location.href = '/';
      }, 2000);
      
    } catch (error) {
      alert(`❌ Lỗi khi gửi báo cáo: ${error.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    window.history.back();
  };

  return (
    <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h1>🚨 BÁO CÁO UXO KHẨN CẤP</h1>
        <p style={{ color: '#666', fontSize: '1.1rem' }}>
          Sử dụng trang này để báo cáo vị trí vật nổ nghi ngờ
        </p>
      </div>

      <div className="alert alert-warning" style={{ marginBottom: '2rem' }}>
        <strong>⚠️ CẢNH BÁO QUAN TRỌNG:</strong>
        <ul style={{ marginTop: '0.5rem', marginBottom: '0' }}>
          <li>KHÔNG chạm vào vật nghi ngờ</li>
          <li>Giữ khoảng cách an toàn ít nhất 50m</li>
          <li>Cảnh báo người xung quanh</li>
          <li>Chờ đội ngũ chuyên môn đến xử lý</li>
        </ul>
      </div>

      <div style={{ marginBottom: '2rem' }}>
        <h3>1. 📍 Chọn vị trí trên bản đồ</h3>
        <p style={{ color: '#666', marginBottom: '1rem' }}>
          Click trên bản đồ để xác định vị trí chính xác
        </p>
        <LocationPicker onLocationSelect={handleLocationSelect} height={400} />
      </div>

      {selectedLocation && (
        <div className="alert alert-success" style={{ marginBottom: '2rem' }}>
          ✅ <strong>Đã chọn vị trí:</strong> {selectedLocation.lat.toFixed(6)}, {selectedLocation.lng.toFixed(6)}
        </div>
      )}

      <div style={{ marginBottom: '2rem' }}>
        <h3>2. 📝 Mô tả chi tiết</h3>
        <div className="form-group">
          <label className="form-label">
            <strong>Thông tin về vật nghi ngờ:</strong>
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="form-input form-textarea"
            placeholder="Ví dụ:
- Vật thể hình quả trứng, màu nâu, kích thước khoảng 30cm
- Nằm trong ruộng lúa, cách đường 20m
- Có dấu hiệu rỉ sét, một phần bị vùi trong đất
- Phát hiện lúc 14:30 ngày 15/12/2024"
            rows={6}
          />
        </div>
      </div>

      <div style={{ marginBottom: '2rem' }}>
        <h3>3. 📞 Thông tin liên hệ (tùy chọn)</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div className="form-group">
            <label className="form-label">Họ và tên</label>
            <input type="text" className="form-input" placeholder="Nguyễn Văn A" />
          </div>
          <div className="form-group">
            <label className="form-label">Số điện thoại</label>
            <input type="tel" className="form-input" placeholder="0912345678" />
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginBottom: '2rem' }}>
        <button 
          onClick={handleSubmitReport}
          disabled={!selectedLocation || isSubmitting || !state.adminToken}
          className="btn btn-primary"
          style={{ padding: '1rem 2rem', fontSize: '1.1rem' }}
        >
          {isSubmitting ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div className="spinner"></div>
              Đang gửi báo cáo...
            </div>
          ) : (
            '🚨 GỬI BÁO CÁO KHẨN CẤP'
          )}
        </button>
        
        <button 
          onClick={handleCancel}
          className="btn btn-secondary"
          style={{ padding: '1rem 2rem', fontSize: '1.1rem' }}
        >
          ❌ Hủy
        </button>
      </div>

      {!state.adminToken && (
        <div className="alert alert-error" style={{ textAlign: 'center' }}>
          <strong>⚠️ CẦN ĐĂNG NHẬP ADMIN</strong>
          <p>Vui lòng đăng nhập tài khoản Admin để gửi báo cáo khẩn cấp</p>
        </div>
      )}

      <div className="alert alert-info">
        <strong>📞 HOTLINE KHẨN CẤP - GỌI NGAY:</strong>
        <div style={{ marginTop: '1rem' }}>
          {UI_TEXT.hotline[state.language].split('\n').map((line, index) => (
            <div key={index} style={{ marginBottom: '0.5rem', fontSize: '1.1rem' }}>
              {line}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default EmergencyReport;