import React from 'react';
import { useApp } from '../../context/AppContext';
import LocationPicker from './LocationPicker';
import { useUXOReport } from '../../hooks/useUXOReport';
import { UI_TEXT } from '../../utils/constants';

const UXOReportMap = () => {
  const { state, dispatch } = useApp();
  const {
    selectedLocation,
    setSelectedLocation,
    description,
    setDescription,
    isSubmitting,
    submitReport,
  } = useUXOReport(state.adminToken); // có token

  if (!state.showReportMap) return null;

  const handleSubmit = async () => {
    try {
      await submitReport();
      alert('✅ Báo cáo khẩn cấp đã được gửi thành công!');
      dispatch({ type: 'SHOW_REPORT_MAP', payload: false });
    } catch (error) {
      alert(`❌ ${error.message}`);
    }
  };

  const handleCancel = () => {
    dispatch({ type: 'SHOW_REPORT_MAP', payload: false });
    setSelectedLocation(null);
    setDescription('');
  };

  return (
    <div className="uxo-report-main" style={{ background: 'white', padding: '2rem' }}>
      <h2>📍 BÁO CÁO UXO KHẨN CẤP</h2>

      <div className="alert alert-warning" style={{ margin: '1rem 0' }}>
        ⚠️ KHÔNG TỰ Ý XỬ LÝ!  
        Hãy chọn vị trí chính xác nơi bạn phát hiện vật nghi ngờ.
      </div>

      <LocationPicker onLocationSelect={setSelectedLocation} height={400} />

      {selectedLocation && (
        <div className="alert alert-success">
          ✅ Vị trí: {selectedLocation.lat.toFixed(6)}, {selectedLocation.lng.toFixed(6)}
        </div>
      )}

      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        className="form-input form-textarea"
        placeholder="Mô tả thêm về hiện trường (ví dụ: vật tròn màu xám, dài khoảng 50cm...)"
        rows={3}
      />

      <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
        <button
          onClick={handleSubmit}
          disabled={!selectedLocation || isSubmitting}
          className="btn btn-primary"
          style={{ flex: 1 }}
        >
          {isSubmitting ? 'Đang gửi...' : '🚨 GỬI BÁO CÁO'}
        </button>
        <button onClick={handleCancel} className="btn btn-danger">
          ❌ Hủy
        </button>
      </div>

      <div className="alert alert-error" style={{ marginTop: '1rem' }}>
        📞 HOTLINE KHẨN CẤP:  
        {UI_TEXT.hotline[state.language].split('\n').map((line, i) => (
          <div key={i}>{line}</div>
        ))}
      </div>
    </div>
  );
};

export default UXOReportMap;
