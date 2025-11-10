import React from 'react';
import { useApp } from '../../context/AppContext';
import LocationPicker from './LocationPicker';
import { useUXOReport } from '../../hooks/useUXOReport';
import { UI_TEXT } from '../../utils/constants';

const UXOMapSidebar = () => {
  const { state } = useApp();
  const {
    selectedLocation,
    setSelectedLocation,
    description,
    setDescription,
    isSubmitting,
    submitReport,
  } = useUXOReport(); // không cần token

  const handleSubmit = async () => {
    try {
      await submitReport();
      alert('✅ Gửi báo cáo thành công!');
    } catch (error) {
      alert(`❌ ${error.message}`);
    }
  };

  return (
    <div className="uxo-map-sidebar">
      <h3 className="sidebar-subtitle">{UI_TEXT.report_uxo[state.language]}</h3>

      <div className="sidebar-map">
        <LocationPicker onLocationSelect={setSelectedLocation} />
      </div>

      {selectedLocation && (
        <div className="alert alert-info">
          📍 Vị trí chọn: {selectedLocation.lat.toFixed(6)}, {selectedLocation.lng.toFixed(6)}
        </div>
      )}

      <div className="form-group">
        <label className="form-label">{UI_TEXT.description[state.language]}</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className="form-input form-textarea"
          placeholder={
            state.language === 'vi'
              ? 'Mô tả thêm về vật nghi ngờ...'
              : 'Additional description about suspected object...'
          }
        />
      </div>

      <button
        onClick={handleSubmit}
        disabled={isSubmitting}
        className="btn btn-primary"
        style={{ width: '100%' }}
      >
        {isSubmitting ? 'Đang gửi...' : UI_TEXT.send_report[state.language]}
      </button>
    </div>
  );
};

export default UXOMapSidebar;
