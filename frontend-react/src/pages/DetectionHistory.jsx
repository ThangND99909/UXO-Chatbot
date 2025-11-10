import React from 'react';
import { useApp } from '../context/AppContext';
import DetectionHistory from '../components/Admin/DetectionHistory';

const DetectionHistoryPage = () => {
  const { state } = useApp();

  if (!state.adminToken) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '50vh',
        flexDirection: 'column',
        gap: '1rem'
      }}>
        <h2>🔒 Detection History</h2>
        <p>Vui lòng đăng nhập để xem lịch sử phát hiện</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h1>📸 Lịch sử phát hiện ảnh</h1>
        <span style={{ fontSize: '0.9rem', color: '#666' }}>
          Tổng số: {state.detectionReports.length} reports
        </span>
      </div>
      
      <DetectionHistory />
    </div>
  );
};

export default DetectionHistoryPage;