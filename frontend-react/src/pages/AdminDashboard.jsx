import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import ChatLogs from '../components/Admin/ChatLogs';
import DetectionHistory from '../components/Admin/DetectionHistory';
import UXOReports from '../components/Admin/UXOReports';

const AdminDashboard = () => {
  const { state } = useApp();
  const [activeTab, setActiveTab] = useState('chatlogs');

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
        <h2>🔒 Admin Dashboard</h2>
        <p>Vui lòng đăng nhập để truy cập trang quản trị</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h1>🔧 Admin Dashboard</h1>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.9rem', color: '#666' }}>
            Session: {state.sessionId?.substring(0, 10)}...
          </span>
        </div>
      </div>

      {/* Tab Navigation */}
      <div style={{ 
        display: 'flex', 
        gap: '0.5rem', 
        marginBottom: '2rem',
        borderBottom: '1px solid #ddd',
        paddingBottom: '0.5rem'
      }}>
        <button
          className={`btn ${activeTab === 'chatlogs' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('chatlogs')}
        >
          💬 Chat Logs
        </button>
        <button
          className={`btn ${activeTab === 'detections' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('detections')}
        >
          📸 Detection History
        </button>
        <button
          className={`btn ${activeTab === 'reports' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('reports')}
        >
          📍 UXO Reports
        </button>
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === 'chatlogs' && <ChatLogs />}
        {activeTab === 'detections' && <DetectionHistory />}
        {activeTab === 'reports' && <UXOReports />}
      </div>
    </div>
  );
};

export default AdminDashboard;