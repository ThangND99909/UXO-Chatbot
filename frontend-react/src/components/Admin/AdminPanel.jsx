import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { apiService } from '../../services/api';
import AdminLogin from './AdminLogin';
import ChatLogs from './ChatLogs';
import DetectionHistory from './DetectionHistory';
import UXOReports from './UXOReports';
import { UI_TEXT } from '../../utils/constants';

const AdminPanel = () => {
  const { state, dispatch } = useApp();
  const [activeTab, setActiveTab] = useState('login');

  const handleLogout = () => {
    dispatch({ type: 'LOGOUT_ADMIN' });
    setActiveTab('login');
  };

  const fetchChatLogs = async () => {
    if (!state.adminToken) return;
    
    try {
      const logs = await apiService.getChatLogs(state.adminToken);
      dispatch({ type: 'SET_CHAT_LOGS', payload: logs });
    } catch (error) {
      console.error('Error fetching chat logs:', error);
    }
  };

  const fetchDetectionReports = async () => {
    if (!state.adminToken) return;
    
    try {
      const reports = await apiService.getAllDetections(state.adminToken);
      dispatch({ type: 'SET_DETECTION_REPORTS', payload: reports });
    } catch (error) {
      console.error('Error fetching detection reports:', error);
    }
  };

  useEffect(() => {
    if (state.adminToken) {
      fetchChatLogs();
      fetchDetectionReports();
      setActiveTab('chatlogs');
    }
  }, [state.adminToken]);

  if (!state.adminToken && !state.showLoginForm) {
    return (
      <div className="admin-panel">
        <h3 className="sidebar-subtitle">{UI_TEXT.admin_manage[state.language]}</h3>
        <button
          onClick={() => dispatch({ type: 'TOGGLE_LOGIN_FORM' })}
          className="btn btn-primary"
          style={{ width: '100%' }}
        >
          🔓 {UI_TEXT.admin_login[state.language]}
        </button>
      </div>
    );
  }

  if (!state.adminToken && state.showLoginForm) {
    return <AdminLogin />;
  }

  return (
    <div className="admin-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3 className="sidebar-subtitle">🔑 {UI_TEXT.admin_manage[state.language]}</h3>
        <button
          onClick={handleLogout}
          className="btn btn-danger"
          style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
        >
          🚪 {UI_TEXT.admin_logout[state.language]}
        </button>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <button
          className={`btn ${activeTab === 'chatlogs' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('chatlogs')}
          style={{ flex: 1, minWidth: '80px' }}
        >
          💬 Chat Logs
        </button>
        <button
          className={`btn ${activeTab === 'detections' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('detections')}
          style={{ flex: 1, minWidth: '80px' }}
        >
          📸 Detections
        </button>
        <button
          className={`btn ${activeTab === 'reports' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('reports')}
          style={{ flex: 1, minWidth: '80px' }}
        >
          📍 Reports
        </button>
      </div>

      {activeTab === 'chatlogs' && <ChatLogs />}
      {activeTab === 'detections' && <DetectionHistory />}
      {activeTab === 'reports' && <UXOReports />}
    </div>
  );
};

export default AdminPanel;