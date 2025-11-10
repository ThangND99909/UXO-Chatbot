import React, { useEffect, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { apiService } from '../../services/api';
import { UI_TEXT } from '../../utils/constants';

const ChatLogs = () => {
  const { state, dispatch } = useApp();
  const [isLoading, setIsLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const fetchChatLogs = async () => {
    if (!state.adminToken) return;
    
    setIsLoading(true);
    try {
      const logs = await apiService.getChatLogs(state.adminToken);
      dispatch({ type: 'SET_CHAT_LOGS', payload: logs });
    } catch (error) {
      console.error('Error fetching chat logs:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchChatLogs();
  }, [state.adminToken]);

  useEffect(() => {
    let interval;
    if (autoRefresh) {
      interval = setInterval(fetchChatLogs, 30000); // 30 seconds
    }
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const newLogsCount = state.chatLogs.length - state.lastLogCount;

  return (
    <div className="chat-logs">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h4>📄 Chat Logs (Admin)</h4>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {newLogsCount > 0 && (
            <span className="alert alert-info" style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}>
              📢 {newLogsCount} mới
            </span>
          )}
          <button
            onClick={fetchChatLogs}
            disabled={isLoading}
            className="btn btn-primary"
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
          >
            {isLoading ? '🔄' : '↻'}
          </button>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`btn ${autoRefresh ? 'btn-success' : 'btn-secondary'}`}
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
          >
            {autoRefresh ? '⏸️' : '▶️'}
          </button>
        </div>
      </div>

      <div style={{ maxHeight: '300px', overflowY: 'auto', border: '1px solid #ddd', borderRadius: '5px', padding: '0.5rem' }}>
        {state.chatLogs.length > 0 ? (
          [...state.chatLogs].reverse().map((log, index) => {
            const isNew = index < newLogsCount;
            return (
              <div
                key={log.id || index}
                style={{
                  padding: '0.5rem',
                  borderBottom: '1px solid #eee',
                  backgroundColor: isNew ? '#fff3cd' : 'transparent',
                  borderRadius: '3px',
                  marginBottom: '0.25rem',
                  fontSize: '0.8rem'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                  <span style={{ fontWeight: 'bold' }}>
                    [{new Date(log.created_at).toLocaleString('vi-VN')}]
                  </span>
                  <code style={{ fontSize: '0.7rem', color: '#666' }}>
                    {log.session_id?.substring(0, 8)}...
                  </code>
                </div>
                <div><strong>Q:</strong> {log.message}</div>
                <div><strong>A:</strong> {log.response}</div>
              </div>
            );
          })
        ) : (
          <div style={{ textAlign: 'center', color: '#666', padding: '1rem' }}>
            {UI_TEXT.no_chat_logs[state.language]}
          </div>
        )}
      </div>

      <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#666', textAlign: 'center' }}>
        Total: {state.chatLogs.length} logs
        {autoRefresh && ' • Auto-refresh enabled'}
      </div>
    </div>
  );
};

export default ChatLogs;