import React, { useEffect, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { apiService } from '../../services/api';
import { UI_TEXT } from '../../utils/constants';

const UXOReports = () => {
  const { state } = useApp();
  const [reports, setReports] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchUXOReports = async () => {
    if (!state.adminToken) return;
    
    setIsLoading(true);
    try {
      const uxoReports = await apiService.getUXOReports(state.adminToken);
      setReports(uxoReports);
    } catch (error) {
      console.error('Error fetching UXO reports:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUXOReports();
  }, [state.adminToken]);

  return (
    <div className="uxo-reports">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h4>📍 {UI_TEXT.report_uxo[state.language]} (Admin)</h4>
        <button
          onClick={fetchUXOReports}
          disabled={isLoading}
          className="btn btn-primary"
          style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
        >
          {isLoading ? '🔄' : '↻'}
        </button>
      </div>

      <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
        {reports.length > 0 ? (
          reports.map((report, index) => (
            <div
              key={report.id}
              style={{
                border: '1px solid #ffcccc',
                borderRadius: '5px',
                padding: '0.75rem',
                marginBottom: '0.5rem',
                backgroundColor: '#fff5f5'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                <div style={{ fontWeight: 'bold', color: '#d63031' }}>
                  ⚠️ Report #{report.id}
                </div>
                <div style={{ fontSize: '0.7rem', color: '#666' }}>
                  {new Date(report.created_at).toLocaleString('vi-VN')}
                </div>
              </div>
              
              <div style={{ fontSize: '0.8rem', marginBottom: '0.25rem' }}>
                <strong>📍 Location:</strong> {report.latitude.toFixed(6)}, {report.longitude.toFixed(6)}
              </div>
              
              <div style={{ fontSize: '0.8rem' }}>
                <strong>📝 Description:</strong> {report.description || UI_TEXT.no_description[state.language]}
              </div>
            </div>
          ))
        ) : (
          <div style={{ textAlign: 'center', color: '#666', padding: '2rem' }}>
            {isLoading ? 'Đang tải...' : UI_TEXT.no_uxo_reports[state.language]}
          </div>
        )}
      </div>

      {reports.length > 0 && (
        <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#666', textAlign: 'center' }}>
          Total: {reports.length} reports
        </div>
      )}
    </div>
  );
};

export default UXOReports;