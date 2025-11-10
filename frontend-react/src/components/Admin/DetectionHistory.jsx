import React, { useEffect, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { apiService } from '../../services/api';
import { UI_TEXT } from '../../utils/constants';

const DetectionHistory = () => {
  const { state, dispatch } = useApp();
  const [isLoading, setIsLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedReport, setSelectedReport] = useState(null);
  const [detectionImage, setDetectionImage] = useState(null);

  const itemsPerPage = 5;

  const fetchDetectionReports = async () => {
    if (!state.adminToken) return;
    
    setIsLoading(true);
    try {
      const reports = await apiService.getAllDetections(state.adminToken);
      dispatch({ type: 'SET_DETECTION_REPORTS', payload: reports });
    } catch (error) {
      console.error('Error fetching detection reports:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchDetectionImage = async (reportId) => {
    if (!state.adminToken) return;
    
    try {
      const imageBlob = await apiService.getDetectionImage(reportId, state.adminToken);
      const imageUrl = URL.createObjectURL(imageBlob);
      setDetectionImage(imageUrl);
    } catch (error) {
      console.error('Error fetching detection image:', error);
      alert('Lỗi khi tải ảnh detection');
    }
  };

  useEffect(() => {
    fetchDetectionReports();
  }, [state.adminToken]);

  // Statistics
  const totalReports = state.detectionReports.length;
  const totalObjects = state.detectionReports.reduce((sum, report) => 
    sum + (report.detected_objects?.length || 0), 0
  );
  const reportsWithObjects = state.detectionReports.filter(
    report => report.detected_objects && report.detected_objects.length > 0
  ).length;

  // Pagination
  const totalPages = Math.ceil(totalReports / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentReports = state.detectionReports.slice(startIndex, startIndex + itemsPerPage);

  const handleViewImage = (report) => {
    setSelectedReport(report);
    fetchDetectionImage(report.id);
  };

  const handleCloseImage = () => {
    setSelectedReport(null);
    setDetectionImage(null);
    if (detectionImage) {
      URL.revokeObjectURL(detectionImage);
    }
  };

  return (
    <div className="detection-history">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h4>📸 {UI_TEXT.detection_history[state.language]}</h4>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={fetchDetectionReports}
            disabled={isLoading}
            className="btn btn-primary"
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
          >
            {isLoading ? '🔄' : '↻'}
          </button>
          <button
            onClick={() => dispatch({ type: 'SET_DETECTION_REPORTS', payload: [] })}
            className="btn btn-secondary"
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
          >
            🗑️ Clear
          </button>
        </div>
      </div>

      {/* Statistics */}
      {totalReports > 0 && (
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(2, 1fr)', 
          gap: '0.5rem', 
          marginBottom: '1rem' 
        }}>
          <div className="alert alert-info" style={{ padding: '0.5rem', textAlign: 'center' }}>
            <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{totalReports}</div>
            <div style={{ fontSize: '0.7rem' }}>Total Reports</div>
          </div>
          <div className="alert alert-success" style={{ padding: '0.5rem', textAlign: 'center' }}>
            <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{totalObjects}</div>
            <div style={{ fontSize: '0.7rem' }}>Objects Found</div>
          </div>
          <div className="alert alert-warning" style={{ padding: '0.5rem', textAlign: 'center' }}>
            <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{reportsWithObjects}</div>
            <div style={{ fontSize: '0.7rem' }}>Positive Cases</div>
          </div>
          <div className="alert alert-secondary" style={{ padding: '0.5rem', textAlign: 'center' }}>
            <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{totalReports - reportsWithObjects}</div>
            <div style={{ fontSize: '0.7rem' }}>Empty Cases</div>
          </div>
        </div>
      )}

      {/* Reports List */}
      <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
        {currentReports.length > 0 ? (
          currentReports.map((report, index) => (
            <div
              key={report.id}
              style={{
                border: '1px solid #ddd',
                borderRadius: '5px',
                padding: '0.75rem',
                marginBottom: '0.5rem',
                backgroundColor: '#f9f9f9'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '0.5rem' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>📁 {report.filename}</div>
                  <div style={{ fontSize: '0.7rem', color: '#666' }}>
                    🆔 {report.id} • 📅 {new Date(report.created_at).toLocaleString('vi-VN')}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#666' }}>
                    👤 Session: {report.session_id?.substring(0, 10)}...
                  </div>
                </div>
                <button
                  onClick={() => handleViewImage(report)}
                  className="btn btn-primary"
                  style={{ padding: '0.25rem 0.5rem', fontSize: '0.7rem' }}
                >
                  👁️ View
                </button>
              </div>

              {/* Detection Results */}
              {report.detected_objects && report.detected_objects.length > 0 ? (
                <div>
                  <div style={{ fontSize: '0.8rem', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                    🎯 Detected Objects ({report.detected_objects.length}):
                  </div>
                  {report.detected_objects.map((obj, idx) => (
                    <div
                      key={idx}
                      style={{
                        fontSize: '0.7rem',
                        padding: '0.25rem',
                        backgroundColor: '#e8f5e8',
                        borderRadius: '3px',
                        marginBottom: '0.1rem'
                      }}
                    >
                      <strong>{obj.class}</strong> ({(obj.confidence * 100).toFixed(1)}%)
                      {obj.bbox && (
                        <span style={{ color: '#666', marginLeft: '0.5rem' }}>
                          📍 [{obj.bbox.map(b => b.toFixed(1)).join(', ')}]
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: '0.8rem', color: '#666', fontStyle: 'italic' }}>
                  📭 No objects detected
                </div>
              )}
            </div>
          ))
        ) : (
          <div style={{ textAlign: 'center', color: '#666', padding: '2rem' }}>
            {isLoading ? 'Đang tải...' : UI_TEXT.no_detection_history[state.language]}
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1rem', marginTop: '1rem' }}>
          <button
            onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
            disabled={currentPage === 1}
            className="btn btn-secondary"
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
          >
            ← Prev
          </button>
          <span style={{ fontSize: '0.8rem' }}>
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
            disabled={currentPage === totalPages}
            className="btn btn-secondary"
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
          >
            Next →
          </button>
        </div>
      )}

      {/* Image Modal */}
      {selectedReport && detectionImage && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.8)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'white',
            padding: '1rem',
            borderRadius: '10px',
            maxWidth: '90%',
            maxHeight: '90%',
            overflow: 'auto'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h4>🖼️ {selectedReport.filename}</h4>
              <button
                onClick={handleCloseImage}
                className="btn btn-danger"
                style={{ padding: '0.25rem 0.5rem' }}
              >
                ✕
              </button>
            </div>
            
            <img
              src={detectionImage}
              alt={`Detection ${selectedReport.id}`}
              style={{ maxWidth: '100%', maxHeight: '70vh', borderRadius: '5px' }}
            />
            
            <div style={{ marginTop: '1rem' }}>
              <div style={{ fontSize: '0.9rem' }}>
                <strong>📏 Image Analysis:</strong>
              </div>
              {selectedReport.detected_objects && selectedReport.detected_objects.length > 0 ? (
                <div>
                  <div style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
                    🎯 <strong>Objects Detected:</strong> {selectedReport.detected_objects.length}
                  </div>
                  {selectedReport.detected_objects.map((obj, idx) => (
                    <div key={idx} style={{ fontSize: '0.8rem', marginLeft: '1rem', marginTop: '0.25rem' }}>
                      • <strong>{obj.class}</strong> - Confidence: {(obj.confidence * 100).toFixed(1)}%
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: '0.8rem', color: '#666', fontStyle: 'italic' }}>
                  No objects detected in this image
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DetectionHistory;