import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import LanguageSwitcher from './LanguageSwitcher';
import AdminPanel from '../Admin/AdminPanel';
import ImageAnalysis from '../ImageAnalysis/ImageAnalysis';
import UXOMapSidebar from '../Map/UXOMapSidebar';
import HotlineInfo from './HotlineInfo';
import { UI_TEXT } from '../../utils/constants';

const Sidebar = () => {
  const { state } = useApp();
  const location = useLocation();

  // Ẩn sidebar trên các trang đặc biệt nếu cần
  if (location.pathname === '/emergency-report') {
    return null;
  }

  return (
    <div className="sidebar">
      <div className="sidebar-section">
        <h1 className="sidebar-title">⚠️ {UI_TEXT.title[state.language]}</h1>
        <p>{UI_TEXT.sidebar_description[state.language]}</p>
        
        <LanguageSwitcher />

        {/* Navigation Links */}
        <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          
            {location.pathname === '/' ? (
              <button 
                className="btn btn-disabled" 
                disabled 
                style={{ textDecoration: 'none', textAlign: 'center', opacity: 0.6, cursor: 'not-allowed' }}
              >
                💬 Chat chính
              </button>
            ) : (
              <Link 
                to="/" 
                className="btn btn-secondary"
                style={{ textDecoration: 'none', textAlign: 'center' }}
              >
                💬 Chat chính
              </Link>
            )}
          
          {state.adminToken && (
            <Link 
              to="/emergency-report" 
              className="btn btn-danger"
              style={{ textDecoration: 'none', textAlign: 'center' }}
            >
              🚨 Báo cáo khẩn cấp
            </Link>
          )}

          {state.adminToken && (
            <>
              <Link 
                to="/admin" 
                className={`btn ${location.pathname === '/admin' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ textDecoration: 'none', textAlign: 'center' }}
              >
                🔧 Admin Dashboard
              </Link>
              
              <Link 
                to="/detection-history" 
                className={`btn ${location.pathname === '/detection-history' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ textDecoration: 'none', textAlign: 'center' }}
              >
                📸 Detection History
              </Link>
            </>
          )}
        </div>
      </div>

      <div className="sidebar-section">
        <ImageAnalysis />
      </div>

      <div className="sidebar-section">
        <AdminPanel />
      </div>

      <div className="sidebar-section">
        <UXOMapSidebar />
      </div>

      <div className="sidebar-section">
        <HotlineInfo />
      </div>
    </div>
  );
};

export default Sidebar;