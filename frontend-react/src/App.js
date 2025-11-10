import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import Sidebar from './components/Common/Sidebar';
import Home from './pages/Home';
import AdminDashboard from './pages/AdminDashboard';
import EmergencyReport from './pages/EmergencyReport';
import DetectionHistory from './pages/DetectionHistory';
import UXOReportMap from './components/Map/UXOReportMap';
import './styles/globals.css';

function App() {
  return (
    <AppProvider>
      <Router>
        <div className="app-container">
          <Sidebar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/admin" element={<AdminDashboard />} />
              <Route path="/emergency-report" element={<EmergencyReport />} />
              <Route path="/detection-history" element={<DetectionHistory />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
            <UXOReportMap />
          </main>
        </div>
      </Router>
    </AppProvider>
  );
}

export default App;