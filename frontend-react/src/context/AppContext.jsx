import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { storageService } from '../utils/storage';

const AppContext = createContext();

const initialState = {
  sessionId: null,
  chatHistory: [],
  language: 'vi',
  adminToken: null,
  chatLogs: [],
  detectionReports: [],
  processedImages: {},
  showReportMap: false,
  lastIntent: null,
  analysisDone: false,
  detectionResult: null,
  processedImage: null,
  adminDetectionHistory: [],
  selectedDetectionImage: null,
  lastLogCount: 0,
  showLoginForm: false
};

function appReducer(state, action) {
  switch (action.type) {
    case 'SET_SESSION_ID':
      return { ...state, sessionId: action.payload };
    
    case 'ADD_CHAT_MESSAGE':
      return { 
        ...state, 
        chatHistory: [...state.chatHistory, action.payload] 
      };
    
    case 'SET_CHAT_HISTORY':
      return { ...state, chatHistory: action.payload };
    
    case 'SET_LANGUAGE':
      return { ...state, language: action.payload };
    
    case 'SET_ADMIN_TOKEN':
      return { ...state, adminToken: action.payload };
    
    case 'SET_CHAT_LOGS':
      return { ...state, chatLogs: action.payload };
    
    case 'SET_DETECTION_RESULT':
      return { 
        ...state, 
        detectionResult: action.payload,
        analysisDone: true 
      };
    
    case 'SET_PROCESSED_IMAGE':
      return { ...state, processedImage: action.payload };
    
    case 'SHOW_REPORT_MAP':
      return { ...state, showReportMap: action.payload };
    
    case 'SET_DETECTION_HISTORY':
      return { ...state, adminDetectionHistory: action.payload };
    
    case 'SET_DETECTION_REPORTS':
      return { ...state, detectionReports: action.payload };
    
    case 'ADD_PROCESSED_IMAGE':
      return {
        ...state,
        processedImages: {
          ...state.processedImages,
          [action.payload]: true
        }
      };
    
    case 'SET_LAST_INTENT':
      return { ...state, lastIntent: action.payload };
    
    case 'SET_LAST_LOG_COUNT':
      return { ...state, lastLogCount: action.payload };
    
    case 'TOGGLE_LOGIN_FORM':
      return { ...state, showLoginForm: !state.showLoginForm };
    
    case 'SET_SELECTED_DETECTION_IMAGE':
      return { ...state, selectedDetectionImage: action.payload };
    
    case 'LOGOUT_ADMIN':
      return {
        ...state,
        adminToken: null,
        chatLogs: [],
        lastLogCount: 0,
        detectionReports: [],
        selectedDetectionImage: null,
        adminDetectionHistory: []
      };
    
    default:
      return state;
  }
}

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Khởi tạo session ID và load data từ localStorage
  useEffect(() => {
    // Tạo hoặc load session ID
    let sessionId = state.sessionId;
    if (!sessionId) {
      sessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
      dispatch({ type: 'SET_SESSION_ID', payload: sessionId });
    }

    // Load session data từ localStorage
    const sessionData = storageService.loadSession(sessionId);
    if (sessionData) {
      if (sessionData.chatHistory) {
        dispatch({ type: 'SET_CHAT_HISTORY', payload: sessionData.chatHistory });
      }
      if (sessionData.showReportMap !== undefined) {
        dispatch({ type: 'SHOW_REPORT_MAP', payload: sessionData.showReportMap });
      }
      if (sessionData.lastIntent) {
        dispatch({ type: 'SET_LAST_INTENT', payload: sessionData.lastIntent });
      }
    }
  }, [state.sessionId]);

  // Auto-save session data khi có thay đổi
  useEffect(() => {
    if (state.sessionId) {
      const sessionData = {
        chatHistory: state.chatHistory,
        showReportMap: state.showReportMap,
        lastIntent: state.lastIntent,
        analysisDone: state.analysisDone,
        detectionResult: state.detectionResult
      };
      storageService.saveCurrentSession(state.sessionId, sessionData);
    }
  }, [state.sessionId, state.chatHistory, state.showReportMap, state.lastIntent]);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};