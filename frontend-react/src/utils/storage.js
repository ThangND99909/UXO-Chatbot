// Mô phỏng chức năng lưu session của Streamlit
const CHAT_SESSIONS_KEY = 'uxo_chat_sessions';

export const storageService = {
  // Lưu sessions
  saveSessions(sessions) {
    try {
      localStorage.setItem(CHAT_SESSIONS_KEY, JSON.stringify(sessions));
    } catch (error) {
      console.error('Error saving sessions:', error);
    }
  },

  // Load sessions
  loadSessions() {
    try {
      const sessions = localStorage.getItem(CHAT_SESSIONS_KEY);
      return sessions ? JSON.parse(sessions) : {};
    } catch (error) {
      console.error('Error loading sessions:', error);
      return {};
    }
  },

  // Lưu session hiện tại
  saveCurrentSession(sessionId, data) {
    const sessions = this.loadSessions();
    sessions[sessionId] = {
      ...data,
      lastUpdated: new Date().toISOString()
    };
    this.saveSessions(sessions);
  },

  // Load session cụ thể
  loadSession(sessionId) {
    const sessions = this.loadSessions();
    return sessions[sessionId] || null;
  },

  // Get all session IDs
  getAllSessionIds() {
    const sessions = this.loadSessions();
    return Object.keys(sessions);
  },

  // Delete session
  deleteSession(sessionId) {
    const sessions = this.loadSessions();
    delete sessions[sessionId];
    this.saveSessions(sessions);
  },

  // Clear all sessions
  clearAllSessions() {
    localStorage.removeItem(CHAT_SESSIONS_KEY);
  }
};