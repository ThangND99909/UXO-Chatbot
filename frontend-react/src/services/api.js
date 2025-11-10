const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8501";

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const config = {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      };

      if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
        config.body = JSON.stringify(options.body);
      }

      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  // Chat endpoints
  async sendChat(message, sessionId, language = 'vi') {
    return this.request('/ask', {
      method: 'POST',
      body: { message, session_id: sessionId, language }
    });
  }

  // Image detection
  async detectUXO(imageFile, sessionId, confidenceThreshold = 0.3) {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('session_id', sessionId);
    formData.append('confidence_threshold', confidenceThreshold.toString());

    const response = await fetch(`${this.baseURL}/admin/detect-uxo/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Image detection failed');
    }

    return response.json();
  }

  // Admin endpoints
  async adminLogin(email, password) {
    return this.request('/admin/login', {
      method: 'POST',
      body: { email, password }
    });
  }

  async getChatLogs(token, skip = 0, limit = 50) {
    return this.request(`/admin/chatlogs?skip=${skip}&limit=${limit}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
  }

  async reportUXO(latitude, longitude, description) {
    return this.request('/admin/report-uxo', {
      method: 'POST',
      
      body: { latitude, longitude, description }
    });
  }

  async getAllDetections(token) {
    return this.request('/admin/all-detections', {
      headers: { Authorization: `Bearer ${token}` }
    });
  }

  async getDetectionImage(detectionId, token) {
    const response = await fetch(`${this.baseURL}/admin/detections/${detectionId}`, {
      headers: { Authorization: `Bearer ${token}` }
    });

    if (!response.ok) {
      throw new Error('Failed to fetch detection image');
    }

    return response.blob();
  }

  async getUXOReports(token) {
    return this.request('/admin/uxo-reports', {
      headers: { Authorization: `Bearer ${token}` }
    });
  }
}

export const apiService = new ApiService();