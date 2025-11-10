import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { apiService } from '../../services/api';
import { UI_TEXT } from '../../utils/constants';

const AdminLogin = () => {
  const { state, dispatch } = useApp();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // 🔹 Load email từ localStorage khi mở lại
  useEffect(() => {
    const savedEmail = localStorage.getItem('admin_email');
    if (savedEmail) {
      setEmail(savedEmail);
      setRememberMe(true);
    }
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();

    if (!email || !password) {
      setError('Vui lòng nhập đầy đủ email và mật khẩu');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await apiService.adminLogin(email, password);
      dispatch({ type: 'SET_ADMIN_TOKEN', payload: response.access_token });
      dispatch({ type: 'TOGGLE_LOGIN_FORM' });

      // 🔹 Ghi nhớ email nếu user chọn
      if (rememberMe) {
        localStorage.setItem('admin_email', email);
      } else {
        localStorage.removeItem('admin_email');
      }

      setPassword('');
    } catch (error) {
      setError(error.message || 'Đăng nhập thất bại');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    dispatch({ type: 'TOGGLE_LOGIN_FORM' });
    setEmail('');
    setPassword('');
    setError('');
  };

  return (
    <div className="admin-login">
      <h3 className="sidebar-subtitle">🔐 {UI_TEXT.admin_login[state.language]}</h3>

      <form onSubmit={handleLogin} className="login-form">
        {/* EMAIL */}
        <div className="form-group">
          <label className="form-label">📧 Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="form-input"
            placeholder="admin@example.com"
            disabled={isLoading}
          />
        </div>

        {/* PASSWORD */}
        <div className="form-group" style={{ position: 'relative' }}>
          <label className="form-label">🔑 Mật khẩu</label>
          <input
            type={showPassword ? 'text' : 'password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="form-input"
            placeholder="Nhập mật khẩu..."
            disabled={isLoading}
          />
          {/* 👁 Toggle show/hide password */}
          <span
            onClick={() => setShowPassword(!showPassword)}
            style={{
              position: 'absolute',
              right: '10px',
              top: '35px',
              cursor: 'pointer',
              fontSize: '0.9rem',
              color: '#666',
            }}
          >
            {showPassword ? '🙈 Ẩn' : '👁 Hiện'}
          </span>
        </div>

        {/* REMEMBER EMAIL */}
        <div style={{ margin: '0.5rem 0' }}>
          <label>
            <input
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
              disabled={isLoading}
            />{' '}
            Ghi nhớ email
          </label>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            type="submit"
            disabled={isLoading}
            className="btn btn-primary"
            style={{ flex: 2 }}
          >
            {isLoading ? '🔄 Đang đăng nhập...' : 'Đăng nhập'}
          </button>
          <button
            type="button"
            onClick={handleCancel}
            className="btn btn-secondary"
            style={{ flex: 1 }}
          >
            ❌ Hủy
          </button>
        </div>
      </form>
    </div>
  );
};

export default AdminLogin;
