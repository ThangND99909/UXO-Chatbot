import React from 'react';
import { useApp } from '../../context/AppContext';
import { UI_TEXT } from '../../utils/constants';

const LanguageSwitcher = () => {
  const { state, dispatch } = useApp();

  const handleLanguageChange = (language) => {
    dispatch({ type: 'SET_LANGUAGE', payload: language });
  };

  return (
    <div className="language-switcher">
      <label className="form-label">
        {UI_TEXT.language_label[state.language]}
      </label>
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <button
          className={`btn ${state.language === 'vi' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => handleLanguageChange('vi')}
          style={{ flex: 1 }}
        >
          🇻🇳 Tiếng Việt
        </button>
        <button
          className={`btn ${state.language === 'en' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => handleLanguageChange('en')}
          style={{ flex: 1 }}
        >
          🇺🇸 English
        </button>
      </div>
    </div>
  );
};

export default LanguageSwitcher;