import React from 'react';
import { useApp } from '../../context/AppContext';
import { UI_TEXT } from '../../utils/constants';

const HotlineInfo = () => {
  const { state } = useApp();

  return (
    <div className="hotline-info">
      <h3 className="sidebar-subtitle">📞 {UI_TEXT.hotline_emergency[state.language]}</h3>
      <div className="alert alert-warning">
        {UI_TEXT.hotline[state.language].split('\n').map((line, index) => (
          <div key={index} style={{ marginBottom: '0.5rem' }}>
            {line}
          </div>
        ))}
      </div>
    </div>
  );
};

export default HotlineInfo;