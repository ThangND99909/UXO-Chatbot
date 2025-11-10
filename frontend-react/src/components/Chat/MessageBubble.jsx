import React from 'react';
import ReactMarkdown from 'react-markdown';

const MessageBubble = ({ message, isLoading = false }) => {
  const { role, content, image, timestamp } = message;

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // Khi đang loading (spinner)
  if (isLoading) {
    return (
      <div className="message-bubble message-assistant">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div className="spinner"></div>
          <span>{content}</span>
        </div>
        <div
          style={{
            fontSize: '0.8rem',
            color: '#666',
            marginTop: '0.25rem',
          }}
        >
          {formatTime(timestamp)}
        </div>
      </div>
    );
  }

  return (
    <div className={`message-bubble message-${role}`}>
      {/* ✅ Nếu là ảnh thì hiển thị ảnh */}
      {image && (
        <img
          src={image}
          alt="user upload"
          className="message-image"
        />
      )}

      {/* ✅ Nếu có nội dung text thì render markdown */}
      {content && (
        <ReactMarkdown>{content}</ReactMarkdown>
      )}

      <div
        style={{
          fontSize: '0.8rem',
          opacity: 0.7,
          marginTop: '0.25rem',
          textAlign: role === 'user' ? 'right' : 'left',
        }}
      >
        {formatTime(timestamp)}
      </div>
    </div>
  );
};

export default MessageBubble;
