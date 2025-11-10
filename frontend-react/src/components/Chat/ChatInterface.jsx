import React, { useState, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { apiService } from '../../services/api';
import MessageBubble from './MessageBubble';
import ImageUploader from '../ImageAnalysis/ImageUploader';
import { UI_TEXT } from '../../utils/constants';

const ChatInterface = () => {
  const { state, dispatch } = useApp();
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [state.chatHistory]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;
    
    const userMessage = { role: 'user', content: inputMessage, timestamp: new Date() };
    dispatch({ type: 'ADD_CHAT_MESSAGE', payload: userMessage });
    setInputMessage('');
    setIsLoading(true);
    
    
    try {
      const response = await apiService.sendChat(
        inputMessage, 
        state.sessionId, 
        state.language
      );
      
      const botMessage = { 
        role: 'assistant', 
        content: response.answer, 
        timestamp: new Date() 
      };
      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: botMessage });
      
      // Handle intents
      if (response.intent === 'report_bomb') {
        dispatch({ type: 'SHOW_REPORT_MAP', payload: true });
      }
      
      if (response.intent) {
        dispatch({ type: 'SET_LAST_INTENT', payload: response.intent });
      }
    } catch (error) {
      const errorMessage = { 
        role: 'assistant', 
        content: '❌ Lỗi kết nối đến chatbot.',
        timestamp: new Date()
      };
      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: errorMessage });
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageAnalysis = async (imageFile) => {
    const userMessage = { 
      role: 'user', 
      content: `📸 Đã tải lên ảnh: ${imageFile.name}`,
      timestamp: new Date()
    };
    dispatch({ type: 'ADD_CHAT_MESSAGE', payload: userMessage });
    
    setIsLoading(true);
    try {
      const result = await apiService.detectUXO(imageFile, state.sessionId);
      
      // Add to processed images
      dispatch({ type: 'ADD_PROCESSED_IMAGE', payload: imageFile.name });
      
      // Add detection results to chat
      let detectionMessage = "📊 **Kết quả phân tích ảnh:**\n";
      if (result.detections && result.detections.length > 0) {
        result.detections.forEach(det => {
          detectionMessage += `- ${det.class} (độ tin cậy: ${det.confidence.toFixed(2)})\n`;
        });
      } else {
        detectionMessage += UI_TEXT.no_detection[state.language];
      }
      
      const botMessage = { 
        role: 'assistant', 
        content: detectionMessage,
        timestamp: new Date()
      };
      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: botMessage });
      
      // Save to detection history if admin
      if (state.adminToken && result.detection_id) {
        const newDetection = {
          id: result.detection_id,
          filename: imageFile.name,
          detections: result.detections || [],
          created_at: new Date().toISOString(),
          session_id: state.sessionId
        };
        
        dispatch({ 
          type: 'SET_DETECTION_HISTORY', 
          payload: [...state.adminDetectionHistory, newDetection] 
        });
      }
    } catch (error) {
      const errorMessage = { 
        role: 'assistant', 
        content: `❌ Lỗi khi phân tích ảnh: ${error.message}`,
        timestamp: new Date()
      };
      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: errorMessage });
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageUploadInChat = (file) => {
    if (!file) return;
    const imageUrl = URL.createObjectURL(file);

    const imageMessage = {
      role: 'user',
      content: '',
      image: imageUrl,
      timestamp: new Date(),
    };

    // Hiển thị ảnh trong chat
    dispatch({ type: 'ADD_CHAT_MESSAGE', payload: imageMessage });
  };


  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-messages">
        {state.chatHistory.map((message, index) => (
          <MessageBubble 
            key={index} 
            message={message} 
          />
        ))}
        {isLoading && (
          <MessageBubble 
            message={{ 
              role: 'assistant', 
              content: 'Đang xử lý...',
              timestamp: new Date()
            }} 
            isLoading 
          />
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input-area">
        <ImageUploader onImageUpload={handleImageAnalysis} />
        
        <div className="message-input-container">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder={UI_TEXT.chat_placeholder[state.language]}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
          />

          {/* Nút chọn ảnh */}
          <label htmlFor="chat-file-upload" className="upload-icon">🖼️</label>
          <input
            id="chat-file-upload"
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => handleImageUploadInChat(e.target.files[0])}
          />

          <button 
            onClick={handleSendMessage} 
            disabled={isLoading || !inputMessage.trim()}
            className="btn-send"
          >
            {isLoading ? <div className="spinner"></div> : 'Gửi'}
          </button>
        </div>

      </div>
    </div>
  );
};

export default ChatInterface;