import React from 'react';
import { useApp } from '../context/AppContext';
import ChatInterface from '../components/Chat/ChatInterface';
import { UI_TEXT } from '../utils/constants';

const Home = () => {
  const { state } = useApp();

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>{UI_TEXT.title[state.language]}</h1>
        <p>{UI_TEXT.main_page_intro[state.language]}</p>
      </div>
      <ChatInterface />
    </div>
  );
};

export default Home;