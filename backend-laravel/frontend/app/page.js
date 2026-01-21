// frontend/app/page.js

'use client';

import { useState, useEffect } from 'react';
import ChatWindow from '@/components/ChatWindow';
import Sidebar from '@/components/Sidebar';

const API_URL = 'http://localhost:8000/api/chat'; 

export default function Home() {
  const [chatHistory, setChatHistory] = useState([]); 
  const [activeChatId, setActiveChatId] = useState(null); 
  const [isSidebarOpen, setIsSidebarOpen] = useState(true); 
  const [loadingChat, setLoadingChat] = useState(false);
  const [quickReplies, setQuickReplies] = useState([]);

  // --- 1. LOAD FROM LOCAL STORAGE ---
  useEffect(() => {
    const storedHistory = localStorage.getItem('garut_data_history');
    if (storedHistory) {
      try {
        const parsed = JSON.parse(storedHistory);
        setChatHistory(parsed);
        if (parsed.length > 0) {
          setActiveChatId(parsed[0].id);
        } else {
          createNewChat();
        }
      } catch (e) {
        createNewChat();
      }
    } else {
      createNewChat();
    }
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  }, []);

  // --- 2. SAVE TO LOCAL STORAGE ---
  useEffect(() => {
    if (chatHistory.length > 0) {
      localStorage.setItem('garut_data_history', JSON.stringify(chatHistory));
    }
  }, [chatHistory]);

  const getDefaultQuickReplies = () => [
    { label: "Data statistik sektoral apa saja yang ada?", value: "Apa saja dataset yang tersedia?" },
    { label: "Jumlah penduduk miskin tahun 2023?", value: "Tampilkan jumlah penduduk miskin tahun 2023" },
    { label: "Apa itu Stunting dan penyebabnya?", value: "Jelaskan apa itu stunting dan penyebabnya" },
    { label: "Objek Wisata populer di Garut?", value: "Sebutkan tempat wisata populer di Garut" }
  ];

  const createNewChat = () => {
    const newId = `chat-${Date.now()}`;
    const newChat = {
      id: newId,
      title: 'Percakapan Baru',
      messages: [],
      createdAt: new Date().toISOString()
    };
    
    setChatHistory(prev => [newChat, ...prev]);
    setActiveChatId(newId);
    setQuickReplies(getDefaultQuickReplies());
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  };

  const deleteChat = (chatId) => {
    const updatedHistory = chatHistory.filter(c => c.id !== chatId);
    setChatHistory(updatedHistory);
    localStorage.setItem('garut_data_history', JSON.stringify(updatedHistory)); 

    if (activeChatId === chatId) {
      if (updatedHistory.length > 0) setActiveChatId(updatedHistory[0].id);
      else createNewChat();
    }
  };

  const selectChat = (chatId) => {
    setActiveChatId(chatId);
    setQuickReplies([]); 
  };

  const activeConversation = chatHistory.find(c => c.id === activeChatId) || null;

  const handleSendMessage = async (message) => {
    if (!message.trim()) return;

    setLoadingChat(true);
    setQuickReplies([]); // Bersihkan quick replies lama
    
    const userMessage = { 
      id: `user-${Date.now()}`, 
      sender: 'user', 
      content: message 
    };

    setChatHistory(prev => {
      return prev.map(chat => {
        if (chat.id === activeChatId) {
          const isFirstMessage = chat.messages.length === 0;
          const newTitle = isFirstMessage 
            ? (message.length > 30 ? message.substring(0, 30) + '...' : message) 
            : chat.title;
          
          return { ...chat, title: newTitle, messages: [...chat.messages, userMessage] };
        }
        return chat;
      });
    });

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message }), 
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();

      // --- LOGIKA UTAMA PERUBAHAN ---
      // Cek apakah ada 'newQuickReplies' dari backend (biasanya list sektor ada di sini)
      let botOptions = data.related_queries || [];
      let bottomQuickReplies = [];

      // Jika jumlah quick replies BANYAK (> 4), kita asumsikan ini adalah LIST DATA/SEKTOR
      // Maka kita pindahkan ke 'botOptions' agar dirender sebagai GRID CARD di dalam chat
      if (data.newQuickReplies && data.newQuickReplies.length > 4) {
         botOptions = [...botOptions, ...data.newQuickReplies];
         bottomQuickReplies = []; // Kosongkan bawah agar tidak duplikat
      } else if (data.newQuickReplies) {
         // Jika sedikit, tetap taruh di bawah sebagai tombol biasa
         bottomQuickReplies = data.newQuickReplies;
      }

      const botMessage = {
        id: `bot-${Date.now()}`,
        sender: 'bot',
        content: data.reply, 
        options: botOptions // Opsi masuk ke sini agar jadi Card
      };

      setChatHistory(prev => {
        return prev.map(chat => {
          if (chat.id === activeChatId) {
            return { ...chat, messages: [...chat.messages, botMessage] };
          }
          return chat;
        });
      });

      setQuickReplies(bottomQuickReplies);

    } catch (error) {
      console.error("Gagal mengirim pesan:", error);
      const errorBotMessage = {
        id: `bot-error-${Date.now()}`,
        sender: 'bot',
        content: `**Maaf, terjadi kesalahan koneksi.**\n\nPastikan server backend sedang berjalan.`
      };
      setChatHistory(prev => prev.map(c => c.id === activeChatId ? {...c, messages: [...c.messages, errorBotMessage]} : c));
    } finally {
      setLoadingChat(false);
    }
  };

  const handleOptionClick = (val) => handleSendMessage(val);
  const handleQuickResponse = (val) => handleSendMessage(val);

  return (
    <main className="flex h-screen w-full bg-gray-900 overflow-hidden font-sans">
      <Sidebar 
        isOpen={isSidebarOpen}
        toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        onNewChat={createNewChat}
        chatHistory={chatHistory}
        activeChatId={activeChatId}
        onSelectChat={selectChat}
        onDeleteChat={deleteChat}
      />

      <div className="flex-1 flex flex-col h-full relative shadow-2xl">
        <ChatWindow
          conversation={activeConversation}
          onSendMessage={handleSendMessage}
          loading={loadingChat}
          onQuickResponse={handleQuickResponse} 
          quickReplies={quickReplies}
          onOptionClick={handleOptionClick} 
          toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          isSidebarOpen={isSidebarOpen}
        />
      </div>
    </main>
  );
}