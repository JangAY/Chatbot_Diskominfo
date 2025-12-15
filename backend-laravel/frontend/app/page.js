// frontend/app/page.js

'use client';
import { useState, useEffect } from 'react';
import ChatWindow from '@/components/ChatWindow';

// PERBAIKAN: Arahkan ke Laravel, bukan Python langsung
// Pastikan Laravel berjalan di port 8000
const API_URL = 'http://localhost:8000/api/chat'; 

export default function Home() {
  const [currentConversation, setCurrentConversation] = useState(null);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [loadingChat, setLoadingChat] = useState(false);
  const [quickReplies, setQuickReplies] = useState([]);

  // Default Quick Replies
  const showDefaultQuickReplies = () => {
    if (!currentConversation) { 
      setQuickReplies([
        { label: "Apa saja dataset yang tersedia?", value: "Apa saja dataset yang tersedia?" },
        { label: "Tampilkan jumlah penduduk miskin.", value: "Tampilkan jumlah penduduk miskin tahun 2023" },
        { label: "Apa itu Stunting?", value: "Jelaskan apa itu stunting dan penyebabnya" }, // Contoh pertanyaan umum
        { label: "Wisata di Garut", value: "Sebutkan tempat wisata populer di Garut" } // Contoh pertanyaan umum
      ]);
    }
  };

  useEffect(() => {
    showDefaultQuickReplies();
  }, []);

  const handleSendMessage = async (message) => {
    if (!message.trim()) return;

    setLoadingChat(true);
    setQuickReplies([]); 

    const userMessage = { 
      id: `user-${Date.now()}`, 
      sender: 'user', 
      content: message 
    };

    // Update UI User Message
    const newConversation = currentConversation 
      ? { ...currentConversation, messages: [...currentConversation.messages, userMessage] }
      : { id: 'conv-1', title: 'Satu Data Garut', messages: [userMessage] };
      
    setCurrentConversation(newConversation);
    
    if (!currentConversationId) {
        setCurrentConversationId('conv-1'); 
    }

try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message }), 
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();

      // Respons Bot
      const botMessage = {
        id: `bot-${Date.now()}`,
        sender: 'bot',
        content: data.reply, // Python sekarang selalu mengembalikan 'reply' (baik data maupun umum)
        options: data.related_queries || []
      };

      setCurrentConversation(prev => ({
        ...prev,
        messages: [...prev.messages, botMessage],
      }));

      // Update Quick Replies jika ada saran dari backend
      if (data.newQuickReplies && data.newQuickReplies.length > 0) {
        setQuickReplies(data.newQuickReplies);
      } else {
        setQuickReplies([]);
      }

    } catch (error) {
      console.error("Gagal mengirim pesan:", error);
      const errorBotMessage = {
        id: `bot-error-${Date.now()}`,
        sender: 'bot',
        content: `**Maaf, terjadi kesalahan koneksi.**\n\nPastikan server backend (Laravel & Python) sedang berjalan.`
      };
      setCurrentConversation(prev => ({
        ...prev,
        messages: [...prev.messages, errorBotMessage],
      }));
    } finally {
      setLoadingChat(false);
    }
  };

// Fungsi helper untuk klik opsi di dalam chat bubble
  const handleOptionClick = (optionValue) => {
      handleSendMessage(optionValue);
  };

  const handleQuickResponse = (optionValue) => {
      handleSendMessage(optionValue);
  };

  return (
    <main className="flex h-screen w-full bg-gray-900">
      <div className="flex-1">
        <ChatWindow
          conversation={currentConversation}
          onSendMessage={handleSendMessage}
          loading={loadingChat}
          onQuickResponse={handleQuickResponse} 
          quickReplies={quickReplies}
          // TAMBAHAN: Pass handler ke ChatWindow
          onOptionClick={handleOptionClick} 
        />
      </div>
    </main>
  );
}

//   const handleQuickResponse = (message) => {
//     handleSendMessage(message);
//   };

//   return (
//     <main className="flex h-screen w-full bg-gray-900">
//       <div className="flex-1">
//         <ChatWindow
//           conversation={currentConversation}
//           onSendMessage={handleSendMessage}
//           loading={loadingChat}
//           onQuickResponse={handleQuickResponse} 
//           quickReplies={quickReplies}
//         />
//       </div>
//     </main>
//   );
// }