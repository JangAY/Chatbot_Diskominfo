// app/page.js (GANTI SEMUA DENGAN KODE INI)

'use client';
import { useState, useEffect } from 'react';
import ChatWindow from '@/components/ChatWindow'; // ChatWindow Anda (normal atau baru)

// 1. GANTI URL API
// Arahkan ke backend Python/Flask Anda, bukan Laravel
const API_URL = 'http://127.0.0.1:5000/api/chat'; 

export default function Home() {
  // State Anda sudah benar
  const [currentConversation, setCurrentConversation] = useState(null);
  const [currentConversationId, setCurrentConversationId] = useState(null); // Kita akan atur ini
  const [loadingChat, setLoadingChat] = useState(false);
  const [quickReplies, setQuickReplies] = useState([]);

  // Fungsi untuk menampilkan quick replies default
  const showDefaultQuickReplies = () => {
    // Tampilkan hanya jika belum ada percakapan
    if (!currentConversation) { 
      setQuickReplies([
        { label: "Apa saja dataset yang tersedia?", value: "Apa saja dataset yang tersedia?" },
        { label: "Tampilkan jumlah penduduk miskin tahun 2024.", value: "Tampilkan jumlah penduduk miskin tahun 2024." },
        { label: "Bagaimana cara menggunakan portal ini?", value: "Bagaimana cara menggunakan portal ini?" },
        { label: "Jelaskan metadata yang ada.", value: "Jelaskan metadata yang ada." }
      ]);
    }
  };

  // Muat quick replies default saat pertama kali buka
  useEffect(() => {
    showDefaultQuickReplies();
  }, []); // Hanya dijalankan sekali

  
  // 2. GANTI TOTAL FUNGSI handleSendMessage
  // Ini adalah perbaikan utama. Logika ini sekarang sesuai
  // dengan respons JSON {"reply": "..."} dari chatbot.py
  const handleSendMessage = async (message) => {
    if (!message.trim()) return;

    setLoadingChat(true); // Tampilkan loading "AI sedang berfikir"
    setQuickReplies([]); // Sembunyikan quick replies

    const userMessage = { 
      id: `user-${Date.now()}`, 
      sender: 'user', 
      content: message 
    };

    // Tampilkan pesan user segera
    const newConversation = currentConversation 
      ? { ...currentConversation, messages: [...currentConversation.messages, userMessage] }
      : { id: 'conv-1', title: 'Satu Data Garut', messages: [userMessage] };
      
    setCurrentConversation(newConversation);
    
    // Set ID percakapan agar layar sambutan hilang
    if (!currentConversationId) {
        setCurrentConversationId('conv-1'); 
    }

    try {
      // Kirim ke API_URL (Python)
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        // Backend Python hanya butuh 'query'
        body: JSON.stringify({ query: message }), 
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      // Ambil data respons: {"reply": "...", "newQuickReplies": [...]}
      const data = await res.json();

      // Buat objek pesan bot
      const botMessage = {
        id: `bot-${Date.now()}`,
        sender: 'bot',
        content: data.reply // Ambil 'reply' dari JSON
      };

      // Update percakapan dengan jawaban bot
      setCurrentConversation(prev => ({
        ...prev,
        messages: [...prev.messages, botMessage],
      }));

      // Cek apakah backend mengirim quick replies baru
      if (data.newQuickReplies && data.newQuickReplies.length > 0) {
        setQuickReplies(data.newQuickReplies);
      } else {
        // Jika tidak ada, kembali ke default (jika ini chat baru)
        // atau biarkan kosong (jika chat sudah berjalan)
        // Untuk Skenario Anda: biarkan kosong setelah jawaban
        setQuickReplies([]);
      }

    } catch (error) {
      console.error("Gagal mengirim pesan:", error);
      // Tampilkan pesan error di chat
      const errorBotMessage = {
        id: `bot-error-${Date.now()}`,
        sender: 'bot',
        content: `**Maaf, terjadi kesalahan koneksi.**\n\nTidak dapat terhubung ke server chatbot. (Error: ${error.message})`
      };
      setCurrentConversation(prev => ({
        ...prev,
        messages: [...prev.messages, errorBotMessage],
      }));
    } finally {
      setLoadingChat(false); // Sembunyikan loading
    }
  };

  // Fungsi untuk menangani klik quick reply
  const handleQuickResponse = (message) => {
    // Kirim pesan seolah-olah user yang mengetik
    handleSendMessage(message);
  };


  // RENDER UI
  // Kita tidak perlu mengubah bagian ini.
  // Selama file ChatWindow.js baru Anda masih menerima props:
  // (conversation, onSendMessage, loading, onQuickResponse, quickReplies)
  // maka ini akan berfungsi.
  return (
    <main className="flex h-screen w-full">
      {/* Komponen Sidebar Anda yang lama mungkin ada di sini.
        Tapi berdasarkan file lama, Anda menghapusnya untuk layout fullscreen.
        Itu tidak masalah.
      */}
      
      {/* Bagian ini memuat komponen ChatWindow Anda */}
      <div className="flex-1">
        <ChatWindow
          conversation={currentConversation}
          onSendMessage={handleSendMessage}
          loading={loadingChat}
          onQuickResponse={handleQuickResponse} 
          quickReplies={quickReplies}
        />
      </div>
    </main>
  );
}