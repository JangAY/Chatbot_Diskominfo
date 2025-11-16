<<<<<<< Updated upstream
=======
// app/page.js
>>>>>>> Stashed changes
'use client';
import { useState, useEffect } from 'react';
// Hapus import Sidebar
import ChatWindow from '@/components/ChatWindow';

const API_URL = 'http://127.0.0.1:8000/api'; // URL Backend Laravel Anda

export default function Home() {
  // ... (Semua state Anda tetap sama)
  const [history, setHistory] = useState([]);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [loadingChat, setLoadingChat] = useState(false);
  const [quickReplies, setQuickReplies] = useState([]); // State untuk quick replies

  // Fungsi untuk memuat riwayat
  const fetchHistory = async () => {
    setLoadingHistory(true);
    try {
      const res = await fetch(`${API_URL}/history`);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setHistory(data);

      // Set quick replies default HANYA jika tidak ada percakapan aktif
      if (!currentConversationId) {
        setQuickReplies([
            { label: "Apa saja dataset yang tersedia?", value: "Apa saja dataset yang tersedia?" },
            { label: "Tampilkan jumlah penduduk Garut.", value: "Tampilkan jumlah penduduk Garut." },
            { label: "Bagaimana cara menggunakan portal ini?", value: "Bagaimana cara menggunakan portal ini?" },
            { label: "Jelaskan metadata yang ada.", value: "Jelaskan metadata yang ada." }
        ]);
      }
    } catch (error) {
      console.error("Gagal memuat riwayat:", error);
      // Mungkin set quick replies default di sini juga jika fetch gagal
      setQuickReplies([
        { label: "Apa saja dataset yang tersedia?", value: "Apa saja dataset yang tersedia?" },
        { label: "Tampilkan jumlah penduduk Garut.", value: "Tampilkan jumlah penduduk Garut." },
        { label: "Bagaimana cara menggunakan portal ini?", value: "Bagaimana cara menggunakan portal ini?" },
        { label: "Jelaskan metadata yang ada.", value: "Jelaskan metadata yang ada." }
      ]);
    } finally {
      setLoadingHistory(false);
    }
  };

  // Fungsi untuk memuat percakapan spesifik
  const fetchConversation = async (id) => {
    setLoadingChat(true);
    setQuickReplies([]); // Kosongkan quick replies saat memuat
    if (!id) {
      setCurrentConversation(null);
      setCurrentConversationId(null);
      setLoadingChat(false);
      fetchHistory(); // Panggil fetchHistory untuk reset quick replies
      return;
    }
    try {
      const res = await fetch(`${API_URL}/conversation/${id}`);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setCurrentConversation(data);
      setCurrentConversationId(id);
      // Percakapan lama biasanya tidak punya quick replies
      setQuickReplies([]); 
    } catch (error) {
      console.error("Gagal memuat percakapan:", error);
    } finally {
      setLoadingChat(false);
    }
  };

  // Fungsi untuk memulai chat baru
  const startNewChat = () => {
    setCurrentConversation(null);
    setCurrentConversationId(null);
    fetchHistory(); // Panggil fetchHistory untuk reset quick replies ke default
  };

  // Muat riwayat saat komponen pertama kali dimuat
  useEffect(() => {
    fetchHistory();
  }, []); // Hanya dijalankan sekali

  // Fungsi mengirim pesan
  const handleSendMessage = async (message) => {
    if (!message.trim()) return;
    setLoadingChat(true); // Tampilkan loading AI
    setQuickReplies([]); // Sembunyikan quick replies saat mengirim

    const userMessage = { id: `temp-${Date.now()}-user`, sender: 'user', content: message };
    
    setCurrentConversation(prev => ({
      ...prev,
      messages: prev && prev.messages ? [...prev.messages, userMessage] : [userMessage],
      title: prev?.title || 'New Conversation'
    }));

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ message, conversation_id: currentConversationId }),
      });

      if (!res.ok) {
          const errorText = await res.text();
          console.error("Gagal mengirim pesan. Status:", res.status, "Respons Server:", errorText);
          // Tampilkan pesan error ke pengguna
          const errorBotMessage = {
            id: `temp-${Date.now()}-bot-error`,
            sender: 'bot',
            content: `**Maaf, terjadi kesalahan.**\n\nServer tidak merespons dengan benar (Status: ${res.status}). Silakan coba lagi nanti.`
          };
          setCurrentConversation(prev => ({
            ...prev,
            messages: [...prev.messages.filter(m => m.id !== userMessage.id), userMessage, errorBotMessage],
          }));
          throw new Error(`Gagal mengirim pesan (Status: ${res.status})`);
      }

      const data = await res.json();

      if (currentConversationId !== data.conversation_id) {
        // Ini adalah percakapan baru, panggil fetchConversation untuk data lengkap
        setCurrentConversationId(data.conversation_id);
        await fetchConversation(data.conversation_id);
        // await fetchHistory(); // panggil jika ingin update sidebar (jika ada)
      } else {
        // Ini adalah percakapan yang ada, update pesannya
        setCurrentConversation(prev => ({
          ...prev,
          messages: [
            ...prev.messages.filter(m => m.id !== userMessage.id),
            data.user_message, // Gunakan data asli dari server
            data.bot_reply
          ],
        }));
      }
      
      // --- PERUBAHAN DI SINI ---
      // Blok kode di bawah ini (yang mengembalikan quick replies baru)
      // telah dihapus agar quick replies tidak muncul lagi setelah chat pertama.
      /*
      setQuickReplies([
        { label: "Tanyakan hal lain?", value: "Apa lagi yang bisa kamu lakukan?" },
        { label: "Terima kasih!", value: "Terima kasih!" },
      ]);
      */
      // --- AKHIR PERUBAHAN ---


    } catch (error) {
      console.error("Gagal mengirim pesan:", error);
      // Pastikan pesan error ditampilkan jika fetch gagal total
      if (!currentConversation?.messages?.find(m => m.id.includes('bot-error'))) {
        const networkErrorBotMessage = {
          id: `temp-${Date.now()}-bot-error`,
          sender: 'bot',
          content: `**Maaf, koneksi ke server gagal.**\n\nPastikan server backend Anda berjalan dan coba lagi.`
        };
        setCurrentConversation(prev => ({
          ...prev,
          messages: [...prev.messages.filter(m => m.id !== userMessage.id), userMessage, networkErrorBotMessage],
        }));
      }
    } finally {
        setLoadingChat(false); // Sembunyikan loading AI
    }
  };

  // Handle respons cepat (quick response buttons)
  const handleQuickResponse = (text) => {
    handleSendMessage(text);
  };


  return (
    // --- TEMA MERAH ---
    <main className="h-screen bg-red-100 font-inter"> 
      
      {/* Sidebar sudah dihapus */}

<<<<<<< Updated upstream
            const data = await response.json(); 

            const botMessage = {
                id: `bot-${Date.now()}`,
                sender: 'bot',
                content: data.reply,
            };

            // Cek tombol dinamis
            if (data.newQuickReplies && data.newQuickReplies.length > 0) {
                setQuickReplies(data.newQuickReplies);
            } else {
                setQuickReplies(DEFAULT_QUICK_REPLIES);
            }
            
            // --- PERUBAHAN: Logika riwayat dihapus ---
            // Cukup tambahkan pesan bot ke state sesi saat ini
            setConversation(prev => ({
                ...prev,
                messages: [...prev.messages, botMessage],
            }));

        } catch (error) {
            console.error("Gagal mengambil respons chatbot:", error);
            const errorMessage = {
                id: `error-${Date.now()}`,
                sender: 'bot',
                content: `Maaf, terjadi kesalahan koneksi ke server. (${error.message})`,
            };
            setConversation(prev => ({
                ...prev,
                messages: [...prev.messages, errorMessage],
            }));
            setQuickReplies(DEFAULT_QUICK_REPLIES); 
        } finally {
            setLoadingReply(false);
        }
    };

    const handleNewChat = () => {
        // Cukup bersihkan state sesi saat ini
        setConversation(null);
        setQuickReplies(DEFAULT_QUICK_REPLIES); 
    };
    
    const handleQuickResponse = (value) => {
        handleSendMessage(value);
    };

    return (
        <main className="flex h-screen bg-gray-100 font-inter">
            
            <ChatWindow
                conversation={conversation}
                onSendMessage={handleSendMessage}
                loading={loadingReply} // Hanya loading balasan
                onQuickResponse={handleQuickResponse}
                quickReplies={quickReplies}
            />
        </main>
    );
=======
      <ChatWindow
        conversation={currentConversation}
        onSendMessage={handleSendMessage}
        loading={loadingChat}
        onQuickResponse={handleQuickResponse}
        quickReplies={quickReplies} // Teruskan quick replies ke ChatWindow
      />
    </main>
  );
>>>>>>> Stashed changes
}