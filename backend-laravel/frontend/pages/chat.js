// pages/chat.js (atau di mana pun Anda ingin menampilkan chat)

import { useState } from 'react';
import ChatWindow from '../components/ChatWindow'; // Sesuaikan path jika perlu

export default function ChatPage() {
    // State untuk menyimpan seluruh percakapan
    const [conversation, setConversation] = useState(null); 
    // State untuk loading indicator
    const [loading, setLoading] = useState(false);

    // Fungsi ini dipanggil saat pengguna mengirim pesan
    const handleSendMessage = async (messageText) => {
        // Jangan kirim pesan baru jika sedang loading
        if (loading) return;

        const userMessage = {
            id: Date.now(), // Unique key
            sender: 'user',
            content: messageText,
        };

        // Langsung tampilkan pesan pengguna di UI
        setConversation(prev => {
            if (!prev) { // Jika ini pesan pertama
                return { id: 'conv-1', messages: [userMessage] };
            }
            return { ...prev, messages: [...prev.messages, userMessage] };
        });

        setLoading(true);

        try {
            // Panggil API backend Laravel Anda
            const response = await fetch('http://localhost:8000/api/chat', { // <-- SESUAIKAN URL BACKEND ANDA
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify({ query: messageText }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            const botMessage = {
                id: Date.now() + 1,
                sender: 'bot',
                content: data.reply || "Maaf, terjadi kesalahan pada server.",
            };

            // Tambahkan jawaban dari bot ke UI
            setConversation(prev => ({
                ...prev,
                messages: [...prev.messages, botMessage],
            }));

        } catch (error) {
            console.error("Gagal mengambil respons chatbot:", error);
            const errorMessage = {
                id: Date.now() + 1,
                sender: 'bot',
                content: "Maaf, saya sedang mengalami gangguan. Silakan coba lagi nanti.",
            };
            // Tampilkan pesan error di UI
            setConversation(prev => ({
                ...prev,
                messages: [...prev.messages, errorMessage],
            }));
        } finally {
            // Hentikan loading
            setLoading(false);
        }
    };

    // Fungsi untuk menangani klik pada tombol respons cepat
    const handleQuickResponse = (text) => {
        // Cukup panggil handleSendMessage dengan teks dari tombol
        handleSendMessage(text);
    };

    return (
        <div className="flex h-screen">
            {/* Sidebar bisa ditambahkan di sini jika perlu */}
            {/* <Sidebar /> */}
            
            <ChatWindow
                conversation={conversation}
                onSendMessage={handleSendMessage}
                loading={loading}
                onQuickResponse={handleQuickResponse}
            />
        </div>
    );
}