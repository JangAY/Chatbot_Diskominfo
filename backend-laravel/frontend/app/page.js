'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar'; // Pastikan path komponen ini benar
import ChatWindow from '@/components/ChatWindow'; // Pastikan path komponen ini benar

// --- PERUBAHAN PENTING ---
// Arahkan API ke backend Laravel Anda, bukan ke server Python secara langsung.
const LARAVEL_API_URL = 'http://localhost:8000/api/chat'; 

export default function Home() {
    const [conversation, setConversation] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSendMessage = async (messageText) => {
        if (loading) return;

        const userMessage = {
            id: `user-${Date.now()}`,
            sender: 'user',
            content: messageText,
        };

        // Tampilkan pesan pengguna di UI terlebih dahulu
        if (!conversation) {
            setConversation({ id: 'conv-1', messages: [userMessage] });
        } else {
            setConversation(prev => ({
                ...prev,
                messages: [...prev.messages, userMessage],
            }));
        }

        setLoading(true);

        try {
            // Kirim permintaan ke backend Laravel
            const response = await fetch(LARAVEL_API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify({ query: messageText }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Buat pesan dari bot berdasarkan hasil dari Laravel
            const botMessage = {
                id: `bot-${Date.now()}`,
                sender: 'bot',
                content: data.reply || "Maaf, saya tidak bisa memproses permintaan itu saat ini.",
            };

            // Tambahkan pesan bot ke UI
            setConversation(prev => ({
                ...prev,
                messages: [...prev.messages, botMessage],
            }));

        } catch (error) {
            console.error("Gagal mengambil respons chatbot dari Laravel:", error);

            const errorMessage = {
                id: `error-${Date.now()}`,
                sender: 'bot',
                content: `Maaf, terjadi kesalahan koneksi ke server. Pastikan semua layanan berjalan. (${error.message})`,
            };

            setConversation(prev => ({
                ...prev,
                messages: [...prev.messages, errorMessage],
            }));
        } finally {
            setLoading(false);
        }
    };

    const startNewChat = () => {
        setConversation(null);
    };

    const handleQuickResponse = (text) => {
        handleSendMessage(text);
    };

    return (
        <main className="flex h-screen bg-gray-100 font-inter">
            {/* Sidebar Anda, jika ada */}
            <Sidebar
                history={[]}
                onSelectConversation={() => {}}
                onFilterChange={() => {}}
                onNewChat={startNewChat}
                loading={false}
            />
            <ChatWindow
                conversation={conversation}
                onSendMessage={handleSendMessage}
                loading={loading}
                onQuickResponse={handleQuickResponse}
            />
        </main>
    );
}
