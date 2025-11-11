'use client';

import { useState } from 'react'; // Hapus useEffect

import ChatWindow from '@/components/ChatWindow';

const LARAVEL_API_URL = 'http://localhost:8000/api';

const DEFAULT_QUICK_REPLIES = [
    { label: "Apa saja dataset yang tersedia?", value: "Apa saja dataset yang tersedia?" },
    { label: "Tampilkan jumlah penduduk Garut.", value: "Tampilkan jumlah penduduk Garut 2022" },
    { label: "Bagaimana cara menggunakan portal ini?", value: "Bagaimana cara menggunakan portal ini?" },
    { label: "Jelaskan metadata yang ada.", value: "Jelaskan metadata yang ada." }
];

export default function Home() {
    // --- PERUBAHAN: State Jauh Lebih Sederhana ---
    // 'conversation' hanya menyimpan pesan di sesi INI.
    const [conversation, setConversation] = useState(null); 
    const [loadingReply, setLoadingReply] = useState(false);
    const [quickReplies, setQuickReplies] = useState(DEFAULT_QUICK_REPLIES);

    // --- HAPUS: fetchHistory() dan useEffect() ---
    
    // --- HAPUS: handleSelectConversation() ---

    const handleSendMessage = async (messageText) => {
        if (loadingReply) return;

        setLoadingReply(true);
        setQuickReplies([]); 

        const userMessage = {
            id: `user-${Date.now()}`,
            sender: 'user',
            content: messageText,
        };

        // Langsung tambahkan pesan user ke state
        if (!conversation) {
            setConversation({ id: 'session-1', messages: [userMessage] }); // ID Sesi (bisa apa saja)
        } else {
            setConversation(prev => ({
                ...prev,
                messages: [...prev.messages, userMessage],
            }));
        }

        try {
            const response = await fetch(`${LARAVEL_API_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify({ 
                    query: messageText,
                    // conversation_id tidak dikirim lagi
                }),
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error("Gagal mengirim pesan. Status:", response.status, "Respons Server:", errorText);
                throw new Error(`Gagal mengirim pesan (Status: ${response.status})`);
            }

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
}