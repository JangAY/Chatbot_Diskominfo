// components/ChatWindow.js

import { useState, useRef, useEffect } from 'react';
import Image from 'next/image';
import ReactMarkdown from 'react-markdown'; // <-- Sudah ada
import remarkGfm from 'remark-gfm'; // <-- Sudah ada (untuk tabel/link)
import rehypeRaw from 'rehype-raw'; // <-- TAMBAHAN BARU (untuk HTML seperti <hr>)

// Komponen untuk pesan individual (SUDAH DIPERBAIKI)
const Message = ({ sender, content }) => {
    const isUser = sender === 'user';
    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
            {!isUser && (
                <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center mr-3 flex-shrink-0 shadow-md">
                    <span className="text-white font-bold">AI</span>
                </div>
            )}
            <div
                className={`max-w-xl p-4 rounded-lg shadow-md ${
                    isUser ? 'bg-blue-500 text-white rounded-br-none' : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none'
                }`}
            >
                {isUser ? (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{content}</div>
                ) : (
                    // --- PERBAIKAN UI DI SINI ---
                    // Kita menambahkan 'rehypeRaw' dan kelas 'prose' untuk styling
                    <div className="prose prose-sm max-w-none prose-table:w-full prose-table:border-collapse prose-tr:border-b prose-td:py-2 prose-td:px-3 prose-th:px-3 prose-a:text-blue-600 hover:prose-a:underline">
                        <ReactMarkdown 
                            remarkPlugins={[remarkGfm]} 
                            rehypePlugins={[rehypeRaw]}
                        >
                            {content}
                        </ReactMarkdown>
                    </div>
                )}
            </div>
        </div>
    );
};


// Komponen utama ChatWindow
// --- PERBAIKAN: Memberi nilai default ke quickReplies = [] ---
export default function ChatWindow({ conversation, onSendMessage, loading, onQuickResponse, quickReplies = [] }) {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [conversation?.messages]);

    const handleSend = (e) => {
        e.preventDefault();
        if (input.trim() && !loading) {
            onSendMessage(input);
            setInput('');
        }
    };

    // Tampilan Selamat Datang (jika tidak ada percakapan)
    if (!conversation && !loading) {
        return (
            <div className="w-full flex flex-col items-center justify-center h-screen bg-gray-50 p-4">
                <div className="text-center">
                    <div className="w-16 h-16 rounded-full bg-blue-500 flex items-center justify-center mx-auto mb-4 shadow-lg">
                        <span className="text-white font-bold text-2xl">SDG</span>
                    </div>
                    <h1 className="text-3xl font-bold text-gray-800">Satu Data Garut</h1>
                    <p className="text-gray-500 mt-2">Tanyakan apapun tentang data di Kabupaten Garut</p>
                </div>
                {/* Bagian Input Tetap Ditampilkan */}
                <div className="w-full max-w-2xl mt-auto p-4">
                     {/* Tombol dinamis */}
                     <div className="grid grid-cols-2 gap-3 mb-3">
                        {/* --- PERBAIKAN: (quickReplies || []) untuk keamanan --- */}
                        {(quickReplies || []).map((reply, index) => (
                            <button
                                key={index}
                                onClick={() => onQuickResponse(reply.value)} // Kirim 'value'
                                className="bg-white border border-gray-300 text-gray-700 text-left p-3 rounded-lg hover:bg-gray-100 transition-colors duration-200 text-sm shadow-sm"
                            >
                                {reply.label} {/* Tampilkan 'label' */}
                            </button>
                        ))}
                    </div>
                    <form onSubmit={handleSend} className="flex items-center bg-white border border-gray-300 rounded-lg shadow-sm p-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ketik pertanyaan anda di sini..."
                            className="flex-grow bg-transparent border-none focus:ring-0 text-gray-800 placeholder-gray-400 p-2"
                        />
                        <button type="submit" className="btn btn-primary btn-circle" disabled={!input.trim() || loading}>
                           <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086L2.279 16.76a.75.75 0 00.95.826l16-5.333a.75.75 0 000-1.418l-16-5.333z" /></svg>
                        </button>
                    </form>
                </div>
            </div>
        );
    }

    return (
        <div className="w-full flex flex-col h-screen bg-gray-50">
            {/* Header Chat */}
            <div className="p-4 border-b border-gray-200 bg-white shadow-sm">
                <h2 className="text-xl font-semibold text-gray-800">{conversation?.title || "Satu Data Garut"}</h2>
            </div>
            
            {/* Area Pesan */}
            <div className="flex-1 p-6 overflow-y-auto custom-scrollbar">
                {loading && !conversation?.messages && (
                    <div className="flex justify-center items-center h-full">
                        <span className="loading loading-dots loading-lg text-primary"></span>
                        <p className="text-gray-500 ml-3">Memuat percakapan...</p>
                    </div>
                )}
                {conversation?.messages?.map((msg, index) => (
                    <Message key={msg.id || `msg-${index}`} sender={msg.sender} content={msg.content} />
                ))}
                {loading && conversation?.messages && (
                    <div className="flex justify-start mb-4">
                        <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center mr-3 flex-shrink-0 shadow-md">
                            <span className="text-white font-bold">AI</span>
                        </div>
                        <div className="max-w-xl p-4 rounded-lg bg-white text-gray-800 border border-gray-200 rounded-bl-none shadow-md">
                            <span className="loading loading-dots loading-sm"></span>
                        </div>
                    </div>
                )}
                 <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-gray-50 border-t border-gray-200">
                 {/* --- PERBAIKAN: Cek null/undefined sebelum .length --- */}
                 {!loading && quickReplies && quickReplies.length > 0 && (
                    <div className="grid grid-cols-2 gap-3 mb-3 max-w-2xl mx-auto">
                        {quickReplies.map((reply, index) => (
                            <button
                                key={index}
                                onClick={() => onQuickResponse(reply.value)} // Kirim 'value'
                                className="bg-white border border-gray-300 text-gray-700 text-left p-3 rounded-lg hover:bg-gray-100 transition-colors duration-200 text-sm shadow-sm"
                            >
                                {reply.label} {/* Tampilkan 'label' */}
                            </button>
                        ))}
                    </div>
                 )}
                {/* Form Input */}
                <div className="max-w-2xl mx-auto">
                    <form onSubmit={handleSend} className="flex items-center bg-white border border-gray-300 rounded-lg shadow-sm p-2">
                         <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ketik pertanyaan anda di sini..."
                            className="flex-grow bg-transparent border-none focus:ring-0 text-gray-800 placeholder-gray-400 p-2"
                        />
                        <button type="submit" className="btn btn-primary btn-circle" disabled={!input.trim() || loading}>
                           <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086L2.279 16.76a.75.75 0 00.95.826l16-5.333a.75.75 0 000-1.418l-16-5.333z" /></svg>
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}