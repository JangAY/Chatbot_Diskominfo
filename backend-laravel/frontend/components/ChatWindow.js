// components/ChatWindow.js

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import Image from 'next/image';
// HAPUS: import { ThemeSwitcher } from './ThemeSwitcher';

// Komponen untuk pesan individual
const Message = ({ sender, content, options, onOptionClick }) => {
    const isUser = sender === 'user';
return (
        <div className={`flex flex-col mb-4 ${isUser ? 'items-end' : 'items-start'}`}>
            <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
                {!isUser && (
                    <div className="w-10 h-10 rounded-full bg-yellow-400 flex items-center justify-center mr-3 flex-shrink-0 shadow-md">
                        <span className="text-black font-bold">AI</span>
                    </div>
                )}
                <div
                    className={`max-w-xl p-4 rounded-lg shadow-md ${
                        isUser ? 'bg-yellow-400 text-black rounded-br-none' 
                               : 'bg-gray-800 text-gray-100 border border-gray-700 rounded-bl-none'
                    }`}
                >
                    {isUser ? (
                        <div className="text-black" style={{ whiteSpace: 'pre-wrap' }}>{content}</div>
                    ) : (
                        <div className="prose prose-sm prose-invert max-w-none">
                            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                                {content}
                            </ReactMarkdown>
                        </div>
                    )}
                </div>
            </div>

            {/* AREA LIST OPSI KLIK (Hanya untuk Bot) */}
            {!isUser && options && options.length > 0 && (
                <div className="ml-14 mt-2 flex flex-col space-y-2 w-full max-w-md">
                    <p className="text-xs text-gray-400 mb-1">Pilih data spesifik:</p>
                    {options.map((opt, idx) => (
                        <button
                            key={idx}
                            onClick={() => onOptionClick(opt.value)}
                            className="text-left px-4 py-2 bg-gray-700 hover:bg-gray-600 text-sm text-white rounded-md border border-gray-600 transition-colors shadow-sm flex items-center justify-between group"
                        >
                            <span>{opt.label}</span>
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                            </svg>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
};


// Komponen utama ChatWindow
export default function ChatWindow({ conversation, onSendMessage, loading, onQuickResponse, quickReplies = [], onOptionClick }) {
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

    return (
        // --- TEMA HITAM --- (Latar belakang konten: abu-abu gelap)
        <div className="w-full flex flex-col h-screen bg-gray-900">
            
            {/* 1. Bar Atas (Header) */}
            {/* --- TEMA HITAM --- (Latar belakang bar: hitam pekat) */}
            <div className="p-4 bg-gradient-to-r from-black via-red-600 to-yellow-400 shadow-md border-b border-gray-700">
                {/* Bagian Kiri: Logo + Judul */}
                <div className="flex items-center space-x-3">
                    <Image
                        src="/logo-satudata.png" // Pastikan nama file ini SAMA
                        alt="Logo Satu Data Garut"
                        width={32} 
                        height={32}
                        className="rounded-full"
                    />
                    
                    {/* --- TEMA KUNING --- (Judul teks putih solid, bukan gradasi) */}
                    <h2 className="text-xl font-semibold text-white">
                        {conversation?.title ? (
                            <span>
                                {conversation.title}
                            </span>
                        ) : (
                            <span>
                                Satu Data Garut
                            </span>
                        )}
                    </h2>
                </div>
            </div>
            
            {/* 2. Area Konten (Chat) */}
            <div className="flex-1 p-6 overflow-y-auto custom-scrollbar">
                
                {/* --- KONDISI A: TAMPILKAN LAYAR SELAMAT DATANG --- */}
                {!conversation && !loading && (
                    <div className="flex flex-col items-center justify-center h-full text-center">
                        <Image
                            src="/logo-satudata.png" // Pastikan nama file ini SAMA
                            alt="Logo Satu Data Garut"
                            width={64} 
                            height={64} 
                            className="mx-auto mb-4"
                            priority 
                        />

                        {/* --- TEMA KUNING --- (Judul teks putih solid) */}
                        <h1 className="text-3xl font-bold text-white">
                            Satu Data Garut
                        </h1>
                        
                        {/* --- TEMA HITAM --- (Teks menjadi abu-abu muda) */}
                        <p className="text-gray-400 mt-2">Tanyakan apapun tentang data di Kabupaten Garut</p>
                    </div>
                )}

                {/* --- KONDISI B: TAMPILKAN PESAN PERCAKAPAN --- */}
                {conversation && (
                    <>
                        {/* Loading saat memuat percakapan lama */}
                        {loading && !conversation.messages && (
                            <div className="flex justify-center items-center h-full">
                                <span className="loading loading-dots loading-lg text-primary"></span>
                                {/* --- TEMA HITAM --- */}
                                <p className="text-gray-400 ml-3">Memuat percakapan...</p>
                            </div>
                        )}
                        
                        {/* Tampilkan pesan-pesan */}
                        {conversation.messages?.map((msg, index) => (
                            <Message 
                            key={msg.id || `msg-${index}`} 
                            sender={msg.sender} 
                            content={msg.content} 
                            options={msg.options} 
                            onOptionClick={onOptionClick}
                            />
                        ))}

                        {/* Loading saat AI sedang mengetik */}
                        {loading && conversation.messages && (
                            <div className="flex justify-start mb-4">
                                {/* --- TEMA KUNING --- (Ikon AI) */}
                                <div className="w-10 h-10 rounded-full bg-yellow-400 flex items-center justify-center mr-3 flex-shrink-0 shadow-md">
                                    <span className="text-black font-bold">AI</span>
                                </div>
                                {/* --- TEMA HITAM --- */}
                                <div className="max-w-xl p-4 rounded-lg bg-gray-800 text-gray-100 border border-gray-700 rounded-bl-none shadow-md">
                                    <div className="flex items-center space-x-2">
                                        <span className="loading loading-dots loading-sm"></span>
                                        {/* --- TEMA HITAM --- */}
                                        <span className="text-sm italic text-gray-300">AI sedang berfikir . . .</span>
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </>
                )}
            </div>

            {/* Tombol Quick Replies (di atas bar bawah) */}
            {!loading && quickReplies && quickReplies.length > 0 && (
                // --- TEMA HITAM --- (Latar belakang sama dengan konten)
                <div className="p-4 pt-0 bg-gray-900"> 
                    <div className="grid grid-cols-2 gap-3 mb-3 max-w-2xl mx-auto">
                        {quickReplies.map((reply, index) => (
                            <button
                                key={index}
                                onClick={() => onQuickResponse(reply.value)}
                                // --- TEMA HITAM ---
                                className="bg-gray-800 border border-gray-700 text-gray-100 text-left p-3 rounded-lg hover:bg-gray-700 transition-colors duration-200 text-sm shadow-sm"
                            >
                                {reply.label}
                            </button>
                        ))}
                    </div>
                </div>
            )}


            {/* 3. Bar Bawah (Input Area) */}
            {/* --- TEMA HITAM --- (Latar belakang bar: hitam pekat) */}
            <div className="p-4 bg-black border-t border-gray-700 shadow-inner">
                {/* Form Input */}
                <div className="max-w-2xl mx-auto">
                    {/* --- TEMA HITAM --- */}
                    <form onSubmit={handleSend} className="flex items-center bg-gray-800 border border-gray-700 rounded-lg shadow-sm p-2">
                         <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ketik pertanyaan anda di sini..."
                            // --- TEMA HITAM ---
                            className="flex-grow bg-transparent border-none focus:ring-0 text-gray-100 placeholder-gray-400 p-2"
                        />
                        {/* --- TEMA KUNING --- (Tombol Send menjadi kuning, ikon hitam) */}
                        <button type="submit" className="btn btn-circle bg-transparent-400 hover:bg-white-500 border-none" disabled={!input.trim() || loading}>
                           <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="white" className="w-5 h-5"><path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086L2.279 16.76a.75.75 0 00.95.826l16-5.333a.75.75 0 000-1.418l-16-5.333z" /></svg>
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}