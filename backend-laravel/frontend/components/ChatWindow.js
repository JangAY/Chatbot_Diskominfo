// frontend/components/ChatWindow.js

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import Image from 'next/image';

const Message = ({ sender, content, options, onOptionClick }) => {
    const isUser = sender === 'user';
    
    // Deteksi Grid Options
    const isGridOptions = options && options.length > 4;

    return (
        <div className={`flex flex-col mb-6 ${isUser ? 'items-end' : 'items-start'}`}>
            <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} w-full`}>
                {!isUser && (
                    <div className="w-8 h-8 rounded-full bg-yellow-400 flex items-center justify-center mr-3 flex-shrink-0 shadow-md">
                        <span className="text-black text-xs font-bold">AI</span>
                    </div>
                )}
                <div
                    className={`max-w-[90%] sm:max-w-2xl p-4 rounded-xl shadow-md ${
                        isUser 
                            // KEMBALI KE TEMA ASLI: KUNING TEXT HITAM
                            ? 'bg-yellow-400 text-black rounded-br-none' 
                            : 'bg-gray-800 text-gray-100 border border-gray-700 rounded-bl-none'
                    }`}
                >
                    {isUser ? (
                        <div className="text-black whitespace-pre-wrap">{content}</div>
                    ) : (
                        <div className="prose prose-sm prose-invert max-w-none">
                            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                                {content}
                            </ReactMarkdown>
                        </div>
                    )}
                </div>
            </div>

            {/* AREA OPSI / PILIHAN DATA */}
            {!isUser && options && options.length > 0 && (
                <div className={`ml-11 mt-3 w-full max-w-3xl animate-fadeIn`}>
                    
                    {isGridOptions ? (
                        // --- TAMPILAN GRID CARD (UNTUK LIST SEKTOR) ---
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                            {options.map((opt, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => onOptionClick(opt.value)}
                                    className="flex items-center p-3 bg-gray-800 border border-gray-700 hover:border-yellow-500 hover:bg-gray-750 rounded-lg transition-all group shadow-sm text-left h-full"
                                >
                                    <div className="p-2 bg-gray-900 rounded-md text-yellow-500 group-hover:text-yellow-400 mr-3 flex-shrink-0">
                                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 21h19.5m-18-18v18m10.5-18v18m6-13.5V21M6.75 6.75h.75m-.75 3h.75m-.75 3h.75m3-6h.75m-.75 3h.75m-.75 3h.75M6.75 21v-3.375c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21M3 3h12m-.75 4.5H21m-3.75 3.75h.008v.008h-.008v-.008zm0 3h.008v.008h-.008v-.008zm0 3h.008v.008h-.008v-.008z" />
                                        </svg>
                                    </div>
                                    <span className="text-xs sm:text-sm font-medium text-gray-200 group-hover:text-white line-clamp-2">
                                        {opt.label}
                                    </span>
                                </button>
                            ))}
                        </div>
                    ) : (
                        // --- TAMPILAN STANDARD CHIPS ---
                        <div className="flex flex-wrap gap-2">
                            {options.map((opt, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => onOptionClick(opt.value)}
                                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 border border-gray-600 text-sm text-white rounded-full transition-colors flex items-center space-x-1"
                                >
                                    <span>{opt.label}</span>
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-3 h-3 opacity-70">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                                    </svg>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default function ChatWindow({ 
    conversation, 
    onSendMessage, 
    loading, 
    onQuickResponse, 
    quickReplies = [], 
    onOptionClick,
    toggleSidebar, 
    isSidebarOpen
}) {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);
    const textareaRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [conversation?.messages, loading]);

    const handleInput = (e) => {
        setInput(e.target.value);
        e.target.style.height = 'auto';
        e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
    };

    const handleSend = (e) => {
        e.preventDefault();
        if (input.trim() && !loading) {
            onSendMessage(input);
            setInput('');
            if (textareaRef.current) textareaRef.current.style.height = 'auto';
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend(e);
        }
    };

    // --- RENDER WELCOME SCREEN (KARTU) ---
    const renderWelcomeScreen = () => (
        <div className="flex flex-col items-center justify-center h-full px-4 text-center">
            {/* Logo Besar */}
            <div className="mb-6 relative w-20 h-20 bg-gray-800 rounded-2xl flex items-center justify-center shadow-lg border border-gray-700">
                <Image src="/logo-satudata.png" alt="Logo" width={50} height={50} />
            </div>

            <h1 className="text-2xl sm:text-3xl font-bold text-white mb-2">
                Halo, ada yang bisa dibantu?
            </h1>
            <p className="text-gray-400 mb-8 max-w-md text-sm">
                Portal Satu Data Garut siap membantu Anda menemukan data statistik dan informasi sektoral.
            </p>

            {/* GRID KARTU WELCOME */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-3xl">
                {quickReplies.map((reply, index) => (
                    <button
                        key={index}
                        onClick={() => onQuickResponse(reply.value)}
                        className="flex items-center space-x-4 p-4 bg-gray-800 border border-gray-700 hover:border-yellow-500 hover:bg-gray-750 rounded-xl transition-all group shadow-sm text-left"
                    >
                        <div className="p-3 bg-gray-900 rounded-lg text-yellow-500 group-hover:text-yellow-400 group-hover:bg-gray-800 transition-colors">
                            {/* Ikon Statis Sederhana untuk Welcome Screen */}
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
                            </svg>
                        </div>
                        <div>
                            <span className="font-semibold text-gray-200 block text-sm group-hover:text-white transition-colors">{reply.label}</span>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );

    return (
        <div className="flex flex-col h-full bg-gray-900 w-full relative">
            
            {/* 1. HEADER STATIS (KEMBALI KE GRADIENT ASLI) */}
            <div className="h-20 px-4 bg-gradient-to-r from-black via-red-600 to-yellow-400 shadow-md border-b border-gray-700 flex items-center justify-between z-10 sticky top-0">
                <div className="flex items-center">
                    
                    {/* TOMBOL TOGGLE */}
                    <button 
                        onClick={toggleSidebar} 
                        className="mr-3 p-2 rounded-lg hover:bg-black/20 text-white transition-colors focus:outline-none"
                    >
                         <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-6 h-6">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
                        </svg>
                    </button>

                    {/* JUDUL APLIKASI + LOGO */}
                    <div className="flex items-center space-x-3">
                        <Image src="/logo-satudata.png" alt="Logo" width={30} height={30} className="rounded-full bg-white/20" />
                        <h2 className="text-lg font-bold text-white tracking-wide drop-shadow-md">
                            Garut Satu Data
                        </h2>
                    </div>
                </div>
            </div>
            
            {/* 2. AREA CHAT */}
            <div className="flex-1 p-4 sm:p-6 overflow-y-auto custom-scrollbar scroll-smooth">
                {!conversation?.messages?.length ? (
                    // Tampilkan Kartu jika belum ada pesan
                    renderWelcomeScreen()
                ) : (
                    <>
                        {conversation.messages.map((msg, index) => (
                            <Message 
                                key={msg.id || index} 
                                sender={msg.sender} 
                                content={msg.content} 
                                options={msg.options} 
                                onOptionClick={onOptionClick}
                            />
                        ))}
                        
                        {/* Loading */}
                        {loading && (
                            <div className="flex justify-start mb-6 animate-pulse">
                                <div className="w-8 h-8 rounded-full bg-yellow-400 flex items-center justify-center mr-3">
                                    <span className="text-black text-xs font-bold">AI</span>
                                </div>
                                <div className="p-4 rounded-xl bg-gray-800 border border-gray-700 rounded-bl-none">
                                    <div className="flex space-x-2">
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </>
                )}
            </div>

            {/* Quick Replies (Tombol Bawah - HANYA MUNCUL JIKA JUMLAHNYA SEDIKIT) */}
            {!loading && conversation?.messages?.length > 0 && quickReplies.length > 0 && (
                 <div className="px-4 pb-2 bg-gray-900"> 
                    <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar">
                        {quickReplies.map((reply, index) => (
                            <button
                                key={index}
                                onClick={() => onQuickResponse(reply.value)}
                                className="whitespace-nowrap bg-gray-800 border border-gray-700 text-gray-300 px-4 py-2 rounded-full hover:bg-gray-700 hover:text-white transition-all text-sm flex-shrink-0"
                            >
                                {reply.label}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* 3. INPUT AREA (KEMBALI KE BACKGROUND HITAM) */}
            <div className="p-4 bg-black border-t border-gray-800">
                <div className="max-w-3xl mx-auto relative">
                    <form onSubmit={handleSend} className="bg-gray-900 border border-gray-700 rounded-full flex items-center p-2 shadow-lg focus-within:border-yellow-500 transition-all">
                        <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={handleInput}
                            onKeyDown={handleKeyDown}
                            placeholder="Ketik pertanyaan Anda..."
                            className="w-full bg-transparent border-none text-gray-100 placeholder-gray-500 focus:ring-0 resize-none max-h-[120px] py-3 px-4 scrollbar-hide leading-relaxed"
                            rows={1}
                        />
                        {/* Tombol Kirim Kembali Kuning */}
                        <button 
                            type="submit" 
                            disabled={!input.trim() || loading}
                            className="p-2 mr-1 rounded-full bg-yellow-500 hover:bg-yellow-400 text-black disabled:bg-gray-700 disabled:text-gray-500 transition-transform active:scale-95 flex-shrink-0"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                                <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                            </svg>
                        </button>
                    </form>
                    <p className="text-center text-[10px] text-gray-600 mt-2">
                        Garut Satu Data AI
                    </p>
                </div>
            </div>
        </div>
    );
}