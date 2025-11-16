// components/ChatWindow.js

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import Image from 'next/image';

// Komponen untuk pesan individual (tidak berubah)
const Message = ({ sender, content }) => {
    const isUser = sender === 'user';
    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
            {!isUser && (
<<<<<<< Updated upstream
                <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center mr-3 flex-shrink-0 shadow-md">
                    <span className="text-white font-bold">AI</span>
=======
                // --- TEMA KUNING --- (Ikon AI)
                <div className="w-10 h-10 rounded-full bg-red-400 flex items-center justify-center mr-3 flex-shrink-0 shadow-md">
                    <span className="text-black font-bold">AI</span>
>>>>>>> Stashed changes
                </div>
            )}
            <div
                className={`max-w-xl p-4 rounded-lg shadow-md ${
<<<<<<< Updated upstream
                    isUser ? 'bg-blue-500 text-white rounded-br-none' : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none'
=======
                    // --- TEMA KUNING ---
                    // Gelembung pengguna menjadi kuning, teks hitam
                    isUser ? 'bg-red-500 text-black rounded-br-none' 
                           // Gelembung AI tetap abu-abu gelap
                           : 'bg-gray-800 text-gray-100 border border-gray-700 rounded-bl-none'
>>>>>>> Stashed changes
                }`}
            >
                {isUser ? (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{content}</div>
                ) : (
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

    return (
        // Ini adalah layout utama (Fullscreen)
        <div className="w-full flex flex-col h-screen bg-gray-50">
            
            {/* 1. Bar Atas (Header) */}
            <div className="p-4 border-b border-gray-200 bg-white shadow-sm">
                <div className="flex items-center space-x-3">
                    <Image
                        src="/logo-satudata.png" // Pastikan nama file ini SAMA
                        alt="Logo Satu Data Garut"
                        width={32} // 32px
                        height={32} // 32px
                        className="rounded-full"
                    />
                    <h2 className="text-xl font-semibold text-gray-800">
                        {conversation?.title || "Satu Data Garut"}
                    </h2>
                </div>
            </div>
            
            {/* 2. Area Konten (Chat) - Latar belakang abu-abu */}
            <div className="flex-1 p-6 overflow-y-auto custom-scrollbar">
                
                {/* --- KONDISI A: TAMPILKAN LAYAR SELAMAT DATANG --- */}
                {/* (Ini sekarang ada di DALAM layout utama) */}
                {!conversation && !loading && (
                    <div className="flex flex-col items-center justify-center h-full text-center">
                        <Image
                            src="/logo-satudata.png" // Pastikan nama file ini SAMA
                            alt="Logo Satu Data Garut"
                            width={64} // 64px
                            height={64} // 64px
                            className="mx-auto mb-4"
                            priority 
                        />
                        <h1 className="text-3xl font-bold text-gray-800">Satu Data Garut</h1>
                        <p className="text-gray-500 mt-2">Tanyakan apapun tentang data di Kabupaten Garut</p>
                    </div>
                )}

                {/* --- KONDISI B: TAMPILKAN PESAN PERCAKAPAN --- */}
                {conversation && (
                    <>
                        {/* Loading saat memuat percakapan lama */}
                        {loading && !conversation.messages && (
                            <div className="flex justify-center items-center h-full">
                                <span className="loading loading-dots loading-lg text-primary"></span>
                                <p className="text-gray-500 ml-3">Memuat percakapan...</p>
                            </div>
                        )}
                        
                        {/* Tampilkan pesan-pesan */}
                        {conversation.messages?.map((msg, index) => (
                            <Message key={msg.id || `msg-${index}`} sender={msg.sender} content={msg.content} />
                        ))}

                        {/* Loading saat AI sedang mengetik */}
                        {loading && conversation.messages && (
                            <div className="flex justify-start mb-4">
                                <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center mr-3 flex-shrink-0 shadow-md">
                                    <span className="text-white font-bold">AI</span>
                                </div>
                                <div className="max-w-xl p-4 rounded-lg bg-white text-gray-800 border border-gray-200 rounded-bl-none shadow-md">
                                    <div className="flex items-center space-x-2">
                                        <span className="loading loading-dots loading-sm"></span>
                                        <span className="text-sm italic text-gray-600">AI sedang berfikir . . .</span>
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </>
                )}
            </div>

            {/* 3. Bar Bawah (Input Area) */}
            {/* Ini adalah "bar bawah" yang Anda inginkan (putih dan terpisah) */}
            <div className="p-4 bg-white border-t border-gray-200 shadow-inner">
                 {!loading && quickReplies && quickReplies.length > 0 && (
                    <div className="grid grid-cols-2 gap-3 mb-3 max-w-2xl mx-auto">
                        {quickReplies.map((reply, index) => (
                            <button
                                key={index}
                                onClick={() => onQuickResponse(reply.value)}
                                className="bg-white border border-gray-300 text-gray-700 text-left p-3 rounded-lg hover:bg-gray-100 transition-colors duration-200 text-sm shadow-sm"
                            >
                                {reply.label}
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
<<<<<<< Updated upstream
                        <button type="submit" className="btn btn-primary btn-circle" disabled={!input.trim() || loading}>
                           <svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086L2.279 16.76a.75.75 0 00.95.826l16-5.333a.75.75 0 000-1.418l-16-5.333z" /></svg>
=======
                        {/* --- TEMA KUNING --- (Tombol Send menjadi kuning, ikon hitam) */}
                        <button type="submit" className="btn btn-circle bg-transparent hover:bg-white-500 border-none" disabled={!input.trim() || loading}>
                           <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="white" className="w-5 h-5"><path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086L2.279 16.76a.75.75 0 00.95.826l16-5.333a.75.75 0 000-1.418l-16-5.333z" /></svg>
>>>>>>> Stashed changes
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}