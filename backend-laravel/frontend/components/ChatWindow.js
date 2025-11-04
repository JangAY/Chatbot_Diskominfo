// components/ChatWindow.js

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown'; // <-- Import
import remarkGfm from 'remark-gfm'; // <-- Import

// Komponen untuk pesan individual (SUDAH DIPERBAIKI)
const Message = ({ sender, content }) => {
    const isUser = sender === 'user';
    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
            {!isUser && (
                <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center mr-3 flex-shrink-0">
                    <span className="text-white font-bold">AI</span>
                </div>
            )}
            <div
                className={`max-w-xl p-4 rounded-lg ${
                    isUser ? 'bg-blue-500 text-white rounded-br-none' : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none'
                }`}
            >
                {isUser ? (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{content}</div>
                ) : (
                    <div className="prose prose-sm max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {content}
                        </ReactMarkdown>
                    </div>
                )}
            </div>
        </div>
    );
};


// Komponen utama ChatWindow
export default function ChatWindow({ conversation, onSendMessage, loading, onQuickResponse }) {
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
        if (input.trim()) {
            onSendMessage(input);
            setInput('');
        }
    };
    
    // Contoh tombol respons cepat
    const quickResponses = [
        "Apa saja dataset yang tersedia?",
        "Tampilkan jumlah penduduk Garut.",
        "Bagaimana cara menggunakan portal ini?",
        "Jelaskan metadata yang ada.",
    ];

    // Tampilan Selamat Datang (jika tidak ada percakapan)
    if (!conversation && !loading) {
        return (
            <div className="w-3/4 flex flex-col items-center justify-center h-screen bg-gray-50 p-4">
                <div className="text-center">
                    {/* Ganti dengan logo Anda */}
                    <div className="w-16 h-16 rounded-full bg-blue-500 flex items-center justify-center mx-auto mb-4">
                        <span className="text-white font-bold text-2xl">SDG</span>
                    </div>
                    <h1 className="text-3xl font-bold text-gray-800">Satu Data Garut</h1>
                    <p className="text-gray-500 mt-2">Tanyakan apapun tentang data di Kabupaten Garut</p>
                </div>
                {/* Bagian Input Tetap Ditampilkan */}
                <div className="w-full max-w-2xl mt-auto p-4">
                     <div className="grid grid-cols-2 gap-3 mb-3">
                        {quickResponses.map((text, index) => (
                            <button
                                key={index}
                                onClick={() => onQuickResponse(text)}
                                className="bg-white border border-gray-300 text-gray-700 text-left p-3 rounded-lg hover:bg-gray-100 transition-colors duration-200 text-sm"
                            >
                                {text}
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
                        <button type="submit" className="btn btn-primary btn-circle">
                           <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086L2.279 16.76a.75.75 0 00.95.826l16-5.333a.75.75 0 000-1.418l-16-5.333z" /></svg>
                        </button>
                    </form>
                </div>
            </div>
        );
    }

    return (
        <div className="w-3/4 flex flex-col h-screen bg-gray-50">
            {/* Header Chat */}
            <div className="p-4 border-b border-gray-200 bg-white shadow-sm">
                <h2 className="text-xl font-semibold text-gray-800">Satu Data Garut</h2>
            </div>
            
            {/* Area Pesan */}
            <div className="flex-1 p-6 overflow-y-auto custom-scrollbar">
                {loading ? (
                    <div className="flex justify-center items-center h-full">
                        <span className="loading loading-dots loading-lg text-primary"></span>
                    </div>
                ) : (
                    conversation?.messages?.map((msg, index) => (
                        <Message key={msg.id || `msg-${index}`} sender={msg.sender} content={msg.content} />
                    ))
                )}
                 <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-gray-50">
                 {/* Tombol Respons Cepat */}
                 <div className="grid grid-cols-2 gap-3 mb-3 max-w-2xl mx-auto">
                    {quickResponses.map((text, index) => (
                        <button
                            key={index}
                            onClick={() => onQuickResponse(text)}
                            className="bg-white border border-gray-300 text-gray-700 text-left p-3 rounded-lg hover:bg-gray-100 transition-colors duration-200 text-sm"
                        >
                            {text}
                        </button>
                    ))}
                </div>
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
                        <button type="submit" className="btn btn-primary btn-circle" disabled={!input.trim()}>
                           <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086L2.279 16.76a.75.75 0 00.95.826l16-5.333a.75.75 0 000-1.418l-16-5.333z" /></svg>
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}