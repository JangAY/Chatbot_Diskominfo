// frontend/components/Sidebar.js
import { useEffect, useState } from 'react';

export default function Sidebar({ 
  isOpen, 
  toggleSidebar, 
  onNewChat, 
  chatHistory, 
  activeChatId, 
  onSelectChat, 
  onDeleteChat 
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  if (!mounted) return null;

  return (
    <>
      {/* Overlay Mobile */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-20 md:hidden"
          onClick={toggleSidebar}
        ></div>
      )}

      {/* Sidebar Container */}
      <div className={`
        fixed inset-y-0 left-0 z-30 w-72 bg-gray-900 border-r border-gray-800 
        transform transition-transform duration-300 ease-in-out flex flex-col shadow-2xl
        ${isOpen ? 'translate-x-0' : '-translate-x-full'} 
        md:relative md:translate-x-0 
        ${!isOpen && 'md:!hidden'} 
      `}>
        
        {/* --- HEADER SIDEBAR (TOMBOL MENYATU) --- */}
        <div className="h-20 flex items-center px-4 border-b border-gray-800 bg-gray-900">
             <button
                onClick={() => {
                  onNewChat();
                  if (window.innerWidth < 768) toggleSidebar();
                }}
                // KEMBALI KE TEMA KUNING (Original)
                className="flex items-center justify-center space-x-2 w-full py-3 bg-yellow-500 hover:bg-yellow-400 text-black rounded-lg transition-all shadow-md font-bold"
             >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                </svg>
                <span>Obrolan Baru</span>
             </button>
        </div>

        {/* --- LIST RIWAYAT --- */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-3">
          <div className="px-2 mb-3 mt-2 text-[11px] font-bold text-gray-500 uppercase tracking-widest">
            Riwayat
          </div>
          
          {chatHistory.length === 0 ? (
            <div className="px-4 mt-10 text-center">
                <p className="text-xs text-gray-600 italic">Belum ada riwayat.</p>
            </div>
          ) : (
            <div className="space-y-1">
              {chatHistory.map((chat) => (
                <div 
                  key={chat.id}
                  className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all ${
                    activeChatId === chat.id 
                      ? 'bg-gray-800 text-yellow-400 border border-gray-700 shadow-sm' // Aktif: Teks Kuning
                      : 'text-gray-400 hover:bg-gray-800/50 hover:text-gray-200 border border-transparent'
                  }`}
                  onClick={() => {
                    onSelectChat(chat.id);
                    if (window.innerWidth < 768) toggleSidebar();
                  }}
                >
                  <div className="flex items-center space-x-3 overflow-hidden">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 min-w-[16px]">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
                    </svg>
                    <span className="text-sm truncate font-medium">
                      {chat.title || 'Percakapan Baru'}
                    </span>
                  </div>

                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteChat(chat.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 p-1 transition-opacity"
                    title="Hapus"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                      <path fillRule="evenodd" d="M8.75 1A2.75 2.75 0 006 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 10.23 1.482l.149-.022.841 10.518A2.75 2.75 0 007.596 19h4.807a2.75 2.75 0 002.742-2.53l.841-10.52.149.023a.75.75 0 00.23-1.482A41.03 41.03 0 0014 4.193V3.75A2.75 2.75 0 0011.25 1h-2.5zM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4zM8.58 7.72a.75.75 0 00-1.5.06l.3 7.5a.75.75 0 101.5-.06l-.3-7.5zm4.34.06a.75.75 0 10-1.5-.06l-.3 7.5a.75.75 0 101.5.06l.3-7.5z" clipRule="evenodd" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer Sidebar */}
        <div className="p-4 border-t border-gray-800 bg-gray-900">
           <div className="text-[10px] text-gray-600 text-center">
             Satu Data Garut AI
           </div>
        </div>
      </div>
    </>
  );
}