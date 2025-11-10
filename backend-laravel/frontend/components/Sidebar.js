// components/Sidebar.js

// --- PERUBAHAN: Hapus semua import 'date-fns' ---

// --- PERUBAHAN: Hapus prop 'history', 'loading', dll. ---
export default function Sidebar({ onNewChat }) {
  
  // --- HAPUS: Semua fungsi 'getGroupedHistory' dan 'renderHistoryGroup' ---

  return (
    <div className="w-1/4 bg-gray-900 text-white flex flex-col h-full shadow-lg">
      {/* Tombol New Chat */}
      <div className="p-4 border-b border-gray-700">
        <button
          onClick={onNewChat}
          className="btn btn-primary w-full flex items-center justify-center space-x-2 rounded-lg py-3 text-sm font-medium"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
          </svg>
          <span>New Chat</span>
        </button>
      </div>

      {/* --- PERUBAHAN: Hapus Daftar Riwayat --- */}
      <div className="flex-1 p-4 overflow-y-auto custom-scrollbar">
         <p className="text-gray-400 text-sm text-center">
            Riwayat chat tidak disimpan dalam sesi ini.
         </p>
      </div>

      {/* Tombol Download Text */}
      <div className="p-4 border-t border-gray-700">
        <button className="btn btn-outline btn-primary w-full rounded-lg py-3 text-sm font-medium">
          Download Text
        </button>
      </div>
    </div>
  );
}