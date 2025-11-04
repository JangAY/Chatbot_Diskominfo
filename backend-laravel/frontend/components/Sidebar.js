// components/Sidebar.js
import { format } from 'date-fns/format';
import { id } from 'date-fns/locale/id'; // Untuk format tanggal dalam Bahasa Indonesia

export default function Sidebar({ history, onSelectConversation, onFilterChange, onNewChat, loading }) {
  // Fungsi helper untuk mengelompokkan riwayat berdasarkan waktu
  const getGroupedHistory = (historyData) => {
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);

    const groups = {
      today: [],
      yesterday: [],
      last_week: [],
      last_month: [],
      earlier: [],
    };

    historyData.forEach(conv => {
      const convDate = new Date(conv.updated_at);
      if (convDate.toDateString() === today.toDateString()) {
        groups.today.push(conv);
      } else if (convDate.toDateString() === yesterday.toDateString()) {
        groups.yesterday.push(conv);
      } else if (convDate > new Date(today.setDate(today.getDate() - 7))) {
        groups.last_week.push(conv);
      } else if (convDate > new Date(today.setMonth(today.getMonth() - 1))) {
        groups.last_month.push(conv);
      } else {
        groups.earlier.push(conv);
      }
    });

    return groups;
  };

  const groupedHistory = getGroupedHistory(history);

  const renderHistoryGroup = (title, conversations) => {
    if (conversations.length === 0) return null;
    return (
      <div className="mb-4">
        <h3 className="text-xs uppercase text-gray-400 font-semibold mb-2 ml-2">{title}</h3>
        <ul>
          {conversations.map((conv) => (
            <li
              key={conv.id}
              onClick={() => onSelectConversation(conv.id)}
              className="cursor-pointer p-2 rounded-lg hover:bg-gray-700 transition-colors duration-200 truncate"
              title={conv.title}
            >
              {conv.title || `Percakapan #${conv.id}`}
            </li>
          ))}
        </ul>
      </div>
    );
  };

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

      {/* Daftar Riwayat Percakapan */}
      <div className="flex-1 p-4 overflow-y-auto custom-scrollbar">
        {loading ? (
          <span className="loading loading-spinner loading-md text-primary"></span>
        ) : (
          <>
            {renderHistoryGroup('Hari Ini', groupedHistory.today)}
            {renderHistoryGroup('Kemarin', groupedHistory.yesterday)}
            {renderHistoryGroup('Minggu Lalu', groupedHistory.last_week)}
            {renderHistoryGroup('Bulan Lalu', groupedHistory.last_month)}
            {renderHistoryGroup('Sebelumnya', groupedHistory.earlier)}
            {history.length === 0 && <p className="text-gray-400 text-sm text-center">Belum ada percakapan.</p>}
          </>
        )}
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