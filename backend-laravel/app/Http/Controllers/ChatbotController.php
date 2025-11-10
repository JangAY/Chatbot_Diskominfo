<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Validator;
// Hapus 'use App\Models...' karena tidak dipakai lagi

class ChatbotController extends Controller
{
    /**
     * Menangani permintaan chat dan meneruskannya ke API Python.
     * Versi ini stateless (tidak menyimpan riwayat).
     */
    public function handleChat(Request $request)
    {
        // Validasi input dari Next.js
        $validator = Validator::make($request->all(), [
            'query' => 'required|string|max:1000',
            // 'conversation_id' tidak diperlukan lagi
        ]);

        if ($validator->fails()) {
            return response()->json(['error' => $validator->errors()->first()], 422);
        }

        $userQuery = $request->input('query');

        // Ambil URL API Python dari file .env
        $pythonApiUrl = env('PYTHON_API_URL');
        if (!$pythonApiUrl) {
            Log::error('PYTHON_API_URL tidak ditemukan di file .env');
            return response()->json(['error' => 'Konfigurasi server backend belum lengkap.'], 500);
        }

        try {
            // Kirim request ke API Python (Flask)
            // Tambahkan timeout lebih lama untuk proses agent yang kompleks
            $response = Http::timeout(120)->post($pythonApiUrl, [
                'query' => $userQuery
            ]);

            if (!$response->successful()) {
                Log::error('Gagal memanggil API Python', ['status' => $response->status(), 'body' => $response->body()]);
                return response()->json(['error' => 'Layanan AI sedang tidak merespons.'], 502);
            }

            // Ambil data JSON LENGKAP dari Python
            $data = $response->json();

            // Kembalikan respons LENGKAP dari Python ke frontend
            // Ini akan mencakup 'reply' dan 'newQuickReplies'
            return response()->json($data);

        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            Log::error('Koneksi ke API Python gagal', ['message' => $e->getMessage()]);
            return response()->json(['error' => 'Tidak dapat terhubung ke layanan AI.'], 504);
        } catch (\Exception $e) {
            Log::error('Kesalahan handleChat', ['message' => $e->getMessage()]);
            return response()->json(['error' => 'Terjadi kesalahan internal pada server.'], 500);
        }
    }

    // Fungsi getHistory() dan getConversationMessages() telah dihapus.
}