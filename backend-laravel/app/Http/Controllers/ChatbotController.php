<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Validator;

class ChatbotController extends Controller
{
    public function handleChat(Request $request)
    {
        // Validasi input dari Next.js
        $validator = Validator::make($request->all(), [
            'query' => 'required|string|max:1000',
        ]);

        if ($validator->fails()) {
            return response()->json(['error' => $validator->errors()->first()], 422);
        }

        $userQuery = $request->input('query');

        // Ambil URL API Python dari file .env
        $pythonApiUrl = env('PYTHON_API_URL');

        if (!$pythonApiUrl) {
            Log::error('PYTHON_API_URL tidak ditemukan di file .env');
            return response()->json([
                'error' => 'Konfigurasi server backend belum lengkap.'
            ], 500);
        }

        try {
            // Kirim request ke API Python (Flask) dengan timeout 60 detik
            $response = Http::timeout(60)->post($pythonApiUrl, [
                'query' => $userQuery
            ]);

            // Jika API Python mengembalikan error
            if (!$response->successful()) {
                Log::error('Gagal memanggil API Python', [
                    'status' => $response->status(),
                    'body' => $response->body()
                ]);
                return response()->json([
                    'error' => 'Layanan AI sedang tidak merespons.',
                    'details' => $response->body()
                ], 502); // 502 Bad Gateway
            }

            // Ambil hasil dari Python API
            $data = $response->json();

            // Kembalikan respons ke frontend Next.js
            return response()->json([
                'reply' => $data['reply'] ?? 'Tidak ada balasan dari layanan AI.'
            ]);

        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            Log::error('Koneksi ke API Python gagal', ['message' => $e->getMessage()]);
            return response()->json([
                'error' => 'Tidak dapat terhubung ke layanan AI. Pastikan server Python berjalan.',
                'details' => $e->getMessage(),
            ], 504); // 504 Gateway Timeout
        } catch (\Exception $e) {
            Log::error('Terjadi kesalahan umum saat menghubungi API Python', ['message' => $e->getMessage()]);
            return response()->json([
                'error' => 'Terjadi kesalahan internal pada server.',
                'details' => $e->getMessage(),
            ], 500);
        }
    }
}
