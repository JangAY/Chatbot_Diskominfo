<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\ChatbotController; // Pastikan ini di-import

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
*/

Route::middleware('auth:sanctum')->get('/user', function (Request $request) {
    return $request->user();
});

// Endpoint utama untuk mengirim pesan (Hanya ini yang kita butuhkan)
Route::post('/chat', [ChatbotController::class, 'handleChat']);

// Rute riwayat ('/chat/history' dan '/chat/{id}') telah dihapus.