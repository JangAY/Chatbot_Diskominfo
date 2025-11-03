<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
// PASTIKAN HANYA CONTROLLER INI YANG DI-IMPORT DAN DIGUNAKAN UNTUK /api/chat
use App\Http\Controllers\ChatbotController;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
*/

// Route ini bisa Anda biarkan jika masih terpakai
Route::middleware('auth:sanctum')->get('/user', function (Request $request) {
    return $request->user();
});

// NONAKTIFKAN SEMUA ROUTE LAMA YANG MENGGUNAKAN ChatController.php DARI FOLDER Api
/*
use App\Http\Controllers\Api\ChatController;
Route::post('/chat', [ChatController::class, 'sendMessage']);
Route::get('/history', [ChatController::class, 'getHistory']);
Route::get('/conversation/{id}', [ChatController::class, 'getConversation']);
*/

// AKTIFKAN HANYA ROUTE INI UNTUK CHATBOT
Route::post('/chat', [ChatbotController::class, 'handleChat']);