<?php
namespace App\Console\Commands;

use Illuminate\Console\Command;
use Symfony\Component\Process\Process;

class UpdateChatbotKnowledge extends Command
{
    protected $signature = 'chatbot:update';
    protected $description = 'Perbarui basis pengetahuan chatbot dari API Satu Data';

    public function handle()
    {
        $this->info('Memulai pembaruan basis pengetahuan chatbot...');

        // Tentukan path ke skrip Python Anda
        // Sesuaikan path ini dengan benar
        $pythonPath = base_path('../garut-satu-data-chatbot/update_knowledge_base.py');
        $pythonExecutable = 'python3'; // atau 'python'

        $process = new Process([$pythonExecutable, $pythonPath]);
        $process->setWorkingDirectory(base_path('../garut-satu-data-chatbot'));
        $process->setTimeout(3600); // Set timeout 1 jam

        $process->run(function ($type, $buffer) {
            // Tampilkan output dari skrip Python secara real-time
            echo $buffer; 
        });

        if ($process->isSuccessful()) {
            $this->info('Pembaruan basis pengetahuan chatbot selesai.');
        } else {
            $this->error('Pembaruan basis pengetahuan gagal.');
        }
    }
}