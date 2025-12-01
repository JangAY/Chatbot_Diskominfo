import argparse
import sys
import os

# Tambahkan root directory agar semua modul bisa ditemukan
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Import modul
from chatbot import main as chatbot_cli
from chatbot_service.app import run_service
from garut_knowledge_base.main import main as update_knowledge


def main():
    parser = argparse.ArgumentParser(description="Unified Runner for Chatbot System")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["cli", "api", "update"],
        help="Pilih mode: cli | api | update"
    )

    args = parser.parse_args()

    if args.mode == "cli":
        print("=== Menjalankan Chatbot CLI (chatbot.py) ===")
        chatbot_cli()

    elif args.mode == "api":
        print("=== Menjalankan Chatbot API Service (chatbot_service/app.py) ===")
        run_service()

    elif args.mode == "update":
        print("=== Menjalankan Update Knowledge Base ===")
        update_knowledge()

    else:
        print("Mode tidak dikenali.")


if __name__ == "__main__":
    main()
