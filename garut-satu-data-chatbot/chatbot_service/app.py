from flask import Flask, jsonify, request
import threading
import time
import traceback
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garut_knowledge_base.main import main as update_knowledge

app = Flask(__name__)

# ==============================
# 🔁 FUNGSI AUTO UPDATE
# ==============================
def auto_update_daemon(interval_hours=6):
    """Jalankan pembaruan otomatis setiap interval jam."""
    while True:
        print("\n🧠 [AUTO UPDATE] Menjalankan pembaruan otomatis knowledge base...")
        try:
            update_knowledge()
            print("✅ [AUTO UPDATE] Pembaruan knowledge base selesai.")
        except Exception as e:
            print("❌ [AUTO UPDATE] Gagal memperbarui knowledge base:")
            traceback.print_exc()
        print(f"⏳ Menunggu {interval_hours} jam sebelum pembaruan berikutnya...\n")
        time.sleep(interval_hours * 3600)  # jeda X jam


# Jalankan background thread auto-update
threading.Thread(target=auto_update_daemon, args=(6,), daemon=True).start()


# ==============================
# 🌐 ENDPOINT API CHATBOT
# ==============================

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Chatbot Garut Knowledge Service aktif 🚀"
    })


@app.route("/update-now", methods=["POST"])
def manual_update():
    """Jalankan update manual (misal dipanggil dari Laravel)."""
    def run_update():
        try:
            update_knowledge()
            print("✅ [MANUAL UPDATE] Pembaruan selesai.")
        except Exception as e:
            print(f"❌ [MANUAL UPDATE] Error: {e}")

    threading.Thread(target=run_update).start()
    return jsonify({
        "status": "processing",
        "message": "Pembaruan manual sedang dijalankan di background."
    })


@app.route("/status", methods=["GET"])
def status():
    """Cek status service."""
    return jsonify({
        "service": "Chatbot Knowledge Updater",
        "auto_update": "aktif",
        "interval_hours": 6
    })


if __name__ == "__main__":
    print("🚀 Menjalankan Chatbot Knowledge Service dengan Auto-Update Daemon...")
    app.run(host="0.0.0.0", port=5000)
