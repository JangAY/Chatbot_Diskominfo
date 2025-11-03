import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- Setup Awal (Hanya berjalan sekali saat server dimulai) ---
try:
    # Download resource NLTK yang diperlukan
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # ← Tambahan penting
    nltk.download('stopwords', quiet=True)
    print("NLTK resources downloaded.")
except Exception as e:
    print(f"Warning: Could not download NLTK data. {e}")

# Inisialisasi daftar stopwords Bahasa Indonesia
list_stopwords = set(stopwords.words('indonesian'))
# Hapus kata-kata yang mungkin penting untuk query (opsional)
list_stopwords.discard('data') 

# Inisialisasi Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- Fungsi Preprocessing ---

def case_folding(text):
    """Mengubah teks menjadi lowercase."""
    return text.lower()

def remove_special_characters(text):
    """Menghapus karakter spesial, angka, dan spasi berlebih."""
    text = re.sub(r'[^a-z\s]', ' ', text) # Hanya simpan huruf dan spasi
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    return text

def normalize_text(text):
    """(Opsional) Normalisasi kata-kata umum."""
    text = re.sub(r'\b(utk)\b', 'untuk', text)
    text = re.sub(r'\b(dgn)\b', 'dengan', text)
    text = re.sub(r'\b(yg)\b', 'yang', text)
    return text

def tokenization(text):
    """Memecah teks menjadi token (kata)."""
    return word_tokenize(text)

def stopwords_removal(tokens):
    """Menghapus stopwords dari daftar token."""
    return [word for word in tokens if word not in list_stopwords]

def stemming(tokens):
    """Mengubah token ke kata dasarnya (stemming)."""
    return [stemmer.stem(word) for word in tokens]

# --- Fungsi Utama ---

def preprocess_text(text: str) -> str:
    """Menjalankan seluruh pipeline preprocessing."""
    text = case_folding(text)
    text = normalize_text(text)
    text = remove_special_characters(text)
    tokens = tokenization(text)
    tokens = stopwords_removal(tokens)
    tokens = stemming(tokens)
    return " ".join(tokens)

if __name__ == '__main__':
    contoh_query = "Cariin data jumlah penduduk miskin di Garut thn 2022 dong"
    hasil_proses = preprocess_text(contoh_query)
    print(f"Query Asli: {contoh_query}")
    print(f"Hasil Proses: {hasil_proses}")
