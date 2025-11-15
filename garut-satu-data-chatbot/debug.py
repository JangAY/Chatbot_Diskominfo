from chromadb import PersistentClient

client = PersistentClient(path="chatbot_db")

print("Available Collections:")
for col in client.list_collections():
    print(col)

# Ambil koleksi embedding dataset
collection = client.get_collection("dataset_embeddings")

# Tampilkan jumlah vector embedding
print("Total embeddings:", collection.count())

# (Opsional) Cek 5 item pertama
items = collection.peek()
print("Sample item IDs:", items["ids"][:5])


print("\n=== Sample Embedding Metadata ===")
print(items["metadatas"][:3])
print(items["documents"][:1])