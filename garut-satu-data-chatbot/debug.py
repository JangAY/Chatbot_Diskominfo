import chromadb
client = chromadb.PersistentClient(path="chatbot_db")
col = client.get_collection("kumpulan_dataset")

peek = col.peek(3)
for m in peek['metadatas'][0:]:
    print(m)