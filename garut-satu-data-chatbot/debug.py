from chromadb import PersistentClient

client = PersistentClient("./chatbot_db")

print("Available Collections:")
print(client.list_collections())