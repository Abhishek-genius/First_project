import chromadb 
client = chromadb.PersistentClient(path="./chroma_db") 
print("Collections:", [c.name for c in client.list_collections()])