import chromadb

# 列出所有collections
def list_collections(db_path):
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    print(f"{db_path} 下有Collections: {collections} length: {len(collections)}")

    for index, collection_name in enumerate(collections):
        print(f"{index}. {collection_name} 有 {collection_name.count()}条记录")

# 删除collection
def delete_collection(db_path, collection_name):
    client = chromadb.PersistentClient(path=db_path)
    client.delete_collection(name=collection_name)

db_path = "./chroma_db"
list_collections(db_path)