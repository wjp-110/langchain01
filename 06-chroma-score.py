
# 1. 定义embedding model
from langchain_ollama import OllamaEmbeddings
embedding = OllamaEmbeddings(model="nomic-embed-text")

# 评分方式
score_measure = [
    "default",
    "cosine",
    "l2",
    "ip"
]

# 2. 建向量库和四个collection
from langchain_chroma import Chroma

persist_dir = "./chroma_score_db"
vector_stores = []
for score in score_measure:
    collection_metadata = {
        "hnsw:space": score
    }
    if score == "default":
        collection_metadata = None
    collection_name = f"my_collection_{score}"
    vector_stores.append(
        Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_dir
        )
    )

def indexing(docs):
    print("\n 加入文档")
    for vector_store in vector_stores:
        ids = vector_store.add_documents(docs)
        print(f"\n 集合名称: {vector_store._collection.name}")
        print(ids)



from langchain_core.documents import Document
docs = [
    Document(page_content="这个小米手机很好用"),
    Document(page_content="中国陕西地区盛产小米")
]

# indexing(docs)


def query(question: str):
    print("\n 检索文档")
    for vector_store in vector_stores:
        # print(f"\n 集合名称: {vector_store._collection.name}")
        print(f'查询: {question}')
        docs = vector_store.similarity_search_with_score(question)
        # print(docs)
        for doc, score in docs:
            print(f"文档：{doc.page_content}, 分数: {score}")
query("我和朋友通完话")

