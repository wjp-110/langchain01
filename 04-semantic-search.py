from langchain_ollama import OllamaEmbeddings
embedding = OllamaEmbeddings(model="nomic-embed-text")


from langchain_chroma import Chroma
vectory_store = Chroma(
    collection_name="test",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

# 通过文本搜索
results = vectory_store.similarity_search("优秀员工绩效工资占部门绩效工资总额多少", k=1)

for index, result in enumerate(results):
    print(f"{index}: {result.page_content}")

# 通过分数搜索
results = vectory_store.similarity_search_with_score("优秀员工绩效工资占部门绩效工资总额多少", k=4)
for index, (result, score) in enumerate(results):
    print(f"{index}")
    print(f"{result.page_content}")
    print(f"{score}")

# 通过向量搜索
vectory = vectory_store.similarity_search_by_vector(
    embedding.embed_query("优秀员工绩效工资占部门绩效工资总额多少")
)
for index, result in enumerate(vectory):
    print(f"{index}: {result.page_content}")
    

# chain
from langchain_core.runnables import chain
from typing import List
from langchain_core.documents import Document


@chain
def retrivers(question: str) -> List[Document]:
    results = vectory_store.similarity_search(question, k=1)
    return results
print("----------------------------------------------------------------------------------------------------------")
print(retrivers.invoke("优秀员工绩效工资占部门绩效工资总额多少"))
