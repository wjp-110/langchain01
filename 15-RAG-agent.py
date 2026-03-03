from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool

load_dotenv()

# 定义嵌入模型
embedding = OllamaEmbeddings(model="nomic-embed-text")

# 定义向量库
vectory_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embedding,
    persist_directory="./chroma_rag_db"
)

@tool(response_format="content_and_artifact")
def get_information(query: str):
    """get the information given query"""
    docs = vectory_store.similarity_search(query)
    content = "\n".join([doc.page_content for doc in docs])
    return content, docs[0].page_content


agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[get_information]
)

results = agent.invoke({"messages": [{"role": "user", "content": "讲一下 3i/Atlas"}] })

messages = results["messages"]
for message in messages:
    message.pretty_print()