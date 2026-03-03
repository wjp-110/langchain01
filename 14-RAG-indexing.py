# 1. 读取网页 bs4
import shutil

from langchain_community.document_loaders import WebBaseLoader
import bs4
import os

if os.path.exists("./chroma_rag_db"):
    shutil.rmtree("./chroma_rag_db")

page_url = "https://news.cctv.com/2025/08/07/ARTIwHXTjBUTWQHIhY3pmv7Z250807.shtml"
bs4_strainer = bs4.SoupStrainer()
loader = WebBaseLoader(
    web_paths=(page_url,),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()    # List[Document]
print(len(docs))

# 2. 分割 pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
docs = text_splitter.split_documents(docs)  # List[Document]
print(len(docs))
print(docs[0].page_content)

# 3. embedding 创建索引
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="nomic-embed-text")

# 4. 存向量库

from langchain_chroma import Chroma
vectory_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embedding,
    persist_directory="./chroma_rag_db"
)
ids = vectory_store.add_documents(docs)
print(len(ids))