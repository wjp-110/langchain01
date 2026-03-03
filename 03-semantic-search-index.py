# 1. 加载pdf pypdf
from langchain_community.document_loaders import PyPDFLoader

file_path = "./test.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()    # List[Document]

# 2. 分割 pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
docs = text_splitter.split_documents(docs)  # List[Document]

# 3. embedding 创建索引
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="nomic-embed-text")

# 4. 存向量库

from langchain_chroma import Chroma
vectory_store = Chroma(
    collection_name="test",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)
ids = vectory_store.add_documents(docs)
print(len(ids))