from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    def __init__(
        self, 
        documents: Optional[List[Document]] | None = None, 
        embedding_function: str = "text-embedding-3-small", 
        collection_name: str = "agentic_rag_collection", 
        persist_directory: str = "./tmp/chroma_db/agentic_rag"
    ):
        embeddings = OpenAIEmbeddings(model=embedding_function)
        
        if documents:
            # 使用 from_documents 创建并添加文档
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,  # ✅ 使用 embedding
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        else:
            # 创建空的向量数据库
            self._vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,  # ✅ 使用 embedding_function
                persist_directory=persist_directory
            )

    def get_vectorstore(self):
        return self._vectorstore