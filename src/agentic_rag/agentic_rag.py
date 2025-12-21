"""Agentic RAG 完整系统 - 修复持久化问题版"""
from typing import List, Optional
from agentic_rag.llm import LLM
from agentic_rag.splitter import DocsSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.agentic_rag.graph import create_agentic_rag_graph, create_initial_state
from src.agentic_rag.vector_store import VectorStore # 建议直接在类中使用 Chroma 以确保存储逻辑控制在手
from src.agentic_rag.threshold_config import ThresholdConfig
from src.utils.config import AgenticRAGConfig
from dotenv import load_dotenv

load_dotenv()

class AgenticRAG:
    """完整的 Agentic RAG 系统"""
    
    def __init__(
        self,
        vectorstore: Optional[Chroma] = None,
        llm: Optional[ChatOpenAI] = None,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        max_iterations: int = 5,
        # 修改点 1: 设置一个默认的本地路径，确保不是 None
        persist_directory: str = "./chroma_db",
        threshold_config: Optional[ThresholdConfig] = None,
        config: Optional[AgenticRAGConfig] = None
    ):
        self.max_iterations = max_iterations
        self.embedding_model = embedding_model
        # 修改点 2: 显式保存持久化路径到实例变量
        self.persist_directory = persist_directory
        
        # 处理阈值配置：优先使用传入的 threshold_config，否则从 config 中获取，最后使用默认值
        if threshold_config is None:
            if config and config.threshold_config:
                threshold_config = config.threshold_config
            else:
                threshold_config = ThresholdConfig.default()
        self.threshold_config = threshold_config
        
        # 初始化 LLM
        self.llm = llm or LLM(model_name="openai:gpt-4o-mini", temperature=0.1).get_llm()
        
        # 初始化文本分割器
        self.init_splitter()

        # 初始化向量数据库
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            print(f"[系统] 正在连接向量数据库，路径: {self.persist_directory}")
            # 修改点 3: 直接使用 Chroma 初始化，确保我们控制持久化逻辑
            # 这样可以避免自定义 VectorStore 类可能存在的逻辑漏洞
            self.vectorstore = VectorStore(
                collection_name="agentic_rag_collection", # 建议指定集合名
                embedding_function="text-embedding-3-small",
                persist_directory=self.persist_directory
            ).get_vectorstore()
            
            try:
                # 检查是否成功加载
                count = self.vectorstore._collection.count()
                if count > 0:
                    print(f"[系统] 成功加载已有向量数据库，包含 {count} 个文档块")
                else:
                    print(f"[系统] 向量数据库已连接但为空 (Count: 0)")
            except Exception as e:
                print(f"[系统] 数据库连接异常: {str(e)}")

        # 创建图（传递 threshold_config）
        self.graph = create_agentic_rag_graph(
            vectorstore=self.vectorstore,
            llm=self.llm,
            max_iterations=max_iterations,
            threshold_config=self.threshold_config
        )

    def init_splitter(self) -> None:
        self.splitters = {
            "text/txt": DocsSplitter(
                docs_type="text/txt",
                chunk_size=500,
                chunk_overlap=100
            ).get_splitter(),
            "text/markdown": DocsSplitter(
                docs_type="text/markdown",
                chunk_size=800,
                chunk_overlap=150
            ).get_splitter(),
            "application/pdf": DocsSplitter(
                docs_type="application/pdf",
                chunk_size=1500,
                chunk_overlap=300
            ).get_splitter(),
            "application/plain": DocsSplitter(
                docs_type="application/plain",
                chunk_size=1000,
                chunk_overlap=200
            ).get_splitter(),
        }

    def load_documents(self, documents: List[Document]) -> List[Document]:
      splits = []
      for document in documents:
          if document.metadata.get("type") == "application/pdf":
              splitter = self.splitters.get("application/pdf")
          elif document.metadata.get("type") == "text/markdown":
              splitter = self.splitters.get("text/markdown")
          elif document.metadata.get("type") == "text/txt":
              splitter = self.splitters.get("text/txt")
          else:
              splitter = self.splitters.get("application/plain")
          
          items = splitter.split_documents([document])
          splits.extend(items)
          # ✅ 修复：打印当前文档的块数，而不是总块数
          print(f"[系统] 「{document.metadata.get('type')}」类型文档切分为 {len(items)} 个块")
      
      print(f"[系统] 总共切分为 {len(splits)} 个块")
      return splits
    
    def build_vectorstore(
        self, 
        documents: List[Document],
        force_rebuild: bool = False
    ):
        """构建向量数据库"""
        # 检查现有数据
        existing_count = 0
        try:
            if hasattr(self.vectorstore, '_collection'):
                existing_count = self.vectorstore._collection.count()
        except Exception:
            pass
        
        # 逻辑判断：已有数据且不强制重建，则跳过
        if existing_count > 0 and not force_rebuild:
            print(f"[系统] 向量数据库已存在数据 ({existing_count} chunks)。跳过构建。")
            return # 直接返回，不做任何操作
        
        # 开始构建逻辑
        if force_rebuild:
            print("[系统] 收到强制重建指令...")
        else:
            print("[系统] 数据库为空，开始初始化...")

        # 切分文档
        splits = self.load_documents(documents)
        
        # 修改点 4: 确保使用 self.persist_directory
        persist_dir = self.persist_directory
        
        # 清理旧数据 (如果是强制重建)
        if force_rebuild:
            # 强制重建时：始终使用现有的 vectorstore，通过 API 清空后再添加
            # 这样可以避免创建新的数据库连接，避免文件锁定问题
            if hasattr(self, 'vectorstore') and self.vectorstore is not None:
                try:
                    print(f"[系统] 强制重建：清空现有 collection...")
                    # 获取所有 IDs 并删除
                    result = self.vectorstore._collection.get()
                    existing_ids = result.get('ids', [])
                    if existing_ids:
                        self.vectorstore.delete(ids=existing_ids)
                        print(f"[系统] ✓ 已清空 {len(existing_ids)} 个文档块")
                    else:
                        print(f"[系统] collection 已为空，无需清空")
                    
                    # 清空成功后，使用现有的 vectorstore 添加新文档
                    print(f"[系统] 将新文档添加到现有 vectorstore...")
                    self.vectorstore.add_documents(splits)
                    print(f"[系统] ✓ 已添加 {len(splits)} 个新文档块")
                except Exception as e:
                    print(f"[系统] ⚠️ 强制重建失败: {str(e)}")
                    raise RuntimeError(f"强制重建失败：{str(e)}。建议先手动删除数据库目录，然后不使用 force_rebuild 重新运行。")
            else:
                raise ValueError("强制重建时，vectorstore 不存在。请先确保 AgenticRAG 已正确初始化。")
        else:
            # 非强制重建：创建新的向量数据库
            print(f"[系统] 创建新的向量数据库，写入数据到目录: {persist_dir}")
            self.vectorstore = VectorStore(
                documents=splits,
                collection_name="agentic_rag_collection", # 建议指定集合名
                embedding_function="text-embedding-3-small",
                persist_directory=self.persist_directory
            ).get_vectorstore()
        # 重新创建图（传递 threshold_config）
        self.graph = create_agentic_rag_graph(
            vectorstore=self.vectorstore,
            llm=self.llm,
            max_iterations=self.max_iterations,
            threshold_config=self.threshold_config
        )

        final_count = self.vectorstore._collection.count()
        print(f"[系统] ✓ 向量数据库构建完成。当前文档数: {final_count}")
        
    def add_documents(self, documents: List[Document], docs_type: str = None):
        """添加文档到向量数据库，如果已存在则先删除"""
        if docs_type:
            splitter = self.splitters.get(docs_type) 
            # 分割文档
            splits = splitter.split_documents(documents)
        else:
            # 如果未指定文档类型，则证明是已经切分好的文档
            splits = documents
        
        # 检查是否有相同来源的文档（通过 source metadata）
        if splits and "source" in splits[0].metadata:
            source = splits[0].metadata["source"]
            
            try:
                # 使用 metadata 过滤查找相同来源的文档
                existing_docs = self.vectorstore.get(
                    where={"source": source}
                )
                
                # 如果找到相同来源的文档，先删除
                if existing_docs.get("ids"):
                    self.vectorstore.delete(ids=existing_docs["ids"])
                    print(f"[系统] 删除已存在的文档: {source} ({len(existing_docs['ids'])} 个块)")
            except Exception as e:
                print(f"[系统] ⚠️ 删除文档时出错: {e}")
        
        # 添加新文档
        self.vectorstore.add_documents(splits)
        print(f"[系统] ✓ 添加文档: {len(splits)} 个块")
        
    def update_vectorstore(self, documents: List[Document]):
        """更新向量数据库"""
        
        self.vectorstore.add_documents(documents)

    # query 和 stream_query 方法保持不变...
    def query(self, question: str, verbose: bool = True) -> dict:
        # ... (保持原本代码)
        return super().query(question, verbose) if hasattr(super(), 'query') else self._query_impl(question, verbose)

    def _query_impl(self, question: str, verbose: bool = True) -> dict:
        # 为了完整性，这里放你原来的 query 代码逻辑
        if verbose:
            print("=" * 60)
            print(f"问题: {question}")
        
        initial_state = create_initial_state(question=question, max_iterations=self.max_iterations)
        config = {"recursion_limit": 50}
        
        try:
            final_state = self.graph.invoke(initial_state, config=config)

            if verbose:
                print(f"答案: {final_state.get('answer', '')[:100]}...")
            return {
                "question": question,
                "answer": final_state.get("answer", ""),
                "answer_quality": final_state.get("answer_quality", 0.0),
                "retrieval_quality": final_state.get("retrieval_quality", 0.0),
                "iteration_count": final_state.get("iteration_count", 0),
                "retrieval_history": final_state.get("retrieval_history", []),
                # ... 其他字段
            }
        except Exception as e:
            return {"error": str(e)}