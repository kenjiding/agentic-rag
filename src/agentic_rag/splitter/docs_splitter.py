"""统一文档分块器实现 - 根据文档类型选择最佳策略

2025 最佳实践：
- Markdown: 结构感知分块
- PDF/长文本: 语义分块
- 短文本: 递归字符分块
"""
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.agentic_rag.splitter.semantic_chunker import SemanticChunker
from src.agentic_rag.splitter.structure_aware_chunker import StructureAwareChunker


class DocsSplitter:
    """统一文档分块器 - 根据文档类型选择最佳策略

    2025 最佳实践：
    - Markdown: 结构感知分块
    - PDF/长文本: 语义分块
    - 短文本: 递归字符分块
    """

    def __init__(
        self,
        docs_type: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        use_semantic_chunking: bool = True,
        embeddings: Optional[OpenAIEmbeddings] = None
    ) -> None:
        if not docs_type:
            raise ValueError("docs_type is required")

        self.docs_type = docs_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking
        self.embeddings = embeddings

        # 初始化分块器
        self.splitter = self._create_splitter()

    def _create_splitter(self):
        """根据文档类型创建最适合的分块器"""

        if self.docs_type == "text/markdown":
            # Markdown 使用结构感知分块
            return StructureAwareChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        elif self.docs_type == "application/pdf":
            # PDF 使用语义分块（如果启用）
            if self.use_semantic_chunking:
                return SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=90,
                    min_chunk_size=200,
                    max_chunk_size=self.chunk_size * 2
                )
            else:
                return RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=self._get_pdf_separators()
                )

        elif self.docs_type == "text/txt":
            # 纯文本使用递归分块
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self._get_txt_separators()
            )

        else:
            # 默认使用递归分块
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )

    def _get_pdf_separators(self) -> List[str]:
        """PDF 文档的分隔符"""
        return [
            "\n\n\n\n",      # 大节分隔
            "\n\n\n",        # 节内段落分隔
            "\n\n",          # 段落分隔
            "\n• ",          # 项目符号列表
            "\n- ",          # 破折号列表
            "\n* ",          # 星号列表
            "\n  ",          # 缩进
            "\n",            # 换行
            "。",            # 中文句号
            ". ",            # 英文句号
            "：",            # 中文冒号
            ": ",            # 英文冒号
            "；",            # 中文分号
            "; ",            # 英文分号
            "，",            # 中文逗号
            ", ",            # 英文逗号
            " ",             # 空格
            ""
        ]

    def _get_txt_separators(self) -> List[str]:
        """纯文本的分隔符"""
        return [
            "\n\n",  # 段落分隔符
            "\n",    # 换行符
            "。",    # 中文句号
            ". ",    # 英文句号
            "，",    # 中文逗号
            ", ",    # 英文逗号
            " ",     # 空格
            ""       # 字符
        ]

    def get_separators(self, docs_type: str = None) -> List[str]:
        """获取分隔符列表（向后兼容）"""
        if docs_type == "text/markdown":
            return ["\n## ", "\n### ", "\n\n", "\n", "。", ". ", " ", ""]
        elif docs_type == "text/txt":
            return self._get_txt_separators()
        elif docs_type == "application/pdf":
            return self._get_pdf_separators()
        else:
            return ["\n\n", "\n", " ", ""]

    def get_splitter(self):
        """获取分块器实例"""
        return self.splitter

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档（统一接口）"""
        if hasattr(self.splitter, 'split_documents'):
            return self.splitter.split_documents(documents)
        else:
            # RecursiveCharacterTextSplitter 的情况
            return self.splitter.split_documents(documents)

