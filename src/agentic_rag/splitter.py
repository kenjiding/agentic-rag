"""文档分块模块 - 2025 最佳实践版

基于 2025 年最新研究实现的文档分块策略：
1. 语义分块 (Semantic Chunking) - 基于嵌入相似度的断点检测
2. 结构感知分块 - 针对 Markdown/HTML 的结构化分块
3. 递归字符分块 - 作为回退策略

参考:
- https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025
- Semantic Chunking 可提高召回率 9%
"""
from typing import List, Optional, Literal
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np


class SemanticChunker:
    """语义分块器 - 基于嵌入向量相似度的断点检测

    2025 最佳实践：根据语义边界切分文档，而不是固定大小
    """

    def __init__(
        self,
        embeddings: Optional[OpenAIEmbeddings] = None,
        breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile"] = "percentile",
        breakpoint_threshold_amount: float = 95,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        """
        初始化语义分块器

        Args:
            embeddings: 嵌入模型（用于计算语义相似度）
            breakpoint_threshold_type: 断点检测方法
                - percentile: 使用百分位数作为阈值
                - standard_deviation: 使用标准差作为阈值
                - interquartile: 使用四分位数作为阈值
            breakpoint_threshold_amount: 断点阈值
                - percentile: 95 表示差异超过 95% 分位数时断开
                - standard_deviation: 3 表示差异超过 3 个标准差时断开
            min_chunk_size: 最小块大小（字符数）
            max_chunk_size: 最大块大小（字符数）
        """
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # 回退分块器（当语义分块失败时使用）
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ". ", " ", ""]
        )

    def _split_to_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        import re
        # 支持中英文句子分割
        sentence_endings = r'(?<=[。！？.!?])\s*'
        sentences = re.split(sentence_endings, text)
        # 过滤空句子
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_cosine_distances(self, embeddings: List[List[float]]) -> List[float]:
        """计算相邻句子嵌入之间的余弦距离"""
        distances = []
        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i + 1])

            # 计算余弦相似度
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            # 转换为距离（1 - 相似度）
            distance = 1 - similarity
            distances.append(distance)

        return distances

    def _get_breakpoint_threshold(self, distances: List[float]) -> float:
        """根据配置的方法计算断点阈值"""
        if not distances:
            return 0.5

        if self.breakpoint_threshold_type == "percentile":
            return np.percentile(distances, self.breakpoint_threshold_amount)
        elif self.breakpoint_threshold_type == "standard_deviation":
            mean = np.mean(distances)
            std = np.std(distances)
            return mean + self.breakpoint_threshold_amount * std
        elif self.breakpoint_threshold_type == "interquartile":
            q1 = np.percentile(distances, 25)
            q3 = np.percentile(distances, 75)
            iqr = q3 - q1
            return q3 + self.breakpoint_threshold_amount * iqr
        else:
            return np.percentile(distances, 95)

    def split_text(self, text: str) -> List[str]:
        """使用语义分块分割文本"""
        # 1. 分割为句子
        sentences = self._split_to_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        try:
            # 2. 计算每个句子的嵌入
            sentence_embeddings = self.embeddings.embed_documents(sentences)

            # 3. 计算相邻句子的距离
            distances = self._calculate_cosine_distances(sentence_embeddings)

            # 4. 确定断点阈值
            threshold = self._get_breakpoint_threshold(distances)

            # 5. 根据阈值找到断点
            chunks = []
            current_chunk = [sentences[0]]
            current_size = len(sentences[0])

            for i, (sentence, distance) in enumerate(zip(sentences[1:], distances)):
                # 检查是否需要断开
                should_break = (
                    distance > threshold or
                    current_size + len(sentence) > self.max_chunk_size
                )

                if should_break and current_size >= self.min_chunk_size:
                    # 保存当前块
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_size = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_size += len(sentence)

            # 添加最后一个块
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

        except Exception as e:
            print(f"[语义分块] 失败，回退到递归分块: {e}")
            return self.fallback_splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_method": "semantic"
                    }
                )
                result.append(new_doc)
        return result


class StructureAwareChunker:
    """结构感知分块器 - 针对 Markdown 等结构化文档

    2025 最佳实践：先按结构分块，再按大小细分
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Markdown 标题分块器
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ]
        )

        # 递归分块器（用于细分过大的块）
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        """分割 Markdown 文本"""
        try:
            # 1. 先按 Markdown 结构分块
            md_docs = self.md_splitter.split_text(text)

            # 2. 对过大的块进行细分
            result = []
            for doc in md_docs:
                content = doc.page_content
                if len(content) > self.chunk_size:
                    # 需要细分
                    sub_chunks = self.recursive_splitter.split_text(content)
                    result.extend(sub_chunks)
                else:
                    result.append(content)

            return result

        except Exception as e:
            print(f"[结构分块] 失败，回退到递归分块: {e}")
            return self.recursive_splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_method": "structure_aware"
                    }
                )
                result.append(new_doc)
        return result


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


# 便捷函数
def create_semantic_splitter(
    embeddings: Optional[OpenAIEmbeddings] = None,
    **kwargs
) -> SemanticChunker:
    """创建语义分块器的便捷函数"""
    return SemanticChunker(embeddings=embeddings, **kwargs)


def create_structure_aware_splitter(**kwargs) -> StructureAwareChunker:
    """创建结构感知分块器的便捷函数"""
    return StructureAwareChunker(**kwargs)
