"""语义分块器实现 - 基于嵌入向量相似度的断点检测

2025 最佳实践：根据语义边界切分文档，而不是固定大小
"""
from typing import List, Optional, Literal
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        """
        计算相邻句子嵌入之间的余弦距离（企业级最佳实践）
        
        注意：这里已经有embedding了，所以直接计算相似度，不需要重新计算embedding。
        使用统一的余弦相似度计算方法。
        """
        distances = []
        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i + 1])

            # 企业级最佳实践：使用统一的余弦相似度计算方法
            # 计算点积
            dot_product = np.dot(vec1, vec2)
            # 计算向量长度
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # 避免除零
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                # 计算余弦相似度
                similarity = dot_product / (norm1 * norm2)
            
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

