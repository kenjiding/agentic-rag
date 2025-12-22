"""BM25 关键词检索器实现"""
from typing import List, Tuple
from langchain_core.documents import Document
import numpy as np

# 尝试导入 BM25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("[警告] rank-bm25 未安装，BM25 检索将不可用。运行: uv add rank-bm25")


class BM25Retriever:
    """BM25 关键词检索器

    2025 最佳实践：结合稀疏检索（BM25）和稠密检索（向量）
    """

    def __init__(self, documents: List[Document] = None):
        """
        初始化 BM25 检索器

        Args:
            documents: 文档列表
        """
        self.documents = documents or []
        self.bm25 = None
        self.tokenized_corpus = []

        if documents:
            self._build_index(documents)

    def _tokenize(self, text: str) -> List[str]:
        """分词（支持中英文）"""
        import re
        # 简单分词：按空格和标点分割
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def _build_index(self, documents: List[Document]):
        """构建 BM25 索引"""
        if not HAS_BM25:
            return

        self.documents = documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def add_documents(self, documents: List[Document]):
        """添加文档到索引"""
        self.documents.extend(documents)
        self._build_index(self.documents)

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        使用 BM25 搜索

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            (文档, 分数) 元组列表
        """
        if not HAS_BM25 or self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取 top-k
        top_indices = np.argsort(scores)[::-1][:k]
        results = [
            (self.documents[i], scores[i])
            for i in top_indices
            if scores[i] > 0
        ]

        return results

