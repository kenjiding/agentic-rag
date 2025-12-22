"""Cross-encoder 重排序器实现"""
from typing import List, Tuple
from langchain_core.documents import Document

# 尝试导入 Cross-encoder
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    print("[警告] sentence-transformers 未安装，Cross-encoder 重排序将不可用。运行: uv add sentence-transformers")


class CrossEncoderReranker:
    """Cross-encoder 重排序器

    2025 最佳实践：使用 Cross-encoder 进行精确重排序
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        初始化 Cross-encoder

        Args:
            model_name: 模型名称
        """
        self.model = None
        self.model_name = model_name

        if HAS_CROSS_ENCODER:
            try:
                self.model = CrossEncoder(model_name)
                print(f"[检索] Cross-encoder 重排序器已加载: {model_name}")
            except Exception as e:
                print(f"[警告] 加载 Cross-encoder 失败: {e}")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回 top-k 个结果

        Returns:
            (文档, 分数) 元组列表
        """
        if not self.model or not documents:
            return [(doc, 0.0) for doc in documents]

        # 准备输入对
        pairs = [[query, doc.page_content] for doc in documents]

        # 计算分数
        scores = self.model.predict(pairs)

        # 排序
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            doc_scores = doc_scores[:top_k]

        return doc_scores

