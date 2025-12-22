"""检索结果融合算法实现"""
from typing import List, Tuple, Dict
from langchain_core.documents import Document


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[Document, float]]],
    k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion (RRF) 融合多个检索结果

    2025 最佳实践：使用 RRF 融合不同检索策略的结果

    公式：RRF(d) = Σ 1/(k + rank(d))

    Args:
        results_list: 多个检索结果列表，每个元素是 (文档, 分数) 元组列表
        k: RRF 参数（默认 60，论文推荐值）

    Returns:
        融合后的 (文档, RRF分数) 列表
    """
    # 用文档内容哈希作为唯一标识
    doc_scores: Dict[int, Tuple[Document, float]] = {}

    for results in results_list:
        for rank, (doc, _) in enumerate(results):
            doc_hash = hash(doc.page_content)
            rrf_score = 1.0 / (k + rank + 1)

            if doc_hash in doc_scores:
                # 累加 RRF 分数
                existing_doc, existing_score = doc_scores[doc_hash]
                doc_scores[doc_hash] = (existing_doc, existing_score + rrf_score)
            else:
                doc_scores[doc_hash] = (doc, rrf_score)

    # 按 RRF 分数排序
    sorted_results = sorted(
        doc_scores.values(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results

