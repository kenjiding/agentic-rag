"""指标聚合模块

提供评测指标的聚合计算。
"""
from typing import List, Dict, Any, Optional
from statistics import mean, stdev


def aggregate_scores(scores: List[float]) -> Dict[str, float]:
    """聚合评分列表
    
    Args:
        scores: 评分列表
        
    Returns:
        聚合结果 {mean, min, max, std}
    """
    if not scores:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    
    return {
        "mean": mean(scores),
        "min": min(scores),
        "max": max(scores),
        "std": stdev(scores) if len(scores) > 1 else 0.0,
    }


def compute_weighted_average(
    scores: List[float],
    weights: Optional[List[float]] = None
) -> float:
    """计算加权平均
    
    Args:
        scores: 评分列表
        weights: 权重列表（可选，默认等权重）
        
    Returns:
        加权平均分
    """
    if not scores:
        return 0.0
    
    if weights is None:
        return mean(scores)
    
    if len(weights) != len(scores):
        raise ValueError("权重列表长度必须与评分列表相同")
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / total_weight


def aggregate_by_tag(
    results: List[Dict[str, Any]],
    score_key: str = "overall_score"
) -> Dict[str, Dict[str, float]]:
    """按标签聚合评测结果
    
    Args:
        results: 评测结果列表
        score_key: 评分字段名
        
    Returns:
        按标签分组的聚合结果
    """
    tag_scores: Dict[str, List[float]] = {}
    
    for result in results:
        score = result.get(score_key, 0.0)
        tags = result.get("tags", [])
        
        for tag in tags:
            if tag not in tag_scores:
                tag_scores[tag] = []
            tag_scores[tag].append(score)
    
    return {
        tag: aggregate_scores(scores)
        for tag, scores in tag_scores.items()
    }
