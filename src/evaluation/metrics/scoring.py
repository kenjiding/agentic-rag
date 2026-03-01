"""评分计算模块

提供常用的评分计算函数。
"""
from typing import List, Set, Tuple, Any


def compute_f1_score(
    actual: List[Any],
    expected: List[Any]
) -> Tuple[float, float, float]:
    """计算F1分数
    
    Args:
        actual: 实际值列表
        expected: 期望值列表
        
    Returns:
        (precision, recall, f1)
    """
    if not expected:
        return (1.0, 1.0, 1.0) if not actual else (0.0, 1.0, 0.0)
    
    if not actual:
        return (0.0, 0.0, 0.0)
    
    actual_set = set(actual)
    expected_set = set(expected)
    
    true_positives = actual_set & expected_set
    
    precision = len(true_positives) / len(actual_set)
    recall = len(true_positives) / len(expected_set)
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return (precision, recall, f1)


def compute_accuracy(
    actual: List[Any],
    expected: List[Any]
) -> float:
    """计算准确率
    
    Args:
        actual: 实际值列表
        expected: 期望值列表
        
    Returns:
        准确率 (0-1)
    """
    if not expected:
        return 1.0 if not actual else 0.0
    
    if len(actual) != len(expected):
        # 长度不匹配时，计算匹配比例
        matches = sum(1 for a in actual if a in expected)
        return matches / max(len(actual), len(expected))
    
    matches = sum(1 for a, e in zip(actual, expected) if a == e)
    return matches / len(expected)


def compute_precision_recall(
    actual: Set[Any],
    expected: Set[Any]
) -> Tuple[float, float]:
    """计算精确率和召回率
    
    Args:
        actual: 实际值集合
        expected: 期望值集合
        
    Returns:
        (precision, recall)
    """
    if not expected:
        return (1.0, 1.0) if not actual else (0.0, 1.0)
    
    if not actual:
        return (0.0, 0.0)
    
    true_positives = actual & expected
    
    precision = len(true_positives) / len(actual)
    recall = len(true_positives) / len(expected)
    
    return (precision, recall)
