"""轨迹质量评测器

基于ICLR 2026 Agent Evaluation Guide的轨迹评测方法，评估Agent的执行轨迹质量。

核心指标：
- tool_selection_accuracy: 工具选择准确率 (Node F1)
- tool_sequence_correctness: 工具调用顺序正确率 (Edge F1)
- step_efficiency: 步骤效率（实际步数/最优步数）
"""
from typing import List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from datetime import datetime
import logging
from collections import Counter

from src.evaluation.evaluators.base import BaseAgentEvaluator
from src.evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    EvaluationStatus,
)
from src.evaluation.config import EvaluationConfig

if TYPE_CHECKING:
    from src.multi_agent.state import MultiAgentState

logger = logging.getLogger(__name__)


class TrajectoryEvaluator(BaseAgentEvaluator):
    """轨迹质量评测器
    
    评估Agent执行轨迹的质量，包括：
    - 工具选择准确率（Node F1）
    - 工具调用顺序（Edge F1）
    - 执行效率
    
    设计原则：
    - 基于ICLR 2026的轨迹评测方法
    - 支持图结构的轨迹分析
    - 提供多维度的轨迹质量评估
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """初始化轨迹评测器"""
        super().__init__(config, name="TrajectoryEvaluator")
    
    async def evaluate(
        self,
        case: EvaluationCase,
        execution_trace: List[Dict[str, Any]],
        final_state: "MultiAgentState"
    ) -> EvaluationResult:
        """评估轨迹质量
        
        评测逻辑：
        1. 提取实际工具调用序列
        2. 计算Node F1（工具选择准确率）
        3. 计算Edge F1（工具顺序正确率）
        4. 计算执行效率
        
        Args:
            case: 评测用例
            execution_trace: 执行轨迹（来自action_audit）
            final_state: 最终状态
            
        Returns:
            评测结果
        """
        result = self._create_base_result(case, execution_trace, final_state)
        
        try:
            # 1. 提取实际工具调用
            actual_tools = self._extract_tool_calls(execution_trace)
            expected_tools = case.expected_tool_calls
            
            # 2. 计算Node F1（工具选择准确率）
            node_f1, node_details = self._compute_node_f1(actual_tools, expected_tools)
            
            # 3. 计算Edge F1（工具顺序正确率）
            edge_f1, edge_details = self._compute_edge_f1(actual_tools, expected_tools)
            
            # 4. 计算步骤效率
            efficiency, efficiency_details = self._compute_efficiency(
                execution_trace, expected_tools
            )
            
            # 5. 计算归一化编辑距离
            edit_distance = self._compute_normalized_edit_distance(
                actual_tools, expected_tools
            )
            
            # 6. 提取节点执行信息
            node_analysis = self._analyze_node_execution(execution_trace)
            
            # 7. 计算综合轨迹质量分数
            trajectory_score = self._calculate_trajectory_score(
                node_f1=node_f1,
                edge_f1=edge_f1,
                efficiency=efficiency,
                edit_distance=edit_distance
            )
            
            # 8. 更新结果
            result.trajectory_quality_score = trajectory_score
            result.tool_accuracy_score = node_f1
            result.overall_score = trajectory_score
            result.step_count = len(execution_trace)
            
            # 判断成功（轨迹评测器关注工具准确率）
            result.success = node_f1 >= self.config.thresholds.tool_accuracy_threshold
            result.status = EvaluationStatus.SUCCESS if result.success else EvaluationStatus.FAILED
            
            # 添加详情
            result.details = {
                "node_f1": {
                    "score": node_f1,
                    **node_details
                },
                "edge_f1": {
                    "score": edge_f1,
                    **edge_details
                },
                "efficiency": {
                    "score": efficiency,
                    **efficiency_details
                },
                "edit_distance": edit_distance,
                "actual_tools": actual_tools,
                "expected_tools": expected_tools,
                "node_analysis": node_analysis,
                "trajectory_score_breakdown": {
                    "node_f1_contribution": node_f1 * 0.4,
                    "edge_f1_contribution": edge_f1 * 0.3,
                    "efficiency_contribution": efficiency * 0.2,
                    "edit_distance_contribution": (1 - edit_distance) * 0.1,
                }
            }
            
        except Exception as e:
            logger.error(f"轨迹质量评测失败: {e}", exc_info=True)
            result.status = EvaluationStatus.ERROR
            result.error_message = str(e)
        
        result.completed_at = datetime.now()
        return result
    
    def _compute_node_f1(
        self,
        actual: List[str],
        expected: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """计算Node F1分数
        
        Node F1衡量工具选择的准确性，不考虑顺序。
        
        Args:
            actual: 实际工具调用列表
            expected: 期望工具调用列表
            
        Returns:
            (F1分数, 详细信息)
        """
        if not expected:
            # 无期望工具时，返回1.0（不评估）
            return 1.0, {"note": "no_expected_tools"}
        
        if not actual:
            return 0.0, {"note": "no_actual_tools"}
        
        # 转换为小写集合进行比较
        actual_set = set(t.lower() for t in actual)
        expected_set = set(t.lower() for t in expected)
        
        # 计算交集
        true_positives = actual_set & expected_set
        
        # 计算Precision和Recall
        precision = len(true_positives) / len(actual_set) if actual_set else 0.0
        recall = len(true_positives) / len(expected_set) if expected_set else 0.0
        
        # 计算F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return f1, {
            "precision": precision,
            "recall": recall,
            "true_positives": list(true_positives),
            "false_positives": list(actual_set - expected_set),
            "false_negatives": list(expected_set - actual_set),
        }
    
    def _compute_edge_f1(
        self,
        actual: List[str],
        expected: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """计算Edge F1分数
        
        Edge F1衡量工具调用顺序的正确性，通过比较相邻工具对。
        
        Args:
            actual: 实际工具调用序列
            expected: 期望工具调用序列
            
        Returns:
            (F1分数, 详细信息)
        """
        if len(expected) < 2:
            return 1.0, {"note": "insufficient_expected_edges"}
        
        if len(actual) < 2:
            return 0.0, {"note": "insufficient_actual_edges"}
        
        # 提取边（相邻工具对）
        actual_edges = self._extract_edges(actual)
        expected_edges = self._extract_edges(expected)
        
        # 转换为集合
        actual_edge_set = set(actual_edges)
        expected_edge_set = set(expected_edges)
        
        # 计算交集
        true_positives = actual_edge_set & expected_edge_set
        
        # 计算Precision和Recall
        precision = len(true_positives) / len(actual_edge_set) if actual_edge_set else 0.0
        recall = len(true_positives) / len(expected_edge_set) if expected_edge_set else 0.0
        
        # 计算F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return f1, {
            "precision": precision,
            "recall": recall,
            "actual_edges": list(actual_edge_set),
            "expected_edges": list(expected_edge_set),
            "matched_edges": list(true_positives),
        }
    
    def _extract_edges(self, tools: List[str]) -> List[Tuple[str, str]]:
        """提取工具序列中的边（相邻对）
        
        Args:
            tools: 工具列表
            
        Returns:
            边列表 [(tool1, tool2), ...]
        """
        edges = []
        tools_lower = [t.lower() for t in tools]
        for i in range(len(tools_lower) - 1):
            edges.append((tools_lower[i], tools_lower[i + 1]))
        return edges
    
    def _compute_efficiency(
        self,
        execution_trace: List[Dict[str, Any]],
        expected_tools: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """计算执行效率
        
        效率 = min(期望步数 / 实际步数, 1.0)
        
        Args:
            execution_trace: 执行轨迹
            expected_tools: 期望工具列表
            
        Returns:
            (效率分数, 详细信息)
        """
        # 计算实际步数（从轨迹中提取有效事件数）
        actual_steps = len([
            e for e in execution_trace
            if e.get("event") not in ("start", "end", "debug")
        ])
        
        # 期望步数（期望工具数 + 一些基本节点）
        expected_steps = max(len(expected_tools), 1)
        
        # 计算效率
        if actual_steps == 0:
            efficiency = 0.0
        elif actual_steps <= expected_steps:
            efficiency = 1.0
        else:
            # 超出期望步数，效率递减
            efficiency = expected_steps / actual_steps
        
        return efficiency, {
            "actual_steps": actual_steps,
            "expected_steps": expected_steps,
            "overhead": max(0, actual_steps - expected_steps),
        }
    
    def _compute_normalized_edit_distance(
        self,
        actual: List[str],
        expected: List[str]
    ) -> float:
        """计算归一化编辑距离
        
        衡量实际序列与期望序列的相似度。
        
        Args:
            actual: 实际工具序列
            expected: 期望工具序列
            
        Returns:
            归一化编辑距离 (0-1, 越小越相似)
        """
        if not expected:
            return 0.0 if not actual else 1.0
        
        # 转换为小写
        actual_lower = [t.lower() for t in actual]
        expected_lower = [t.lower() for t in expected]
        
        # 计算Levenshtein距离
        distance = self._levenshtein_distance(actual_lower, expected_lower)
        
        # 归一化
        max_len = max(len(actual_lower), len(expected_lower))
        normalized = distance / max_len if max_len > 0 else 0.0
        
        return min(1.0, normalized)
    
    def _levenshtein_distance(self, s1: List[str], s2: List[str]) -> int:
        """计算Levenshtein编辑距离
        
        Args:
            s1: 序列1
            s2: 序列2
            
        Returns:
            编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # 插入、删除、替换操作
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _analyze_node_execution(
        self,
        execution_trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析节点执行情况
        
        Args:
            execution_trace: 执行轨迹
            
        Returns:
            节点执行分析结果
        """
        node_counts: Dict[str, int] = Counter()
        event_types: Dict[str, int] = Counter()
        error_events: List[Dict[str, Any]] = []
        
        for event in execution_trace:
            node = event.get("node", "unknown")
            node_counts[node] += 1
            
            event_type = event.get("event", "unknown")
            event_types[event_type] += 1
            
            # 收集错误事件
            if "error" in event_type.lower() or "fail" in event_type.lower():
                error_events.append(event)
        
        return {
            "node_counts": dict(node_counts),
            "event_types": dict(event_types),
            "total_nodes": len(node_counts),
            "total_events": len(execution_trace),
            "error_count": len(error_events),
            "errors": error_events[:5],  # 最多保留5个错误
        }
    
    def _calculate_trajectory_score(
        self,
        node_f1: float,
        edge_f1: float,
        efficiency: float,
        edit_distance: float
    ) -> float:
        """计算综合轨迹质量分数
        
        评分权重：
        - Node F1: 40%（工具选择最重要）
        - Edge F1: 30%（顺序其次）
        - 效率: 20%
        - 编辑距离: 10%
        
        Args:
            node_f1: Node F1分数
            edge_f1: Edge F1分数
            efficiency: 效率分数
            edit_distance: 归一化编辑距离
            
        Returns:
            综合分数 (0-1)
        """
        # 编辑距离转换为相似度分数
        similarity = 1 - edit_distance
        
        trajectory_score = (
            node_f1 * 0.4 +
            edge_f1 * 0.3 +
            efficiency * 0.2 +
            similarity * 0.1
        )
        
        return min(1.0, max(0.0, trajectory_score))
