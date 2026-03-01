"""任务完成评测器

基于VitaBench/ECom-Bench的任务完成度评测方法，评估Agent是否成功完成任务目标。

核心指标：
- success_rate: 任务完全成功率
- partial_credit: 部分完成分数（milestone-based）
- outcome_match: 预期结果匹配度
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime
import logging
import re

from src.evaluation.evaluators.base import BaseAgentEvaluator
from src.evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    EvaluationStatus,
    MilestoneResult,
    OutcomeType,
    ExpectedOutcome,
)
from src.evaluation.config import EvaluationConfig

if TYPE_CHECKING:
    from src.multi_agent.state import MultiAgentState

logger = logging.getLogger(__name__)


class TaskCompletionEvaluator(BaseAgentEvaluator):
    """任务完成评测器
    
    评估Agent是否成功完成任务目标，支持：
    - 预期结果验证（状态匹配、响应包含、Agent调用等）
    - 里程碑式部分完成度评估
    - 多种验证方式组合
    
    设计原则：
    - 基于VitaBench/ECom-Bench的评测方法
    - 支持复杂的多步骤任务评估
    - 提供细粒度的评测反馈
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """初始化任务完成评测器"""
        super().__init__(config, name="TaskCompletionEvaluator")
    
    async def evaluate(
        self,
        case: EvaluationCase,
        execution_trace: List[Dict[str, Any]],
        final_state: "MultiAgentState"
    ) -> EvaluationResult:
        """评估任务完成度
        
        评测逻辑：
        1. 验证预期结果是否达成
        2. 检查里程碑完成情况
        3. 计算综合任务完成分数
        
        Args:
            case: 评测用例
            execution_trace: 执行轨迹（来自action_audit）
            final_state: 最终状态
            
        Returns:
            评测结果
        """
        result = self._create_base_result(case, execution_trace, final_state)
        
        try:
            # 1. 验证预期结果
            outcome_scores = []
            outcome_details = []
            
            for expected in case.expected_outcomes:
                score, detail = self._verify_outcome(
                    expected, execution_trace, final_state
                )
                outcome_scores.append(score * expected.weight)
                outcome_details.append({
                    "type": expected.outcome_type.value,
                    "expected": str(expected.expected_value),
                    "score": score,
                    "detail": detail
                })
            
            # 2. 检查里程碑
            milestone_results = []
            milestone_scores = []
            
            for milestone in case.milestones:
                achieved, detail = self._check_milestone(
                    milestone, execution_trace, final_state
                )
                milestone_results.append(MilestoneResult(
                    milestone_id=milestone.milestone_id,
                    achieved=achieved,
                    details=detail
                ))
                if milestone.required or achieved:
                    milestone_scores.append(
                        milestone.weight if achieved else 0.0
                    )
            
            # 3. 验证Agent和意图
            agent_match = self._verify_agent(case, result.actual_agent)
            intent_match = self._verify_intent(case, result.actual_intent)
            
            # 4. 计算任务完成分数
            task_score = self._calculate_task_score(
                outcome_scores=outcome_scores,
                milestone_scores=milestone_scores,
                agent_match=agent_match,
                intent_match=intent_match,
                case=case
            )
            
            # 5. 更新结果
            result.task_completion_score = task_score
            result.milestone_results = milestone_results
            result.overall_score = task_score  # 任务完成评测器只关注任务完成度
            
            # 判断成功
            result.success = task_score >= self.config.thresholds.success_threshold
            result.status = EvaluationStatus.SUCCESS if result.success else EvaluationStatus.FAILED
            
            # 添加详情
            result.details = {
                "outcome_verification": outcome_details,
                "milestone_completion": [m.model_dump() for m in milestone_results],
                "agent_match": agent_match,
                "intent_match": intent_match,
                "task_score_breakdown": {
                    "outcome_contribution": sum(outcome_scores) / max(len(outcome_scores), 1),
                    "milestone_contribution": sum(milestone_scores) / max(len(milestone_scores), 1),
                    "agent_bonus": 0.1 if agent_match else 0.0,
                    "intent_bonus": 0.1 if intent_match else 0.0,
                }
            }
            
        except Exception as e:
            logger.error(f"任务完成评测失败: {e}", exc_info=True)
            result.status = EvaluationStatus.ERROR
            result.error_message = str(e)
        
        result.completed_at = datetime.now()
        return result
    
    def _verify_outcome(
        self,
        expected: ExpectedOutcome,
        execution_trace: List[Dict[str, Any]],
        final_state: "MultiAgentState"
    ) -> tuple[float, str]:
        """验证预期结果
        
        Args:
            expected: 预期结果定义
            execution_trace: 执行轨迹
            final_state: 最终状态
            
        Returns:
            (分数, 详情说明)
        """
        outcome_type = expected.outcome_type
        expected_value = expected.expected_value
        tolerance = expected.tolerance
        
        if outcome_type == OutcomeType.STATE_MATCH:
            # 状态字段匹配
            return self._verify_state_match(expected_value, final_state, tolerance)
        
        elif outcome_type == OutcomeType.RESPONSE_CONTAINS:
            # 响应包含特定内容
            return self._verify_response_contains(expected_value, final_state, tolerance)
        
        elif outcome_type == OutcomeType.AGENT_CALLED:
            # 特定Agent被调用
            return self._verify_agent_called(expected_value, execution_trace)
        
        elif outcome_type == OutcomeType.TOOL_CALLED:
            # 特定工具被调用
            return self._verify_tool_called(expected_value, execution_trace)
        
        elif outcome_type == OutcomeType.MILESTONE_REACHED:
            # 达成特定里程碑（委托给milestone检查）
            return 1.0, "delegated_to_milestone_check"
        
        return 0.0, f"unknown_outcome_type: {outcome_type}"
    
    def _verify_state_match(
        self,
        expected_value: Dict[str, Any],
        final_state: "MultiAgentState",
        tolerance: float
    ) -> tuple[float, str]:
        """验证状态字段匹配
        
        Args:
            expected_value: 期望的状态字段值 {"field": "value"}
            final_state: 最终状态
            tolerance: 容差
            
        Returns:
            (分数, 详情说明)
        """
        if not isinstance(expected_value, dict):
            return 0.0, "expected_value must be dict for STATE_MATCH"
        
        state_dict = final_state.model_dump()
        matches = 0
        total = len(expected_value)
        details = []
        
        for field, expected_val in expected_value.items():
            actual_val = state_dict.get(field)
            if actual_val == expected_val:
                matches += 1
                details.append(f"{field}: matched")
            elif tolerance > 0 and isinstance(actual_val, (int, float)) and isinstance(expected_val, (int, float)):
                # 数值类型支持容差匹配
                if abs(actual_val - expected_val) <= tolerance * abs(expected_val):
                    matches += 1
                    details.append(f"{field}: matched_with_tolerance")
                else:
                    details.append(f"{field}: mismatch ({actual_val} != {expected_val})")
            else:
                details.append(f"{field}: mismatch ({actual_val} != {expected_val})")
        
        score = matches / total if total > 0 else 0.0
        return score, "; ".join(details)
    
    def _verify_response_contains(
        self,
        expected_value: str,
        final_state: "MultiAgentState",
        tolerance: float
    ) -> tuple[float, str]:
        """验证响应包含特定内容
        
        Args:
            expected_value: 期望包含的文本或正则模式
            final_state: 最终状态
            tolerance: 容差（用于模糊匹配）
            
        Returns:
            (分数, 详情说明)
        """
        # 从状态中获取响应内容
        response = final_state.content or ""
        
        # 也检查最后的消息
        if final_state.messages:
            last_msg = final_state.messages[-1]
            if hasattr(last_msg, 'content'):
                response = last_msg.content or response
        
        # 精确包含检查
        if expected_value in response:
            return 1.0, "exact_match_found"
        
        # 不区分大小写检查
        if expected_value.lower() in response.lower():
            return 0.9, "case_insensitive_match"
        
        # 正则模式匹配
        try:
            if re.search(expected_value, response, re.IGNORECASE):
                return 0.8, "regex_match_found"
        except re.error:
            pass
        
        # 模糊匹配（如果启用容差）
        if tolerance > 0:
            # 简单的词重叠度计算
            expected_words = set(expected_value.lower().split())
            response_words = set(response.lower().split())
            overlap = len(expected_words & response_words) / len(expected_words) if expected_words else 0
            if overlap >= (1 - tolerance):
                return overlap, f"partial_match ({overlap:.2%})"
        
        return 0.0, "no_match_found"
    
    def _verify_agent_called(
        self,
        expected_agent: str,
        execution_trace: List[Dict[str, Any]]
    ) -> tuple[float, str]:
        """验证特定Agent被调用
        
        Args:
            expected_agent: 期望的Agent名称
            execution_trace: 执行轨迹
            
        Returns:
            (分数, 详情说明)
        """
        for event in execution_trace:
            agent_name = event.get("agent_name") or event.get("node", "")
            if expected_agent.lower() in agent_name.lower():
                return 1.0, f"agent_called: {agent_name}"
        
        return 0.0, f"agent_not_called: {expected_agent}"
    
    def _verify_tool_called(
        self,
        expected_tool: str,
        execution_trace: List[Dict[str, Any]]
    ) -> tuple[float, str]:
        """验证特定工具被调用
        
        Args:
            expected_tool: 期望的工具名称
            execution_trace: 执行轨迹
            
        Returns:
            (分数, 详情说明)
        """
        tools = self._extract_tool_calls(execution_trace)
        
        for tool in tools:
            if expected_tool.lower() in tool.lower():
                return 1.0, f"tool_called: {tool}"
        
        return 0.0, f"tool_not_called: {expected_tool}"
    
    def _check_milestone(
        self,
        milestone: "Milestone",
        execution_trace: List[Dict[str, Any]],
        final_state: "MultiAgentState"
    ) -> tuple[bool, str]:
        """检查里程碑是否达成
        
        Args:
            milestone: 里程碑定义
            execution_trace: 执行轨迹
            final_state: 最终状态
            
        Returns:
            (是否达成, 详情说明)
        """
        from src.evaluation.models import Milestone
        
        condition_type = milestone.condition_type
        condition_value = milestone.condition_value
        
        if condition_type == "event_exists":
            # 检查事件是否存在
            event_type = condition_value.get("event_type")
            for event in execution_trace:
                if event.get("event") == event_type:
                    return True, f"event_found: {event_type}"
            return False, f"event_not_found: {event_type}"
        
        elif condition_type == "state_field":
            # 检查状态字段
            field = condition_value.get("field")
            expected = condition_value.get("value")
            actual = getattr(final_state, field, None)
            if actual == expected:
                return True, f"state_match: {field}={expected}"
            return False, f"state_mismatch: {field} ({actual} != {expected})"
        
        elif condition_type == "response_pattern":
            # 检查响应模式
            pattern = condition_value.get("pattern", "")
            response = final_state.content or ""
            if re.search(pattern, response, re.IGNORECASE):
                return True, f"pattern_matched: {pattern}"
            return False, f"pattern_not_matched: {pattern}"
        
        return False, f"unknown_condition: {condition_type}"
    
    def _verify_agent(self, case: EvaluationCase, actual_agent: Optional[str]) -> bool:
        """验证Agent匹配"""
        if not case.expected_agent:
            return True  # 未指定期望Agent，视为匹配
        if not actual_agent:
            return False
        return case.expected_agent.lower() in actual_agent.lower()
    
    def _verify_intent(self, case: EvaluationCase, actual_intent: Optional[str]) -> bool:
        """验证意图匹配"""
        if not case.expected_intent:
            return True  # 未指定期望意图，视为匹配
        if not actual_intent:
            return False
        return case.expected_intent.lower() == actual_intent.lower()
    
    def _calculate_task_score(
        self,
        outcome_scores: List[float],
        milestone_scores: List[float],
        agent_match: bool,
        intent_match: bool,
        case: EvaluationCase
    ) -> float:
        """计算综合任务完成分数
        
        评分策略：
        - 预期结果: 60%权重
        - 里程碑: 30%权重
        - Agent/意图匹配: 10%奖励
        
        Args:
            outcome_scores: 预期结果分数列表
            milestone_scores: 里程碑分数列表
            agent_match: Agent是否匹配
            intent_match: 意图是否匹配
            case: 评测用例
            
        Returns:
            综合分数 (0-1)
        """
        # 计算预期结果平均分
        outcome_avg = sum(outcome_scores) / len(outcome_scores) if outcome_scores else 0.5
        
        # 计算里程碑完成率
        milestone_avg = sum(milestone_scores) / len(milestone_scores) if milestone_scores else 0.5
        
        # 基础分数
        base_score = outcome_avg * 0.6 + milestone_avg * 0.3
        
        # 奖励分数
        bonus = 0.0
        if agent_match:
            bonus += 0.05
        if intent_match:
            bonus += 0.05
        
        # 综合分数
        total_score = base_score + bonus
        
        # 如果没有预期结果和里程碑定义，使用Agent/意图匹配作为主要指标
        if not case.expected_outcomes and not case.milestones:
            total_score = 1.0 if (agent_match and intent_match) else 0.5
        
        return min(1.0, max(0.0, total_score))
