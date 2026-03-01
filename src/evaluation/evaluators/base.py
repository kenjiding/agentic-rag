"""评测器基类

定义评测器的抽象接口和通用实现，采用与RetrievalQualityEvaluator一致的设计模式。

设计原则：
1. 抽象接口：定义统一的evaluate方法
2. 可配置：支持通过配置定制评测行为
3. 可组合：支持组合多个评测器
4. 可扩展：易于添加新的评测维度
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime
import logging

from src.evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    EvaluationStatus,
)
from src.evaluation.config import EvaluationConfig

if TYPE_CHECKING:
    from src.multi_agent.state import MultiAgentState

logger = logging.getLogger(__name__)


class BaseAgentEvaluator(ABC):
    """评测器基类
    
    定义评测器的标准接口，所有具体评测器必须实现此接口。
    
    设计原则：
    - 复用现有RetrievalQualityEvaluator的架构模式
    - 支持异步评测（适配LangGraph的异步执行）
    - 提供丰富的调试信息
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        name: Optional[str] = None
    ):
        """初始化评测器
        
        Args:
            config: 评测配置
            name: 评测器名称（用于日志和报告）
        """
        self.config = config or EvaluationConfig.default()
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    async def evaluate(
        self,
        case: EvaluationCase,
        execution_trace: List[Dict[str, Any]],
        final_state: "MultiAgentState"
    ) -> EvaluationResult:
        """执行评测
        
        Args:
            case: 评测用例
            execution_trace: 执行轨迹（来自action_audit）
            final_state: 最终状态
            
        Returns:
            评测结果
        """
        pass
    
    def _create_base_result(
        self,
        case: EvaluationCase,
        execution_trace: List[Dict[str, Any]],
        final_state: "MultiAgentState"
    ) -> EvaluationResult:
        """创建基础评测结果
        
        提取通用信息，供子类复用。
        
        Args:
            case: 评测用例
            execution_trace: 执行轨迹
            final_state: 最终状态
            
        Returns:
            预填充的评测结果
        """
        # 从执行轨迹提取信息
        actual_tools = self._extract_tool_calls(execution_trace)
        actual_agent = self._extract_agent_name(execution_trace)
        actual_intent = self._extract_intent(execution_trace)
        
        return EvaluationResult(
            case_id=case.case_id,
            status=EvaluationStatus.RUNNING,
            actual_tool_calls=actual_tools,
            actual_agent=actual_agent,
            actual_intent=actual_intent,
            execution_trace=execution_trace,
            started_at=datetime.now()
        )
    
    def _extract_tool_calls(self, execution_trace: List[Dict[str, Any]]) -> List[str]:
        """从执行轨迹提取工具调用序列
        
        Args:
            execution_trace: action_audit轨迹
            
        Returns:
            工具名称列表
        """
        tools = []
        for event in execution_trace:
            # 从不同类型的事件中提取工具调用
            if event.get("event") == "tool_called":
                tool_name = event.get("tool_name")
                if tool_name:
                    tools.append(tool_name)
            elif "tool" in event:
                tools.append(event["tool"])
            elif event.get("node") and "agent" in event.get("node", "").lower():
                # 从agent节点事件中提取
                tool_name = event.get("tool_name") or event.get("action")
                if tool_name:
                    tools.append(tool_name)
        return tools
    
    def _extract_agent_name(self, execution_trace: List[Dict[str, Any]]) -> Optional[str]:
        """从执行轨迹提取最后执行的Agent名称
        
        Args:
            execution_trace: action_audit轨迹
            
        Returns:
            Agent名称
        """
        for event in reversed(execution_trace):
            if event.get("event") == "agent_executed":
                return event.get("agent_name")
            elif event.get("node") and "agent" in event.get("node", "").lower():
                return event.get("node")
        return None
    
    def _extract_intent(self, execution_trace: List[Dict[str, Any]]) -> Optional[str]:
        """从执行轨迹提取意图识别结果
        
        Args:
            execution_trace: action_audit轨迹
            
        Returns:
            意图类型
        """
        for event in execution_trace:
            if event.get("event") == "intent_classified":
                return event.get("intent_type")
            elif event.get("node") == "intent_router":
                return event.get("intent_type") or event.get("business_intent_type")
        return None
    
    def _calculate_overall_score(
        self,
        task_score: float,
        trajectory_score: float,
        tool_score: float,
        policy_score: float
    ) -> float:
        """计算综合评分
        
        使用配置的权重计算加权平均分。
        
        Args:
            task_score: 任务完成分数
            trajectory_score: 轨迹质量分数
            tool_score: 工具准确率分数
            policy_score: 策略合规分数
            
        Returns:
            综合评分 (0-1)
        """
        weights = self.config.weights
        
        # 使用配置的权重
        # 注意：这里使用task_completion和consistency的权重来平衡不同维度
        overall = (
            task_score * weights.task_completion +
            trajectory_score * weights.trajectory_quality +
            tool_score * weights.consistency +  # 工具准确率使用一致性权重
            policy_score * weights.safety
        )
        
        return min(1.0, max(0.0, overall))


class CompositeEvaluator(BaseAgentEvaluator):
    """组合评测器
    
    组合多个评测器，汇总各维度的评测结果。
    用于执行全面的评测。
    """
    
    def __init__(
        self,
        evaluators: List[BaseAgentEvaluator],
        config: Optional[EvaluationConfig] = None
    ):
        """初始化组合评测器
        
        Args:
            evaluators: 子评测器列表
            config: 评测配置
        """
        super().__init__(config, name="CompositeEvaluator")
        self.evaluators = evaluators
    
    async def evaluate(
        self,
        case: EvaluationCase,
        execution_trace: List[Dict[str, Any]],
        final_state: "MultiAgentState"
    ) -> EvaluationResult:
        """执行组合评测
        
        依次执行所有子评测器，汇总结果。
        
        Args:
            case: 评测用例
            execution_trace: 执行轨迹
            final_state: 最终状态
            
        Returns:
            汇总的评测结果
        """
        # 创建基础结果
        result = self._create_base_result(case, execution_trace, final_state)
        
        # 收集各评测器的结果
        sub_results: Dict[str, EvaluationResult] = {}
        
        for evaluator in self.evaluators:
            try:
                sub_result = await evaluator.evaluate(case, execution_trace, final_state)
                sub_results[evaluator.name] = sub_result
                
                # 合并分数（取各维度的最佳值）
                result.task_completion_score = max(
                    result.task_completion_score,
                    sub_result.task_completion_score
                )
                result.trajectory_quality_score = max(
                    result.trajectory_quality_score,
                    sub_result.trajectory_quality_score
                )
                result.tool_accuracy_score = max(
                    result.tool_accuracy_score,
                    sub_result.tool_accuracy_score
                )
                # 策略合规取最低值（保守策略）
                result.policy_compliance_score = min(
                    result.policy_compliance_score,
                    sub_result.policy_compliance_score
                )
                
                # 合并违规记录
                result.policy_violations.extend(sub_result.policy_violations)
                
                # 合并里程碑结果
                result.milestone_results.extend(sub_result.milestone_results)
                
            except Exception as e:
                logger.error(f"评测器 {evaluator.name} 执行失败: {e}")
                result.details[f"{evaluator.name}_error"] = str(e)
        
        # 计算综合评分
        result.overall_score = self._calculate_overall_score(
            result.task_completion_score,
            result.trajectory_quality_score,
            result.tool_accuracy_score,
            result.policy_compliance_score
        )
        
        # 判断是否成功
        result.success = (
            result.overall_score >= self.config.thresholds.success_threshold and
            len([v for v in result.policy_violations if v.severity == "critical"]) == 0
        )
        
        # 设置状态
        result.status = EvaluationStatus.SUCCESS if result.success else EvaluationStatus.FAILED
        result.completed_at = datetime.now()
        
        # 保存子评测器详情
        result.details["sub_evaluators"] = {
            name: sub.to_dict() for name, sub in sub_results.items()
        }
        
        return result
