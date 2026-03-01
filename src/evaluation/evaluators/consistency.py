"""一致性评测器

基于τ-bench的一致性评测方法，评估Agent在多次执行中的一致性表现。

核心指标：
- pass@k: k次尝试中至少成功1次的概率
- pass^k: k次尝试全部成功的概率
- success_rate: 成功率
"""
from typing import List, Dict, Any, Optional, Callable, Awaitable, TYPE_CHECKING
from datetime import datetime
import asyncio
import logging
import uuid

from src.evaluation.evaluators.base import BaseAgentEvaluator
from src.evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    EvaluationStatus,
    ConsistencyMetrics,
)
from src.evaluation.config import EvaluationConfig

if TYPE_CHECKING:
    from src.multi_agent.state import MultiAgentState

logger = logging.getLogger(__name__)


# 定义执行函数类型
ExecutorFunc = Callable[[EvaluationCase], Awaitable[tuple[List[Dict[str, Any]], "MultiAgentState"]]]


class ConsistencyEvaluator(BaseAgentEvaluator):
    """一致性评测器
    
    评估Agent在多次执行中的一致性表现，支持：
    - pass@k: 至少成功一次
    - pass^k: 全部成功
    - 成功率统计
    
    设计原则：
    - 基于τ-bench的一致性评测方法
    - 支持并行执行多次测试
    - 支持不同温度参数的变化测试
    
    注意：一致性评测需要执行器函数来多次运行测试用例。
    """
    
    def __init__(
        self,
        executor: Optional[ExecutorFunc] = None,
        base_evaluator: Optional[BaseAgentEvaluator] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """初始化一致性评测器
        
        Args:
            executor: 执行器函数，用于运行单次评测
            base_evaluator: 基础评测器，用于评估单次执行结果
            config: 评测配置
        """
        super().__init__(config, name="ConsistencyEvaluator")
        self.executor = executor
        self.base_evaluator = base_evaluator
    
    def set_executor(self, executor: ExecutorFunc) -> None:
        """设置执行器函数"""
        self.executor = executor
    
    def set_base_evaluator(self, evaluator: BaseAgentEvaluator) -> None:
        """设置基础评测器"""
        self.base_evaluator = evaluator
    
    async def evaluate(
        self,
        case: EvaluationCase,
        execution_trace: List[Dict[str, Any]],
        final_state: "MultiAgentState"
    ) -> EvaluationResult:
        """评估单次执行的一致性贡献
        
        注意：这个方法用于评估单次执行结果。
        要进行完整的一致性评测，请使用 evaluate_consistency 方法。
        
        Args:
            case: 评测用例
            execution_trace: 执行轨迹
            final_state: 最终状态
            
        Returns:
            评测结果（包含一致性相关信息）
        """
        result = self._create_base_result(case, execution_trace, final_state)
        
        # 单次评测时，一致性分数等于任务完成分数
        # 真正的一致性评测需要通过 evaluate_consistency 方法
        
        if self.base_evaluator:
            base_result = await self.base_evaluator.evaluate(
                case, execution_trace, final_state
            )
            result.task_completion_score = base_result.task_completion_score
            result.trajectory_quality_score = base_result.trajectory_quality_score
            result.tool_accuracy_score = base_result.tool_accuracy_score
            result.success = base_result.success
            result.details["base_evaluation"] = base_result.to_dict()
        
        result.status = EvaluationStatus.SUCCESS if result.success else EvaluationStatus.FAILED
        result.completed_at = datetime.now()
        
        return result
    
    async def evaluate_consistency(
        self,
        case: EvaluationCase,
        k: Optional[int] = None
    ) -> tuple[EvaluationResult, ConsistencyMetrics]:
        """执行完整的一致性评测
        
        运行k次评测，计算pass@k和pass^k指标。
        
        Args:
            case: 评测用例
            k: 执行次数（默认使用配置值）
            
        Returns:
            (汇总结果, 一致性指标)
            
        Raises:
            ValueError: 如果未设置执行器或基础评测器
        """
        if not self.executor:
            raise ValueError("未设置执行器函数，请先调用 set_executor")
        
        if not self.base_evaluator:
            raise ValueError("未设置基础评测器，请先调用 set_base_evaluator")
        
        k = k or self.config.consistency.k
        
        # 执行k次测试
        results: List[EvaluationResult] = []
        individual_success: List[bool] = []
        
        if self.config.consistency.parallel_runs:
            # 并行执行
            tasks = [
                self._run_single_evaluation(case, run_index=i)
                for i in range(k)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常
            processed_results = []
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"一致性测试执行失败: {r}")
                    error_result = EvaluationResult(
                        case_id=case.case_id,
                        status=EvaluationStatus.ERROR,
                        error_message=str(r)
                    )
                    processed_results.append(error_result)
                    individual_success.append(False)
                else:
                    processed_results.append(r)
                    individual_success.append(r.success)
            results = processed_results
        else:
            # 顺序执行
            for i in range(k):
                try:
                    result = await self._run_single_evaluation(case, run_index=i)
                    results.append(result)
                    individual_success.append(result.success)
                except Exception as e:
                    logger.error(f"一致性测试执行失败 (run {i}): {e}")
                    error_result = EvaluationResult(
                        case_id=case.case_id,
                        status=EvaluationStatus.ERROR,
                        error_message=str(e)
                    )
                    results.append(error_result)
                    individual_success.append(False)
        
        # 计算一致性指标
        pass_at_k = any(individual_success)
        pass_power_k = all(individual_success)
        success_rate = sum(individual_success) / k if k > 0 else 0.0
        
        consistency_metrics = ConsistencyMetrics(
            k=k,
            pass_at_k=pass_at_k,
            pass_power_k=pass_power_k,
            success_rate=success_rate,
            individual_results=individual_success
        )
        
        # 汇总结果
        summary_result = self._aggregate_results(case, results, consistency_metrics)
        
        return summary_result, consistency_metrics
    
    async def _run_single_evaluation(
        self,
        case: EvaluationCase,
        run_index: int
    ) -> EvaluationResult:
        """执行单次评测
        
        Args:
            case: 评测用例
            run_index: 执行索引
            
        Returns:
            评测结果
        """
        # 调用执行器获取轨迹和状态
        execution_trace, final_state = await self.executor(case)
        
        # 使用基础评测器评估
        result = await self.base_evaluator.evaluate(
            case, execution_trace, final_state
        )
        
        # 添加运行索引信息
        result.details["run_index"] = run_index
        
        return result
    
    def _aggregate_results(
        self,
        case: EvaluationCase,
        results: List[EvaluationResult],
        consistency_metrics: ConsistencyMetrics
    ) -> EvaluationResult:
        """汇总多次执行结果
        
        Args:
            case: 评测用例
            results: 各次执行结果
            consistency_metrics: 一致性指标
            
        Returns:
            汇总的评测结果
        """
        # 计算平均分数
        valid_results = [r for r in results if r.status != EvaluationStatus.ERROR]
        
        if valid_results:
            avg_task = sum(r.task_completion_score for r in valid_results) / len(valid_results)
            avg_traj = sum(r.trajectory_quality_score for r in valid_results) / len(valid_results)
            avg_tool = sum(r.tool_accuracy_score for r in valid_results) / len(valid_results)
            avg_policy = sum(r.policy_compliance_score for r in valid_results) / len(valid_results)
            avg_latency = sum(r.latency_ms for r in valid_results) / len(valid_results)
        else:
            avg_task = avg_traj = avg_tool = avg_policy = 0.0
            avg_latency = 0
        
        # 创建汇总结果
        summary = EvaluationResult(
            case_id=case.case_id,
            status=EvaluationStatus.SUCCESS if consistency_metrics.pass_at_k else EvaluationStatus.FAILED,
            success=consistency_metrics.pass_at_k,
            task_completion_score=avg_task,
            trajectory_quality_score=avg_traj,
            tool_accuracy_score=avg_tool,
            policy_compliance_score=avg_policy,
            overall_score=consistency_metrics.success_rate,
            latency_ms=int(avg_latency),
            started_at=results[0].started_at if results else datetime.now(),
            completed_at=datetime.now(),
            details={
                "consistency": {
                    "k": consistency_metrics.k,
                    "pass_at_k": consistency_metrics.pass_at_k,
                    "pass_power_k": consistency_metrics.pass_power_k,
                    "success_rate": consistency_metrics.success_rate,
                    "individual_results": consistency_metrics.individual_results,
                },
                "individual_runs": [r.to_dict() for r in results],
                "score_variance": self._calculate_variance(
                    [r.overall_score for r in valid_results]
                ) if valid_results else 0.0,
            }
        )
        
        return summary
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """计算分数方差"""
        if not scores:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance


class StatelessConsistencyEvaluator(ConsistencyEvaluator):
    """无状态一致性评测器
    
    适用于已有多次执行结果的场景，不需要执行器函数。
    直接从提供的多个结果中计算一致性指标。
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """初始化无状态一致性评测器"""
        super().__init__(config=config)
        self.name = "StatelessConsistencyEvaluator"
    
    def evaluate_from_results(
        self,
        case_id: str,
        results: List[EvaluationResult]
    ) -> tuple[EvaluationResult, ConsistencyMetrics]:
        """从已有结果计算一致性指标
        
        Args:
            case_id: 用例ID
            results: 多次执行结果列表
            
        Returns:
            (汇总结果, 一致性指标)
        """
        k = len(results)
        individual_success = [r.success for r in results]
        
        # 计算指标
        pass_at_k = any(individual_success)
        pass_power_k = all(individual_success)
        success_rate = sum(individual_success) / k if k > 0 else 0.0
        
        consistency_metrics = ConsistencyMetrics(
            k=k,
            pass_at_k=pass_at_k,
            pass_power_k=pass_power_k,
            success_rate=success_rate,
            individual_results=individual_success
        )
        
        # 创建虚拟用例进行汇总
        dummy_case = EvaluationCase(
            case_id=case_id,
            input_messages=[""]  # placeholder
        )
        
        summary = self._aggregate_results(dummy_case, results, consistency_metrics)
        
        return summary, consistency_metrics
