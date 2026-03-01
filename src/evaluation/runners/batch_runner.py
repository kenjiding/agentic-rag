"""批量评测运行器

并行执行多个评测用例，支持一致性测试。

设计原则：
- 支持并行执行提高效率
- 支持进度追踪
- 支持一致性测试（pass@k）
"""
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING
from datetime import datetime
import asyncio
import logging
import uuid

from src.evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    EvaluationSummary,
    EvaluationStatus,
    ConsistencyMetrics,
)
from src.evaluation.config import EvaluationConfig
from src.evaluation.runners.single_run import SingleRunner
from src.evaluation.evaluators.consistency import ConsistencyEvaluator

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

# 进度回调类型
ProgressCallback = Callable[[int, int, EvaluationResult], None]


class BatchRunner:
    """批量评测运行器
    
    并行执行多个评测用例，支持：
    - 并发控制
    - 进度追踪
    - 结果汇总
    - 一致性测试
    """
    
    def __init__(
        self,
        graph: Optional["CompiledStateGraph"] = None,
        config: Optional[EvaluationConfig] = None,
        single_runner: Optional[SingleRunner] = None
    ):
        """初始化批量运行器
        
        Args:
            graph: LangGraph编译后的图
            config: 评测配置
            single_runner: 自定义单次运行器
        """
        self.config = config or EvaluationConfig.default()
        
        if single_runner:
            self.single_runner = single_runner
        else:
            self.single_runner = SingleRunner(graph, self.config)
        
        self._progress_callback: Optional[ProgressCallback] = None
    
    def set_graph(self, graph: "CompiledStateGraph") -> None:
        """设置LangGraph图"""
        self.single_runner.set_graph(graph)
    
    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """设置进度回调
        
        Args:
            callback: 回调函数 (completed, total, result) -> None
        """
        self._progress_callback = callback
    
    async def run(
        self,
        cases: List[EvaluationCase],
        name: str = "",
        description: str = ""
    ) -> EvaluationSummary:
        """批量执行评测
        
        Args:
            cases: 评测用例列表
            name: 评测名称
            description: 评测描述
            
        Returns:
            评测汇总
        """
        summary_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        # 按优先级排序
        sorted_cases = sorted(cases, key=lambda c: c.priority, reverse=True)
        
        # 并行执行
        results = await self._run_parallel(sorted_cases)
        
        # 生成汇总
        summary = EvaluationSummary.from_results(
            summary_id=summary_id,
            results=results,
            name=name or f"Batch Evaluation {summary_id}",
            description=description
        )
        
        summary.started_at = start_time
        summary.completed_at = datetime.now()
        
        return summary
    
    async def _run_parallel(
        self,
        cases: List[EvaluationCase]
    ) -> List[EvaluationResult]:
        """并行执行评测用例
        
        Args:
            cases: 用例列表
            
        Returns:
            结果列表
        """
        max_parallel = self.config.runner.max_parallel
        semaphore = asyncio.Semaphore(max_parallel)
        
        results: List[EvaluationResult] = []
        completed = 0
        total = len(cases)
        
        async def run_with_semaphore(case: EvaluationCase) -> EvaluationResult:
            nonlocal completed
            
            async with semaphore:
                result = await self.single_runner.run_with_retry(case)
                
                completed += 1
                if self._progress_callback:
                    self._progress_callback(completed, total, result)
                
                return result
        
        # 创建任务
        tasks = [run_with_semaphore(case) for case in cases]
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"评测执行异常: {r}")
                error_result = EvaluationResult(
                    case_id=cases[i].case_id,
                    status=EvaluationStatus.ERROR,
                    error_message=str(r),
                    completed_at=datetime.now()
                )
                processed_results.append(error_result)
            else:
                processed_results.append(r)
        
        return processed_results
    
    async def run_consistency_test(
        self,
        cases: List[EvaluationCase],
        k: Optional[int] = None,
        name: str = "",
        description: str = ""
    ) -> EvaluationSummary:
        """执行一致性测试
        
        对每个用例执行k次，计算pass@k和pass^k指标。
        
        Args:
            cases: 评测用例列表
            k: 每个用例执行次数
            name: 评测名称
            description: 评测描述
            
        Returns:
            包含一致性指标的评测汇总
        """
        k = k or self.config.consistency.k
        summary_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        # 为每个用例执行k次
        all_results: List[EvaluationResult] = []
        consistency_data: Dict[str, ConsistencyMetrics] = {}
        
        completed = 0
        total = len(cases) * k
        
        for case in cases:
            case_results: List[EvaluationResult] = []
            
            # 执行k次
            for i in range(k):
                result = await self.single_runner.run_with_retry(case)
                result.details["consistency_run"] = i
                case_results.append(result)
                
                completed += 1
                if self._progress_callback:
                    self._progress_callback(completed, total, result)
            
            # 计算该用例的一致性指标
            individual_success = [r.success for r in case_results]
            metrics = ConsistencyMetrics(
                k=k,
                pass_at_k=any(individual_success),
                pass_power_k=all(individual_success),
                success_rate=sum(individual_success) / k,
                individual_results=individual_success
            )
            consistency_data[case.case_id] = metrics
            
            # 创建汇总结果
            summary_result = self._aggregate_case_results(case, case_results, metrics)
            all_results.append(summary_result)
        
        # 计算整体一致性指标
        overall_pass_at_k = all(m.pass_at_k for m in consistency_data.values())
        overall_pass_power_k = all(m.pass_power_k for m in consistency_data.values())
        overall_success_rate = sum(m.success_rate for m in consistency_data.values()) / len(cases) if cases else 0.0
        
        overall_consistency = ConsistencyMetrics(
            k=k,
            pass_at_k=overall_pass_at_k,
            pass_power_k=overall_pass_power_k,
            success_rate=overall_success_rate,
            individual_results=[]
        )
        
        # 生成汇总
        summary = EvaluationSummary.from_results(
            summary_id=summary_id,
            results=all_results,
            name=name or f"Consistency Test {summary_id}",
            description=description
        )
        
        summary.consistency_metrics = overall_consistency
        summary.started_at = start_time
        summary.completed_at = datetime.now()
        
        # 添加详细的一致性数据
        summary.results_by_tag["consistency"] = {
            case_id: metrics.model_dump()
            for case_id, metrics in consistency_data.items()
        }
        
        return summary
    
    def _aggregate_case_results(
        self,
        case: EvaluationCase,
        results: List[EvaluationResult],
        metrics: ConsistencyMetrics
    ) -> EvaluationResult:
        """汇总单个用例的多次执行结果
        
        Args:
            case: 评测用例
            results: 执行结果列表
            metrics: 一致性指标
            
        Returns:
            汇总结果
        """
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
        
        return EvaluationResult(
            case_id=case.case_id,
            status=EvaluationStatus.SUCCESS if metrics.pass_at_k else EvaluationStatus.FAILED,
            success=metrics.pass_at_k,
            task_completion_score=avg_task,
            trajectory_quality_score=avg_traj,
            tool_accuracy_score=avg_tool,
            policy_compliance_score=avg_policy,
            overall_score=metrics.success_rate,
            latency_ms=int(avg_latency),
            started_at=results[0].started_at if results else datetime.now(),
            completed_at=datetime.now(),
            details={
                "consistency": metrics.model_dump(),
                "individual_runs": [r.to_dict() for r in results],
            }
        )
    
    async def run_by_tags(
        self,
        cases: List[EvaluationCase],
        tags: List[str],
        name: str = "",
        description: str = ""
    ) -> EvaluationSummary:
        """按标签筛选并执行评测
        
        Args:
            cases: 全部用例列表
            tags: 要执行的标签列表
            name: 评测名称
            description: 评测描述
            
        Returns:
            评测汇总
        """
        # 筛选用例
        filtered_cases = [
            case for case in cases
            if any(tag in case.tags for tag in tags)
        ]
        
        logger.info(f"按标签 {tags} 筛选，共 {len(filtered_cases)}/{len(cases)} 个用例")
        
        return await self.run(
            filtered_cases,
            name=name or f"Tag Filter: {', '.join(tags)}",
            description=description
        )


class EvaluationOrchestrator:
    """评测编排器
    
    高级评测工具，支持：
    - 多数据集评测
    - 分层评测
    - 综合报告生成
    """
    
    def __init__(
        self,
        graph: Optional["CompiledStateGraph"] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """初始化编排器"""
        self.config = config or EvaluationConfig.default()
        self.batch_runner = BatchRunner(graph, config)
    
    def set_graph(self, graph: "CompiledStateGraph") -> None:
        """设置LangGraph图"""
        self.batch_runner.set_graph(graph)
    
    async def run_full_evaluation(
        self,
        cases: List[EvaluationCase],
        include_consistency: bool = True,
        consistency_k: Optional[int] = None
    ) -> Dict[str, EvaluationSummary]:
        """执行完整评测
        
        包括基础评测和可选的一致性测试。
        
        Args:
            cases: 评测用例列表
            include_consistency: 是否包含一致性测试
            consistency_k: 一致性测试的k值
            
        Returns:
            评测结果字典 {"basic": summary, "consistency": summary}
        """
        results = {}
        
        # 基础评测
        logger.info(f"开始基础评测，共 {len(cases)} 个用例")
        basic_summary = await self.batch_runner.run(
            cases,
            name="Basic Evaluation",
            description="基础功能评测"
        )
        results["basic"] = basic_summary
        
        # 一致性测试
        if include_consistency:
            logger.info(f"开始一致性测试，k={consistency_k or self.config.consistency.k}")
            consistency_summary = await self.batch_runner.run_consistency_test(
                cases,
                k=consistency_k,
                name="Consistency Test",
                description="一致性评测（pass@k）"
            )
            results["consistency"] = consistency_summary
        
        return results
    
    async def run_dataset_evaluation(
        self,
        datasets: Dict[str, List[EvaluationCase]]
    ) -> Dict[str, EvaluationSummary]:
        """按数据集执行评测
        
        Args:
            datasets: 数据集字典 {name: cases}
            
        Returns:
            按数据集分组的评测结果
        """
        results = {}
        
        for name, cases in datasets.items():
            logger.info(f"评测数据集: {name}，共 {len(cases)} 个用例")
            summary = await self.batch_runner.run(
                cases,
                name=f"Dataset: {name}",
                description=f"数据集 {name} 评测"
            )
            results[name] = summary
        
        return results
