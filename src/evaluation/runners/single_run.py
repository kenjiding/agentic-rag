"""单次评测运行器

执行单个评测用例，返回评测结果。

设计原则：
- 与LangGraph图执行集成
- 支持超时控制
- 提供详细的执行追踪
"""
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime
import asyncio
import logging
import time

from src.evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    EvaluationStatus,
)
from src.evaluation.config import EvaluationConfig
from src.evaluation.evaluators.base import BaseAgentEvaluator, CompositeEvaluator
from src.evaluation.evaluators.task_completion import TaskCompletionEvaluator
from src.evaluation.evaluators.trajectory import TrajectoryEvaluator
from src.evaluation.evaluators.safety import SafetyEvaluator

if TYPE_CHECKING:
    from src.multi_agent.state import MultiAgentState
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class SingleRunner:
    """单次评测运行器
    
    执行单个评测用例，支持：
    - LangGraph图执行
    - 多维度评测
    - 超时控制
    - 重试机制
    """
    
    def __init__(
        self,
        graph: Optional["CompiledStateGraph"] = None,
        config: Optional[EvaluationConfig] = None,
        evaluators: Optional[List[BaseAgentEvaluator]] = None
    ):
        """初始化单次运行器
        
        Args:
            graph: LangGraph编译后的图
            config: 评测配置
            evaluators: 自定义评测器列表（默认使用全部评测器）
        """
        self.graph = graph
        self.config = config or EvaluationConfig.default()
        
        # 初始化评测器
        if evaluators:
            self.evaluator = CompositeEvaluator(evaluators, self.config)
        else:
            # 默认使用全部评测器
            self.evaluator = CompositeEvaluator([
                TaskCompletionEvaluator(self.config),
                TrajectoryEvaluator(self.config),
                SafetyEvaluator(config=self.config),
            ], self.config)
    
    def set_graph(self, graph: "CompiledStateGraph") -> None:
        """设置LangGraph图"""
        self.graph = graph
    
    async def run(
        self,
        case: EvaluationCase,
        thread_id: Optional[str] = None
    ) -> EvaluationResult:
        """执行单次评测
        
        Args:
            case: 评测用例
            thread_id: 线程ID（用于多轮对话）
            
        Returns:
            评测结果
        """
        if not self.graph:
            raise ValueError("未设置LangGraph图，请先调用 set_graph")
        
        start_time = time.time()
        result = EvaluationResult(
            case_id=case.case_id,
            status=EvaluationStatus.RUNNING,
            started_at=datetime.now()
        )
        
        try:
            # 执行图
            execution_trace, final_state = await self._execute_graph(
                case, thread_id
            )
            
            # 计算延迟
            result.latency_ms = int((time.time() - start_time) * 1000)
            
            # 执行评测
            eval_result = await self.evaluator.evaluate(
                case, execution_trace, final_state
            )
            
            # 合并结果
            result = self._merge_results(result, eval_result)
            result.execution_trace = execution_trace
            result.final_response = final_state.content if final_state else None
            
        except asyncio.TimeoutError:
            result.status = EvaluationStatus.ERROR
            result.error_message = f"执行超时 ({case.timeout_seconds}s)"
            result.latency_ms = int((time.time() - start_time) * 1000)
            
        except Exception as e:
            logger.error(f"评测执行失败: {e}", exc_info=True)
            result.status = EvaluationStatus.ERROR
            result.error_message = str(e)
            result.latency_ms = int((time.time() - start_time) * 1000)
        
        result.completed_at = datetime.now()
        return result
    
    async def _execute_graph(
        self,
        case: EvaluationCase,
        thread_id: Optional[str]
    ) -> tuple[List[Dict[str, Any]], "MultiAgentState"]:
        """执行LangGraph图
        
        Args:
            case: 评测用例
            thread_id: 线程ID
            
        Returns:
            (执行轨迹, 最终状态)
        """
        from langchain_core.messages import HumanMessage
        import uuid
        
        # 准备初始状态
        initial_state = case.initial_state or {}
        
        # 构建消息 - 只用第一条消息（单轮评测）
        first_message = case.input_messages[0] if case.input_messages else ""
        initial_state["messages"] = [HumanMessage(content=first_message)]
        
        # 准备配置 - 使用唯一的thread_id
        thread_id = thread_id or f"eval_{case.case_id}_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 执行图（带超时）
        execution_trace: List[Dict[str, Any]] = []
        final_state = None
        
        async def execute_with_trace():
            nonlocal execution_trace, final_state
            
            async for event in self.graph.astream(
                initial_state,
                config=config,
                stream_mode="values"
            ):
                # 收集执行轨迹
                if isinstance(event, dict):
                    if "action_audit" in event:
                        execution_trace = event.get("action_audit", [])
                    final_state = event
                elif hasattr(event, "action_audit"):
                    execution_trace = event.action_audit or []
                    final_state = event
                else:
                    final_state = event
            
            return final_state
        
        # 执行带超时
        final_state = await asyncio.wait_for(
            execute_with_trace(),
            timeout=case.timeout_seconds
        )
        
        # 如果轨迹为空，从最终状态提取
        if not execution_trace and final_state:
            if isinstance(final_state, dict):
                execution_trace = final_state.get("action_audit", [])
            elif hasattr(final_state, "action_audit"):
                execution_trace = final_state.action_audit or []
        
        # 转换final_state为MultiAgentState（如果是字典）
        if isinstance(final_state, dict):
            from src.multi_agent.state import MultiAgentState
            try:
                final_state = MultiAgentState(**final_state)
            except Exception:
                # 如果转换失败，创建一个基本状态
                final_state = MultiAgentState(
                    content=final_state.get("content", ""),
                    messages=final_state.get("messages", []),
                    action_audit=execution_trace
                )
        
        return execution_trace, final_state
    
    def _merge_results(
        self,
        base: EvaluationResult,
        eval_result: EvaluationResult
    ) -> EvaluationResult:
        """合并评测结果
        
        Args:
            base: 基础结果（包含运行时信息）
            eval_result: 评测结果（包含评分）
            
        Returns:
            合并后的结果
        """
        base.status = eval_result.status
        base.success = eval_result.success
        base.task_completion_score = eval_result.task_completion_score
        base.trajectory_quality_score = eval_result.trajectory_quality_score
        base.tool_accuracy_score = eval_result.tool_accuracy_score
        base.policy_compliance_score = eval_result.policy_compliance_score
        base.overall_score = eval_result.overall_score
        base.milestone_results = eval_result.milestone_results
        base.policy_violations = eval_result.policy_violations
        base.actual_intent = eval_result.actual_intent
        base.actual_agent = eval_result.actual_agent
        base.actual_tool_calls = eval_result.actual_tool_calls
        base.step_count = eval_result.step_count
        base.details = eval_result.details
        
        return base
    
    async def run_with_retry(
        self,
        case: EvaluationCase,
        thread_id: Optional[str] = None
    ) -> EvaluationResult:
        """执行评测（带重试）
        
        Args:
            case: 评测用例
            thread_id: 线程ID
            
        Returns:
            评测结果
        """
        max_retries = self.config.runner.max_retries if self.config.runner.retry_on_error else 0
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await self.run(case, thread_id)
                
                # 如果成功或者不是错误状态，返回结果
                if result.status != EvaluationStatus.ERROR:
                    return result
                
                # 记录错误以便重试
                last_error = result.error_message
                logger.warning(f"评测失败 (attempt {attempt + 1}/{max_retries + 1}): {last_error}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"评测异常 (attempt {attempt + 1}/{max_retries + 1}): {e}")
        
        # 所有重试都失败
        return EvaluationResult(
            case_id=case.case_id,
            status=EvaluationStatus.ERROR,
            error_message=f"重试{max_retries + 1}次后仍失败: {last_error}",
            completed_at=datetime.now()
        )


class MockGraphRunner(SingleRunner):
    """模拟图运行器
    
    用于测试评测系统，不需要真实的LangGraph图。
    """
    
    def __init__(
        self,
        mock_responses: Optional[Dict[str, tuple[List[Dict], Dict]]] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """初始化模拟运行器
        
        Args:
            mock_responses: 模拟响应 {case_id: (execution_trace, final_state_dict)}
            config: 评测配置
        """
        # 不调用super().__init__，直接初始化需要的属性
        self.graph = None  # Mock不需要真实图
        self.config = config or EvaluationConfig.default()
        self.mock_responses = mock_responses or {}
        
        # 初始化评测器
        self.evaluator = CompositeEvaluator([
            TaskCompletionEvaluator(self.config),
            TrajectoryEvaluator(self.config),
            SafetyEvaluator(config=self.config),
        ], self.config)
    
    def add_mock_response(
        self,
        case_id: str,
        execution_trace: List[Dict[str, Any]],
        final_state_dict: Dict[str, Any]
    ) -> None:
        """添加模拟响应"""
        self.mock_responses[case_id] = (execution_trace, final_state_dict)
    
    async def run(
        self,
        case: EvaluationCase,
        thread_id: Optional[str] = None
    ) -> EvaluationResult:
        """执行模拟评测（覆盖父类方法）"""
        start_time = time.time()
        result = EvaluationResult(
            case_id=case.case_id,
            status=EvaluationStatus.RUNNING,
            started_at=datetime.now()
        )
        
        try:
            # 获取模拟响应
            execution_trace, final_state = await self._execute_graph(case, thread_id)
            
            # 计算延迟
            result.latency_ms = int((time.time() - start_time) * 1000)
            
            # 执行评测
            eval_result = await self.evaluator.evaluate(
                case, execution_trace, final_state
            )
            
            # 合并结果
            result = self._merge_results(result, eval_result)
            result.execution_trace = execution_trace
            result.final_response = final_state.content if final_state else None
            
        except Exception as e:
            logger.error(f"Mock评测执行失败: {e}", exc_info=True)
            result.status = EvaluationStatus.ERROR
            result.error_message = str(e)
            result.latency_ms = int((time.time() - start_time) * 1000)
        
        result.completed_at = datetime.now()
        return result
    
    async def _execute_graph(
        self,
        case: EvaluationCase,
        thread_id: Optional[str]
    ) -> tuple[List[Dict[str, Any]], "MultiAgentState"]:
        """模拟图执行"""
        from src.multi_agent.state import MultiAgentState
        
        if case.case_id in self.mock_responses:
            trace, state_dict = self.mock_responses[case.case_id]
            state = MultiAgentState(**state_dict)
            return trace, state
        
        # 默认返回空响应
        return [], MultiAgentState()
