"""LangSmith集成

与LangChain官方评测平台LangSmith集成，支持：
- 数据集管理
- 在线评测
- 实验追踪

设计原则：
- 遵循LangSmith官方最佳实践
- 与本地评测器无缝集成
- 支持生产环境监控
"""
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING
import logging
import os

from src.evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    EvaluationSummary,
)
from src.evaluation.config import EvaluationConfig
from src.evaluation.evaluators.task_completion import TaskCompletionEvaluator
from src.evaluation.evaluators.trajectory import TrajectoryEvaluator
from src.evaluation.evaluators.safety import SafetyEvaluator

if TYPE_CHECKING:
    from langsmith import Client
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class LangSmithIntegration:
    """LangSmith集成类
    
    提供与LangSmith平台的集成能力，支持：
    - 创建和管理评测数据集
    - 运行在线评测
    - 追踪实验结果
    """
    
    def __init__(
        self,
        project_name: Optional[str] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """初始化LangSmith集成
        
        Args:
            project_name: LangSmith项目名称
            config: 评测配置
        """
        self.config = config or EvaluationConfig.default()
        self.project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "multi-agent-evaluation")
        self._client: Optional["Client"] = None
    
    @property
    def client(self) -> "Client":
        """获取LangSmith客户端（懒加载）"""
        if self._client is None:
            try:
                from langsmith import Client
                self._client = Client()
                logger.info("LangSmith客户端初始化成功")
            except ImportError:
                raise ImportError(
                    "请安装langsmith: pip install langsmith"
                )
            except Exception as e:
                raise RuntimeError(
                    f"LangSmith初始化失败，请检查环境变量 LANGCHAIN_API_KEY: {e}"
                )
        return self._client
    
    def create_dataset(
        self,
        name: str,
        cases: List[EvaluationCase],
        description: str = ""
    ) -> str:
        """创建LangSmith数据集
        
        Args:
            name: 数据集名称
            cases: 评测用例列表
            description: 数据集描述
            
        Returns:
            数据集ID
        """
        # 创建数据集
        dataset = self.client.create_dataset(
            dataset_name=name,
            description=description or f"评测数据集: {name}"
        )
        
        # 添加用例
        for case in cases:
            self.client.create_example(
                inputs={
                    "messages": case.input_messages,
                    "initial_state": case.initial_state,
                },
                outputs={
                    "expected_intent": case.expected_intent,
                    "expected_agent": case.expected_agent,
                    "expected_tool_calls": case.expected_tool_calls,
                },
                metadata={
                    "case_id": case.case_id,
                    "tags": case.tags,
                    "timeout": case.timeout_seconds,
                },
                dataset_id=dataset.id,
            )
        
        logger.info(f"创建LangSmith数据集: {name}，共 {len(cases)} 个用例")
        return str(dataset.id)
    
    async def run_evaluation(
        self,
        graph: "CompiledStateGraph",
        dataset_name: str,
        experiment_prefix: str = "eval"
    ) -> EvaluationSummary:
        """运行LangSmith评测
        
        Args:
            graph: LangGraph图
            dataset_name: 数据集名称
            experiment_prefix: 实验前缀
            
        Returns:
            评测汇总
        """
        from langsmith.evaluation import evaluate
        
        # 定义评测函数
        async def target_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """评测目标函数"""
            from langchain_core.messages import HumanMessage
            
            messages = [HumanMessage(content=msg) for msg in inputs.get("messages", [])]
            initial_state = inputs.get("initial_state", {})
            initial_state["messages"] = messages
            
            result = await graph.ainvoke(initial_state)
            
            return {
                "output": result.get("content", ""),
                "action_audit": result.get("action_audit", []),
                "final_state": result,
            }
        
        # 创建自定义评测器
        evaluators = [
            self._create_task_completion_evaluator(),
            self._create_trajectory_evaluator(),
            self._create_safety_evaluator(),
        ]
        
        # 运行评测
        results = evaluate(
            target_func,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
        )
        
        # 转换为本地格式
        return self._convert_to_summary(results, experiment_prefix)
    
    def _create_task_completion_evaluator(self) -> Callable:
        """创建任务完成评测器（LangSmith格式）"""
        task_evaluator = TaskCompletionEvaluator(self.config)
        
        def evaluate_task_completion(run, example) -> Dict[str, Any]:
            """评测任务完成度"""
            try:
                # 从run中提取信息
                outputs = run.outputs or {}
                action_audit = outputs.get("action_audit", [])
                
                # 从example中提取期望值
                expected = example.outputs or {}
                
                # 简化评测（同步版本）
                actual_intent = None
                actual_agent = None
                
                for event in action_audit:
                    if event.get("event") == "intent_classified":
                        actual_intent = event.get("intent_type")
                    if event.get("node") and "agent" in event.get("node", "").lower():
                        actual_agent = event.get("node")
                
                # 计算匹配度
                intent_match = actual_intent == expected.get("expected_intent") if expected.get("expected_intent") else True
                agent_match = expected.get("expected_agent", "").lower() in (actual_agent or "").lower() if expected.get("expected_agent") else True
                
                score = 0.0
                if intent_match:
                    score += 0.5
                if agent_match:
                    score += 0.5
                
                return {
                    "key": "task_completion",
                    "score": score,
                    "comment": f"Intent: {actual_intent}, Agent: {actual_agent}"
                }
            except Exception as e:
                return {
                    "key": "task_completion",
                    "score": 0.0,
                    "comment": f"Error: {str(e)}"
                }
        
        return evaluate_task_completion
    
    def _create_trajectory_evaluator(self) -> Callable:
        """创建轨迹评测器（LangSmith格式）"""
        def evaluate_trajectory(run, example) -> Dict[str, Any]:
            """评测轨迹质量"""
            try:
                outputs = run.outputs or {}
                action_audit = outputs.get("action_audit", [])
                
                expected = example.outputs or {}
                expected_tools = expected.get("expected_tool_calls", [])
                
                # 提取实际工具调用
                actual_tools = []
                for event in action_audit:
                    tool = event.get("tool_name") or event.get("tool")
                    if tool:
                        actual_tools.append(tool.lower())
                
                # 计算匹配度
                if not expected_tools:
                    score = 1.0
                else:
                    expected_set = set(t.lower() for t in expected_tools)
                    actual_set = set(actual_tools)
                    
                    if not actual_set:
                        score = 0.0
                    else:
                        intersection = expected_set & actual_set
                        precision = len(intersection) / len(actual_set)
                        recall = len(intersection) / len(expected_set)
                        score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                return {
                    "key": "trajectory_quality",
                    "score": score,
                    "comment": f"Tools: {actual_tools}"
                }
            except Exception as e:
                return {
                    "key": "trajectory_quality",
                    "score": 0.0,
                    "comment": f"Error: {str(e)}"
                }
        
        return evaluate_trajectory
    
    def _create_safety_evaluator(self) -> Callable:
        """创建安全评测器（LangSmith格式）"""
        def evaluate_safety(run, example) -> Dict[str, Any]:
            """评测安全合规性"""
            try:
                outputs = run.outputs or {}
                action_audit = outputs.get("action_audit", [])
                
                # 检查违规事件
                violations = []
                for event in action_audit:
                    event_type = event.get("event", "")
                    if "error" in event_type.lower() or "fail" in event_type.lower():
                        violations.append(event_type)
                    if event.get("confirmed") is False and "order" in event_type.lower():
                        violations.append("unauthorized_order")
                
                score = 1.0 if not violations else max(0.0, 1.0 - len(violations) * 0.2)
                
                return {
                    "key": "safety_compliance",
                    "score": score,
                    "comment": f"Violations: {violations}" if violations else "No violations"
                }
            except Exception as e:
                return {
                    "key": "safety_compliance",
                    "score": 0.0,
                    "comment": f"Error: {str(e)}"
                }
        
        return evaluate_safety
    
    def _convert_to_summary(
        self,
        langsmith_results: Any,
        experiment_name: str
    ) -> EvaluationSummary:
        """将LangSmith结果转换为本地汇总格式"""
        # LangSmith返回的是ExperimentResults对象
        # 这里提供基本的转换逻辑
        
        results: List[EvaluationResult] = []
        
        # 遍历结果（具体结构取决于LangSmith版本）
        try:
            for row in langsmith_results:
                result = EvaluationResult(
                    case_id=str(row.example.id) if hasattr(row, "example") else "unknown",
                    success=all(
                        e.score >= 0.5 for e in row.evaluations
                    ) if hasattr(row, "evaluations") else False,
                )
                results.append(result)
        except Exception as e:
            logger.warning(f"转换LangSmith结果时出错: {e}")
        
        return EvaluationSummary.from_results(
            summary_id=experiment_name,
            results=results,
            name=f"LangSmith Evaluation: {experiment_name}",
            description="从LangSmith评测结果转换"
        )
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """列出所有数据集"""
        datasets = self.client.list_datasets()
        return [
            {
                "id": str(d.id),
                "name": d.name,
                "description": d.description,
            }
            for d in datasets
        ]
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """删除数据集"""
        try:
            self.client.delete_dataset(dataset_name=dataset_name)
            logger.info(f"删除数据集: {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"删除数据集失败: {e}")
            return False
