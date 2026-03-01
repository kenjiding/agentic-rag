"""评测数据模型

本模块定义评测系统的核心数据结构，采用Pydantic模型确保类型安全和数据验证。

设计原则：
1. 与现有MultiAgentState保持一致的Pydantic风格
2. 支持序列化/反序列化（JSON兼容）
3. 提供丰富的元数据支持评测分析
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# =========================
# 枚举定义
# =========================
class EvaluationStatus(str, Enum):
    """评测状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


class OutcomeType(str, Enum):
    """预期结果类型"""
    STATE_MATCH = "state_match"  # 状态字段匹配
    RESPONSE_CONTAINS = "response_contains"  # 响应包含特定内容
    AGENT_CALLED = "agent_called"  # 特定Agent被调用
    TOOL_CALLED = "tool_called"  # 特定工具被调用
    MILESTONE_REACHED = "milestone_reached"  # 达成特定里程碑


# =========================
# 基础模型
# =========================
class Milestone(BaseModel):
    """评测里程碑 - 用于部分完成度评估
    
    基于VitaBench/ECom-Bench的milestone-based评测方法，
    支持长时程任务的部分完成度评估。
    """
    milestone_id: str = Field(..., description="里程碑唯一标识")
    name: str = Field(..., description="里程碑名称")
    description: str = Field(default="", description="里程碑描述")
    required: bool = Field(default=True, description="是否为必需里程碑")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="里程碑权重")
    
    # 验证条件
    condition_type: Literal["event_exists", "state_field", "response_pattern"] = Field(
        default="event_exists",
        description="验证条件类型"
    )
    condition_value: Dict[str, Any] = Field(
        default_factory=dict,
        description="验证条件参数"
    )


class PolicyRule(BaseModel):
    """安全策略规则 - 用于合规性评测
    
    基于ST-WebAgentBench的Completion under Policy (CuP)评测方法，
    定义Agent必须遵守的策略规则。
    """
    rule_id: str = Field(..., description="规则唯一标识")
    name: str = Field(..., description="规则名称")
    description: str = Field(default="", description="规则描述")
    severity: Literal["critical", "high", "medium", "low"] = Field(
        default="high",
        description="违规严重程度"
    )
    
    # 检测条件
    detection_type: Literal["event_pattern", "state_condition", "tool_restriction"] = Field(
        default="event_pattern",
        description="检测类型"
    )
    detection_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="检测配置"
    )
    
    def is_violated(self, event: Dict[str, Any]) -> bool:
        """检查事件是否违反此规则
        
        Args:
            event: 来自action_audit的事件
            
        Returns:
            是否违规
        """
        if self.detection_type == "event_pattern":
            # 事件模式匹配
            pattern = self.detection_config.get("pattern", {})
            for key, value in pattern.items():
                if event.get(key) == value:
                    return True
        elif self.detection_type == "tool_restriction":
            # 工具调用限制
            restricted_tools = self.detection_config.get("restricted_tools", [])
            if event.get("tool_name") in restricted_tools:
                return True
        return False


class ExpectedOutcome(BaseModel):
    """预期结果定义
    
    支持多种验证方式，用于判断任务是否成功完成。
    """
    outcome_type: OutcomeType = Field(..., description="结果类型")
    expected_value: Any = Field(..., description="期望值")
    tolerance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="容差（用于模糊匹配）"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="此结果在整体评估中的权重"
    )


# =========================
# 评测用例模型
# =========================
class EvaluationCase(BaseModel):
    """评测用例定义
    
    定义单个评测场景的完整信息，包括输入、预期输出和评测条件。
    
    设计原则：
    - 支持单轮和多轮对话场景
    - 支持多种预期结果验证方式
    - 支持里程碑式部分完成度评估
    """
    # 基础信息
    case_id: str = Field(..., description="用例唯一标识")
    name: str = Field(default="", description="用例名称")
    description: str = Field(default="", description="用例描述")
    
    # 输入定义
    input_messages: List[str] = Field(
        ...,
        min_length=1,
        description="输入消息序列（支持多轮对话）"
    )
    initial_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="初始状态（可选，用于设置预置条件）"
    )
    
    # 预期输出
    expected_intent: Optional[str] = Field(
        default=None,
        description="期望的意图识别结果"
    )
    expected_agent: Optional[str] = Field(
        default=None,
        description="期望被调用的Agent"
    )
    expected_tool_calls: List[str] = Field(
        default_factory=list,
        description="期望的工具调用序列"
    )
    expected_outcomes: List[ExpectedOutcome] = Field(
        default_factory=list,
        description="预期结果列表"
    )
    
    # 里程碑定义（用于部分完成度评估）
    milestones: List[Milestone] = Field(
        default_factory=list,
        description="评测里程碑列表"
    )
    
    # 安全策略
    applicable_policies: List[str] = Field(
        default_factory=list,
        description="适用的安全策略规则ID列表"
    )
    
    # 元数据
    tags: List[str] = Field(
        default_factory=list,
        description="标签（如multi_turn, order_flow, product_search）"
    )
    timeout_seconds: int = Field(
        default=60,
        ge=1,
        description="超时时间（秒）"
    )
    priority: int = Field(
        default=0,
        description="优先级（用于批量测试排序）"
    )
    
    @property
    def is_multi_turn(self) -> bool:
        """是否为多轮对话用例"""
        return len(self.input_messages) > 1


# =========================
# 评测结果模型
# =========================
class MilestoneResult(BaseModel):
    """里程碑评测结果"""
    milestone_id: str
    achieved: bool
    details: Optional[str] = None


class PolicyViolation(BaseModel):
    """策略违规记录"""
    rule_id: str
    rule_name: str
    severity: str
    event: Dict[str, Any]
    timestamp: Optional[datetime] = None


class EvaluationResult(BaseModel):
    """单次评测结果
    
    记录单个评测用例的完整执行结果，包括各维度评分和详细信息。
    
    设计原则：
    - 与现有RetrievalQualityEvaluator的EvaluationResult保持一致的风格
    - 提供多维度评分（任务完成、轨迹质量、安全合规）
    - 支持详细的调试和分析信息
    """
    # 基础信息
    case_id: str = Field(..., description="对应的评测用例ID")
    status: EvaluationStatus = Field(
        default=EvaluationStatus.PENDING,
        description="评测状态"
    )
    success: bool = Field(default=False, description="是否成功（综合判断）")
    
    # 核心评分（0-1）
    task_completion_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="任务完成度评分"
    )
    trajectory_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="轨迹质量评分"
    )
    tool_accuracy_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="工具调用准确率"
    )
    policy_compliance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="策略合规评分"
    )
    
    # 综合评分
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="综合评分"
    )
    
    # 性能指标
    latency_ms: int = Field(default=0, ge=0, description="执行延迟（毫秒）")
    token_usage: int = Field(default=0, ge=0, description="Token消耗")
    step_count: int = Field(default=0, ge=0, description="执行步数")
    
    # 详细结果
    milestone_results: List[MilestoneResult] = Field(
        default_factory=list,
        description="里程碑达成情况"
    )
    policy_violations: List[PolicyViolation] = Field(
        default_factory=list,
        description="策略违规记录"
    )
    
    # 实际输出
    actual_intent: Optional[str] = Field(default=None, description="实际意图识别结果")
    actual_agent: Optional[str] = Field(default=None, description="实际调用的Agent")
    actual_tool_calls: List[str] = Field(
        default_factory=list,
        description="实际工具调用序列"
    )
    final_response: Optional[str] = Field(default=None, description="最终响应内容")
    
    # 执行轨迹
    execution_trace: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="执行轨迹（来自action_audit）"
    )
    
    # 错误信息
    error_message: Optional[str] = Field(default=None, description="错误信息")
    
    # 时间戳
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    
    # 调试信息
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="详细调试信息"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()
    
    @property
    def duration_ms(self) -> int:
        """计算执行时长"""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds() * 1000)
        return self.latency_ms


# =========================
# 评测汇总模型
# =========================
class ConsistencyMetrics(BaseModel):
    """一致性指标（基于τ-bench）"""
    k: int = Field(..., description="测试次数")
    pass_at_k: bool = Field(
        default=False,
        description="k次中至少成功1次"
    )
    pass_power_k: bool = Field(
        default=False,
        description="k次全部成功"
    )
    success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="成功率"
    )
    individual_results: List[bool] = Field(
        default_factory=list,
        description="各次执行结果"
    )


class EvaluationSummary(BaseModel):
    """评测汇总报告
    
    汇总多个评测用例的结果，提供整体评估视图。
    """
    # 基础信息
    summary_id: str = Field(..., description="汇总报告ID")
    name: str = Field(default="", description="评测名称")
    description: str = Field(default="", description="评测描述")
    
    # 统计信息
    total_cases: int = Field(default=0, ge=0, description="总用例数")
    passed_cases: int = Field(default=0, ge=0, description="通过用例数")
    failed_cases: int = Field(default=0, ge=0, description="失败用例数")
    error_cases: int = Field(default=0, ge=0, description="错误用例数")
    
    # 综合评分
    overall_success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="总体成功率"
    )
    avg_task_completion: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="平均任务完成度"
    )
    avg_trajectory_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="平均轨迹质量"
    )
    avg_tool_accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="平均工具准确率"
    )
    avg_policy_compliance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="平均策略合规率"
    )
    
    # 一致性指标
    consistency_metrics: Optional[ConsistencyMetrics] = Field(
        default=None,
        description="一致性评测指标"
    )
    
    # 性能统计
    avg_latency_ms: float = Field(default=0.0, ge=0.0, description="平均延迟")
    total_token_usage: int = Field(default=0, ge=0, description="总Token消耗")
    
    # 策略违规统计
    total_violations: int = Field(default=0, ge=0, description="总违规次数")
    violations_by_rule: Dict[str, int] = Field(
        default_factory=dict,
        description="按规则统计的违规次数"
    )
    
    # 按标签分组的结果
    results_by_tag: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="按标签分组的评测结果"
    )
    
    # 详细结果
    results: List[EvaluationResult] = Field(
        default_factory=list,
        description="所有评测结果"
    )
    
    # 时间戳
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    
    @classmethod
    def from_results(
        cls,
        summary_id: str,
        results: List[EvaluationResult],
        name: str = "",
        description: str = ""
    ) -> "EvaluationSummary":
        """从评测结果列表创建汇总
        
        Args:
            summary_id: 汇总ID
            results: 评测结果列表
            name: 评测名称
            description: 评测描述
            
        Returns:
            评测汇总对象
        """
        if not results:
            return cls(summary_id=summary_id, name=name, description=description)
        
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if r.status == EvaluationStatus.FAILED)
        errors = sum(1 for r in results if r.status == EvaluationStatus.ERROR)
        
        # 计算平均分数
        avg_task = sum(r.task_completion_score for r in results) / total
        avg_traj = sum(r.trajectory_quality_score for r in results) / total
        avg_tool = sum(r.tool_accuracy_score for r in results) / total
        avg_policy = sum(r.policy_compliance_score for r in results) / total
        avg_latency = sum(r.latency_ms for r in results) / total
        
        # 统计违规
        total_violations = sum(len(r.policy_violations) for r in results)
        violations_by_rule: Dict[str, int] = {}
        for r in results:
            for v in r.policy_violations:
                violations_by_rule[v.rule_id] = violations_by_rule.get(v.rule_id, 0) + 1
        
        return cls(
            summary_id=summary_id,
            name=name,
            description=description,
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            error_cases=errors,
            overall_success_rate=passed / total if total > 0 else 0.0,
            avg_task_completion=avg_task,
            avg_trajectory_quality=avg_traj,
            avg_tool_accuracy=avg_tool,
            avg_policy_compliance=avg_policy,
            avg_latency_ms=avg_latency,
            total_token_usage=sum(r.token_usage for r in results),
            total_violations=total_violations,
            violations_by_rule=violations_by_rule,
            results=results,
        )
