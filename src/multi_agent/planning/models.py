"""Planning models (Pydantic) for plan-driven multi-agent execution.

These models are stored in LangGraph state and must be JSON-serializable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from src.multi_agent.constants import AgentName, ActionName
from src.multi_agent.planning.query_intent import QueryIntent


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PolicyMethod(str, Enum):
    """How risk/policy was determined (for auditing/observability)."""

    NO_PLAN_FALLBACK = "no_plan_fallback"
    MAX_STEP_RISK = "max_step_risk"


class PlanStatus(str, Enum):
    ACTIVE = "active"
    NEEDS_USER_INPUT = "needs_user_input"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStepType(str, Enum):
    """Step types are intentionally coarse right now.

    We start with AGENT_CALL to reuse existing agent implementations and keep
    control-flow explicit in the graph.
    """

    AGENT_CALL = "agent_call"
    ASK_USER = "ask_user"
    FINISH = "finish"


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # 用于 fallback 场景：上一步已成功，跳过当前步骤


# =============================================================================
# 声明式条件执行机制 (Declarative Conditional Execution)
# =============================================================================
class StepConditionType(str, Enum):
    """步骤执行条件类型

    设计原则：
    - 条件是数据，不是代码
    - 新增条件类型只需扩展此枚举，不需修改 executor
    - 条件评估逻辑集中在一处，便于测试和审计
    """

    ALWAYS = "always"  # 无条件执行（默认）
    IF_PREVIOUS_EMPTY = "if_previous_empty"  # 仅当引用 agent 返回空结果时执行


class StepCondition(BaseModel):
    """声明式步骤执行条件

    用于在 Plan 中声明条件逻辑，避免在 executor 中硬编码业务规则。

    示例：
    - 无条件执行：condition = None 或 condition.type = "always"
    - Fallback 执行：condition.type = "if_previous_empty", reference_agent = "product_agent", result_key = "products"
      → 仅当 product_agent 的 products 为空时才执行此步骤
    """

    type: StepConditionType = Field(
        default=StepConditionType.ALWAYS, description="条件类型"
    )
    reference_agent: Optional[AgentName] = Field(
        default=None,
        description="引用的 agent（用于检查其结果）",
    )
    result_key: Optional[str] = Field(
        default=None,
        description="要检查的结果字段（如 'products'）",
    )

    @model_validator(mode="after")
    def validate_condition(self) -> "StepCondition":
        """验证条件配置的完整性"""
        if self.type == StepConditionType.IF_PREVIOUS_EMPTY:
            if self.reference_agent is None or self.result_key is None:
                raise ValueError(
                    "IF_PREVIOUS_EMPTY condition requires reference_agent and result_key"
                )
        return self


class PlanStep(BaseModel):
    """A single executable step in a plan."""

    step_id: str = Field(..., description="Stable step identifier (unique within a plan)")
    step_type: PlanStepType = Field(..., description="Type of the step")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="Risk level for the step")

    # Execution routing (for AGENT_CALL)
    selected_agent: Optional[AgentName] = Field(
        default=None, description="Target agent for AGENT_CALL steps"
    )
    next_action: Optional[ActionName] = Field(
        default=None, description="Graph action to reach selected_agent"
    )

    # Human interaction (for ASK_USER)
    ask_user_message: Optional[str] = Field(
        default=None, description="Message to ask user for missing info"
    )

    # General instruction/context for the step
    instruction: str = Field(
        default="", description="What this step should accomplish (planner-provided)"
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict, description="Structured inputs for the step"
    )
    outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Execution outputs (results, skip reason, etc.)"
    )

    status: PlanStepStatus = Field(default=PlanStepStatus.PENDING)

    # 声明式条件执行（替代 executor 中的硬编码逻辑）
    execution_condition: Optional[StepCondition] = Field(
        default=None,
        description="步骤执行条件（None 或 type=always 表示无条件执行）",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "PlanStep":
        if self.step_type == PlanStepType.AGENT_CALL:
            if self.selected_agent is None or self.next_action is None:
                raise ValueError("AGENT_CALL step requires selected_agent and next_action")
        if self.step_type == PlanStepType.ASK_USER:
            if not self.ask_user_message:
                raise ValueError("ASK_USER step requires ask_user_message")
        return self


class Plan(BaseModel):
    """A plan produced by the planner node and executed by plan_executor."""

    version: str = Field(default="1.0")
    goal: str = Field(..., description="User goal in natural language")
    status: PlanStatus = Field(default=PlanStatus.ACTIVE)

    steps: List[PlanStep] = Field(default_factory=list)
    current_step_index: int = Field(default=0, ge=0)

    # Optional fields for structured clarification / failure
    missing_information: List[str] = Field(default_factory=list)
    failure_reason: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_steps(self) -> "Plan":
        if not self.steps:
            raise ValueError("Plan.steps must not be empty")
        step_ids = [s.step_id for s in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Plan.step_id must be unique within a plan")
        return self

    def is_done(self) -> bool:
        return self.status in (PlanStatus.COMPLETED, PlanStatus.FAILED)

    def current_step(self) -> Optional[PlanStep]:
        if self.current_step_index < 0 or self.current_step_index >= len(self.steps):
            return None
        return self.steps[self.current_step_index]


class PlanningOutput(BaseModel):
    """Output for the planner node.
    
    After Intent-Planner separation refactoring:
    - QueryIntent is now handled by IntentRouter (upstream node)
    - Planner only outputs the executable plan
    """

    plan: Plan = Field(description="可执行的多步骤计划")

