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

    status: PlanStepStatus = Field(default=PlanStepStatus.PENDING)

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
    """Single-shot output for the planner node.

    Combines:
    - query_intent/entities extraction (previously done by intent_recognition node)
    - an executable multi-step plan
    """

    query_intent: QueryIntent = Field(description="意图识别+实体提取的结构化结果")
    plan: Plan = Field(description="可执行的多步骤计划")

