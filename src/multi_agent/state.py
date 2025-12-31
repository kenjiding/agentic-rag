"""多Agent系统状态定义 - 2025-2026 企业级最佳实践

本模块定义了多Agent系统的全局状态结构，采用Pydantic模型确保类型安全和数据验证。
状态设计遵循以下原则：
1. 清晰的状态字段，便于各Agent访问和更新
2. 支持消息历史，实现对话上下文管理
3. 记录Agent执行历史，便于调试和追踪
4. 工具使用追踪，支持审计和分析
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class TaskStep(BaseModel):
    """任务步骤定义

    用于任务链中的单个步骤，支持多步骤任务编排。
    """
    step_id: str  # 步骤唯一标识
    step_type: Literal["product_search", "user_selection", "order_creation", "confirmation", "web_search", "rag_search"]  # 步骤类型（支持动态扩展）
    status: Literal["pending", "in_progress", "completed", "skipped"]  # 步骤状态
    agent_name: Optional[str] = None  # 执行该步骤的Agent名称（如果需要）
    result_data: Optional[Dict[str, Any]] = None  # 步骤执行结果数据
    metadata: Optional[Dict[str, Any]] = None  # 额外的元数据


class TaskChain(BaseModel):
    """任务链定义

    用于编排多步骤任务，如"搜索商品 → 用户选择 → 创建订单"。
    """
    chain_id: str  # 任务链唯一标识
    chain_type: str  # 任务链类型（如 "order_with_search"）
    steps: List[TaskStep]  # 任务步骤列表
    current_step_index: int  # 当前步骤索引
    created_at: str  # 创建时间（ISO格式）
    context_data: Dict[str, Any] = Field(default_factory=dict)  # 任务链上下文数据


class PendingSelection(BaseModel):
    """待选择操作定义

    用于需要用户从多个选项中选择的场景（如选择商品、地址等）。
    类似于 ConfirmationPending，但用于选择而非确认。
    """
    selection_id: str  # 选择操作唯一标识
    selection_type: str  # 选择类型（"product", "address", 等）
    options: List[Dict[str, Any]]  # 可选项列表
    display_message: str  # 展示给用户的提示消息
    metadata: Optional[Dict[str, Any]] = None  # 额外的元数据


class MultiAgentState(BaseModel):
    """多Agent系统全局状态定义

    2025-2026 最佳实践：
    - 使用Pydantic确保类型安全和数据验证
    - 支持消息历史管理
    - 记录Agent执行轨迹
    - 支持工具调用追踪

    Attributes:
        messages: 对话消息历史，包含用户输入和Agent回复
        current_agent: 当前正在执行的Agent名称
        agent_results: 各Agent的执行结果字典，key为Agent名称，value为结果
        tools_used: 已使用的工具列表，记录工具名称和调用信息
        metadata: 元数据字典，存储额外的上下文信息
        error_message: 错误信息，如果执行过程中出现错误
        iteration_count: 迭代次数，用于控制循环执行
        max_iterations: 最大迭代次数，防止无限循环
    """
    # 消息历史 - 核心对话数据
    messages: List[BaseMessage] = Field(default_factory=list)

    # Agent管理
    current_agent: Optional[str] = None  # 当前执行的Agent名称
    agent_results: Dict[str, Any] = Field(default_factory=dict)  # 各Agent的执行结果
    agent_history: List[Dict[str, Any]] = Field(default_factory=list)  # Agent执行历史记录

    # 工具管理
    tools_used: List[Dict[str, Any]] = Field(default_factory=list)  # 已使用的工具列表，包含工具名称、参数、结果等

    # 元数据和上下文
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 额外的元数据，如用户ID、会话ID等
    error_message: Optional[str] = None  # 错误信息

    # 控制流
    iteration_count: int = 0  # 当前迭代次数
    max_iterations: int = 10  # 最大迭代次数，默认10

    # 路由决策
    next_action: Optional[Literal["rag_search", "chat", "product_search", "order_management", "tool_call", "execute_task_chain", "wait_for_selection", "wait_for_confirmation", "finish"]] = None  # 下一步行动
    routing_reason: Optional[str] = None  # 路由决策的原因说明

    # 意图识别
    query_intent: Optional[Dict[str, Any]] = None  # 意图识别结果（QueryIntent转字典）
    original_question: Optional[str] = None  # 用户原始问题（用于意图识别）

    # 业务功能扩展
    confirmation_pending: Optional[Dict[str, Any]] = None  # 等待用户确认的操作

    # 多步骤任务编排
    task_chain: Optional[TaskChain] = None  # 活跃的任务链
    pending_selection: Optional[PendingSelection] = None  # 等待用户选择的操作

    # 实体信息（2025最佳实践：使用 LangGraph checkpointer 持久化）
    entities: Dict[str, Any] = Field(default_factory=dict)  # 提取的实体信息：{"user_phone": "138...", "quantity": 2, "search_keyword": "西门子", ...}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，保持向后兼容性"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiAgentState":
        """从字典创建实例"""
        return cls(**data)

