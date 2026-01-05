"""多Agent系统状态定义 - 2025-2026 企业级最佳实践

本模块定义了多Agent系统的全局状态结构，采用Pydantic模型确保类型安全和数据验证。

一步一步智能模式设计原则：
1. 每次请求都重新进行意图识别和路由决策
2. 通过 entities 字段存储上下文信息，实现多轮对话状态管理
3. Supervisor 根据 entities 状态智能路由到对应 Agent
4. 支持消息历史，实现对话上下文管理
5. 记录Agent执行历史，便于调试和追踪
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


# 对话阶段类型定义
ConversationPhase = Literal[
    "idle",              # 空闲状态，没有正在进行的任务
    "product_selecting", # 正在选择产品
    "order_creating",    # 正在创建订单（等待确认）
    "order_completed",   # 订单已完成
]


class MultiAgentState(BaseModel):
    """多Agent系统全局状态定义

    2025-2026 最佳实践：
    - 使用Pydantic确保类型安全和数据验证
    - 支持消息历史管理
    - 记录Agent执行轨迹
    - 支持工具调用追踪

    一步一步智能模式：
    - 每次请求都重新进行意图识别和路由决策
    - 通过 entities 字段存储上下文信息
    - Supervisor 根据 entities 智能路由到对应 Agent

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

    # 路由决策（一步一步智能模式：基于 entities 智能路由）
    next_action: Optional[Literal["rag_search", "chat", "product_search", "order_management", "finish"]] = None
    routing_reason: Optional[str] = None  # 路由决策的原因说明

    # 意图识别
    query_intent: Optional[Dict[str, Any]] = None  # 意图识别结果（QueryIntent转字典）
    original_question: Optional[str] = None  # 用户原始问题（用于意图识别）

    # 业务功能扩展
    confirmation_pending: Optional[Dict[str, Any]] = None  # 等待用户确认的操作

    # 实体信息（一步一步智能模式核心：通过 entities 存储上下文，实现多轮对话状态管理）
    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="提取的实体信息，用于多轮对话状态管理"
    )

    # 最近的搜索上下文（用于用户取消后重新发起请求时恢复上下文）
    last_product_search_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="最近一次产品搜索的上下文，包含 products、search_keyword、quantity 等"
    )

    # 对话阶段（用于跟踪当前对话状态，实现任务完成后的状态清理）
    conversation_phase: ConversationPhase = Field(
        default="idle",
        description="当前对话阶段：idle=空闲, product_selecting=选择产品中, order_creating=创建订单中, order_completed=订单已完成"
    )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiAgentState":
        """从字典创建实例"""
        return cls(**data)
