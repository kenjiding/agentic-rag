"""多Agent系统状态定义 - 2025-2026 企业级最佳实践

本模块定义了多Agent系统的全局状态结构，采用TypedDict确保类型安全。
状态设计遵循以下原则：
1. 清晰的状态字段，便于各Agent访问和更新
2. 支持消息历史，实现对话上下文管理
3. 记录Agent执行历史，便于调试和追踪
4. 工具使用追踪，支持审计和分析
"""
from typing import TypedDict, List, Optional, Dict, Any, Literal
from langchain_core.messages import BaseMessage


class MultiAgentState(TypedDict):
    """多Agent系统全局状态定义
    
    2025-2026 最佳实践：
    - 使用TypedDict确保类型安全
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
    messages: List[BaseMessage]
    
    # Agent管理
    current_agent: Optional[str]  # 当前执行的Agent名称
    agent_results: Dict[str, Any]  # 各Agent的执行结果
    agent_history: List[Dict[str, Any]]  # Agent执行历史记录
    
    # 工具管理
    tools_used: List[Dict[str, Any]]  # 已使用的工具列表，包含工具名称、参数、结果等
    
    # 元数据和上下文
    metadata: Dict[str, Any]  # 额外的元数据，如用户ID、会话ID等
    error_message: Optional[str]  # 错误信息
    
    # 控制流
    iteration_count: int  # 当前迭代次数
    max_iterations: int  # 最大迭代次数，默认10
    
    # 路由决策
    next_action: Optional[Literal["rag_search", "chat", "tool_call", "finish"]]  # 下一步行动
    routing_reason: Optional[str]  # 路由决策的原因说明

    # 意图识别
    query_intent: Optional[Dict[str, Any]]  # 意图识别结果（QueryIntent转字典）
    original_question: Optional[str]  # 用户原始问题（用于意图识别）

