"""多Agent系统 - 2025-2026 企业级最佳实践

本模块提供了一个基于LangGraph 1.x的多Agent智能体框架。

核心特性：
- Supervisor模式：智能路由和协调
- 模块化设计：易于扩展和维护
- 工具集成：统一管理MCP工具
- 状态管理：清晰的状态流转

使用示例：
    from src.multi_agent import MultiAgentGraph
    
    graph = MultiAgentGraph()
    result = graph.invoke("你的问题")
    print(result["messages"][-1].content)
"""
from src.multi_agent.graph import MultiAgentGraph
from src.multi_agent.state import MultiAgentState
from src.multi_agent.supervisor import SupervisorAgent
from src.multi_agent.agents.base_agent import BaseAgent, ToolEnabledAgent
from src.multi_agent.agents.rag_agent import RAGAgent
from src.multi_agent.agents.chat_agent import ChatAgent
from src.multi_agent.tools.tool_registry import (
    ToolRegistry,
    ToolCategory,
    ToolPermission
)
from src.multi_agent.tools.tool_config import ToolConfig, ToolConfigManager

__all__ = [
    "MultiAgentGraph",
    "MultiAgentState",
    "SupervisorAgent",
    "BaseAgent",
    "ToolEnabledAgent",
    "RAGAgent",
    "ChatAgent",
    "ToolRegistry",
    "ToolCategory",
    "ToolPermission",
    "ToolConfig",
    "ToolConfigManager",
]

