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
from typing import TYPE_CHECKING
import importlib

if TYPE_CHECKING:
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

_LAZY_IMPORTS = {
    "MultiAgentGraph": "src.multi_agent.graph",
    "MultiAgentState": "src.multi_agent.state",
    "SupervisorAgent": "src.multi_agent.supervisor",
    "BaseAgent": "src.multi_agent.agents.base_agent",
    "ToolEnabledAgent": "src.multi_agent.agents.base_agent",
    "RAGAgent": "src.multi_agent.agents.rag_agent",
    "ChatAgent": "src.multi_agent.agents.chat_agent",
    "ToolRegistry": "src.multi_agent.tools.tool_registry",
    "ToolCategory": "src.multi_agent.tools.tool_registry",
    "ToolPermission": "src.multi_agent.tools.tool_registry",
    "ToolConfig": "src.multi_agent.tools.tool_config",
    "ToolConfigManager": "src.multi_agent.tools.tool_config",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))

