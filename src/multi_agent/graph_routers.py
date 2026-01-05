"""Graph路由处理器 - 封装所有路由决策逻辑（一步一步智能模式）

将路由决策逻辑从主图类中分离，提高代码可维护性。

2025-2026 最佳实践：
- 每次请求都重新进行意图识别和路由决策
- 通过 entities 字段存储上下文信息
- Supervisor 根据 entities 智能路由到对应 Agent
"""
import logging
from src.multi_agent.state import MultiAgentState

logger = logging.getLogger(__name__)


class GraphRouter:
    """图路由处理器 - 封装所有路由决策逻辑（一步一步智能模式）"""

    def __init__(self, graph_instance):
        """
        初始化路由处理器

        Args:
            graph_instance: MultiAgentGraph实例，用于访问配置
        """
        self.graph = graph_instance

    def route_after_supervisor(self, state: MultiAgentState) -> str:
        """Supervisor后的路由决策（一步一步智能模式）"""
        next_action = state.next_action or "finish"

        if next_action == "rag_search":
            return "rag_agent"
        elif next_action == "chat":
            return "chat_agent"
        elif next_action == "product_search" and self.graph.enable_business_agents:
            return "product_agent"
        elif next_action == "order_management" and self.graph.enable_business_agents:
            return "order_agent"
        else:
            return "finish"

    def route_after_agent(self, state: MultiAgentState) -> str:
        """Agent执行后的路由决策（一步一步智能模式）"""
        if state.error_message or state.iteration_count >= self.graph.max_iterations:
            return "finish"

        # RAG降级：答案质量低时切换到Chat Agent
        current_agent = state.current_agent
        if current_agent == "rag_agent":
            rag_result = state.agent_results.get("rag_agent")
            if rag_result:
                answer = rag_result.get("answer", "")
                if (rag_result.get("answer_quality", 0.0) < 0.5 or
                    not answer or "无法从知识库中找到" in answer):
                    agent_names = [r.get("agent") for r in state.agent_history]
                    if "chat_agent" not in agent_names:
                        return "chat_agent"

        return "finish"
