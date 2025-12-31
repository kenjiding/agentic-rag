"""Graph路由处理器 - 封装所有路由决策逻辑

将路由决策逻辑从主图类中分离，提高代码可维护性。
"""
import logging
from src.multi_agent.state import MultiAgentState

logger = logging.getLogger(__name__)


class GraphRouter:
    """图路由处理器 - 封装所有路由决策逻辑"""
    
    def __init__(self, graph_instance):
        """初始化路由处理器
        
        Args:
            graph_instance: MultiAgentGraph实例，用于访问配置
        """
        self.graph = graph_instance
    
    def route_after_supervisor(self, state: MultiAgentState) -> str:
        """Supervisor后的路由决策"""
        next_action = state.next_action or "finish"

        if next_action == "execute_task_chain":
            return "task_orchestrator"
        elif next_action == "rag_search":
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
        """Agent执行后的路由决策"""
        if state.error_message or state.iteration_count >= self.graph.max_iterations:
            return "finish"

        # 检查是否需要等待确认（优先级最高）
        if state.next_action == "wait_for_confirmation":
            logger.info("需要等待用户确认，暂停 graph 执行")
            return "wait_for_confirmation"

        # 任务链模式：继续执行任务链
        if state.task_chain and state.next_action == "execute_task_chain":
            if state.current_agent in ["product_agent", "order_agent"]:
                return "task_orchestrator"

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

    def route_after_orchestrator(self, state: MultiAgentState) -> str:
        """Task Orchestrator后的路由决策"""
        next_action = state.next_action or "finish"
        logger.info(f"[路由决策] Task Orchestrator后: next_action={next_action}, enable_business_agents={self.graph.enable_business_agents}")

        if next_action == "product_search" and self.graph.enable_business_agents:
            logger.info("[路由决策] 路由到 product_agent")
            return "product_agent"
        elif next_action == "order_management" and self.graph.enable_business_agents:
            logger.info("[路由决策] 路由到 order_agent")
            return "order_agent"
        else:
            logger.info(f"[路由决策] 任务链结束: next_action={next_action}")
            return "finish"

