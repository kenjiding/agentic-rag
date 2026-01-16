"""Graph路由处理器 - 封装所有路由决策逻辑（一步一步智能模式）

将路由决策逻辑从主图类中分离，提高代码可维护性。

2025-2026 最佳实践：
- 每次请求都重新进行意图识别和路由决策
- 通过 entities 字段存储上下文信息
- Supervisor 根据 entities 智能路由到对应 Agent
"""
import logging
from src.multi_agent.state import MultiAgentState
from src.multi_agent.constants import ActionName, AgentName

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
        next_action = state.next_action or ActionName.FINISH

        if next_action == ActionName.RAG_SEARCH:
            return AgentName.RAG_AGENT.value
        elif next_action == ActionName.CHAT:
            return AgentName.CHAT_AGENT.value
        elif next_action == ActionName.PRODUCT_SEARCH and self.graph.enable_business_agents:
            return AgentName.PRODUCT_AGENT.value
        elif next_action == ActionName.ORDER_MANAGEMENT and self.graph.enable_business_agents:
            return AgentName.ORDER_AGENT.value
        elif next_action == ActionName.CONSULTATION and self.graph.enable_business_agents:
            return AgentName.CONSULTATION_AGENT.value
        else:
            return ActionName.FINISH.value

    def route_after_agent(self, state: MultiAgentState) -> str:
        """Agent执行后的路由决策（一步一步智能模式）
        
        企业级最佳实践：根据当前状态智能判断是否需要继续路由到supervisor进行下一次决策。
        符合"一步一步智能模式"设计：agent执行后由supervisor进行路由决策。
        """
        if state.error_message or state.iteration_count >= self.graph.max_iterations:
            return "finish"

        current_agent = state.current_agent
        
        # RAG降级：答案质量低时切换到Chat Agent
        if current_agent == AgentName.RAG_AGENT:
            rag_result = state.agent_results.get(AgentName.RAG_AGENT.value)
            if rag_result:
                answer = rag_result.get("answer", "")
                if (rag_result.get("answer_quality", 0.0) < 0.5 or
                    not answer or "无法从知识库中找到" in answer):
                    agent_names = [r.get("agent") for r in state.agent_history]
                    if AgentName.CHAT_AGENT.value not in agent_names:
                        return AgentName.CHAT_AGENT.value
        
        # Product Agent：对比场景需要继续路由到consultation_agent
        if current_agent == AgentName.PRODUCT_AGENT:
            # 检查是否为对比场景
            # 方法1：从query_intent判断（意图识别阶段已通过LLM判断）
            query_intent = state.query_intent
            entities = state.entities
            product_ids = entities.get("product_ids")
            has_product_ids = bool(product_ids) and isinstance(product_ids, list) and len(product_ids) >= 2
            
            # 如果是对比场景且已提取到product_ids，路由回supervisor进行下一次决策
            if query_intent:
                intent_type = query_intent.get("intent_type")
                if intent_type == "comparison" and has_product_ids:
                    # 对比场景且已提取到product_ids，路由回supervisor
                    logger.info(f"检测到product_agent对比场景，已提取product_ids={product_ids}，路由回supervisor进行下一次决策")
                    return "supervisor"
        
        return ActionName.FINISH.value
