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
from src.multi_agent.constants import SystemNodeName

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

    def route_after_plan_executor(self, state: MultiAgentState) -> str:
        """Plan executor后的路由决策

        - 如果没有plan（或planner失败），路由到planner重新规划
        - 如果有plan但没有next_action且plan未完成，继续执行plan（可能是下一个ASK_USER step）
        - 如果有plan且有next_action，路由到对应的agent
        """
        if state.plan is None:
            # Plan被清除，需要重新规划（例如：用户回复了ASK_USER后，plan只有一个step）
            # 路由到planner以基于新的用户输入重新规划
            logger.info("Plan为None，路由到planner进行重新规划")
            return SystemNodeName.PLANNER.value
        
        # 如果有plan但没有next_action，检查是否需要继续执行plan
        if state.next_action is None:
            # Plan存在但没有next_action
            if state.plan.is_done():
                # Plan已完成
                return ActionName.FINISH.value
            
            # Plan未完成但没有next_action，这通常发生在：
            # 1. 从ASK_USER恢复后，下一个step也是ASK_USER（需要再次在plan_executor中处理）
            # 2. plan_executor处理完ASK_USER后，需要继续处理plan中的下一个step
            # 这种情况下，应该再次执行plan_executor
            logger.info(
                f"Plan未完成但没有next_action，继续执行plan_executor "
                f"(current_index={state.plan.current_step_index}, status={state.plan.status})"
            )
            return SystemNodeName.PLAN_EXECUTOR.value
        
        # 如果有plan且有next_action，路由到对应的agent
        return self.route_after_supervisor(state)

    def route_after_agent(self, state: MultiAgentState) -> str:
        """Agent执行后的路由决策（一步一步智能模式）
        
        企业级最佳实践：根据当前状态智能判断是否需要继续路由到supervisor进行下一次决策。
        符合"一步一步智能模式"设计：agent执行后由supervisor进行路由决策。
        """
        if state.error_message or state.iteration_count >= self.graph.max_iterations:
            return "finish"

        # Plan-driven mode: if plan exists and not done, continue executing plan steps
        if state.plan is not None and not state.plan.is_done():
            return SystemNodeName.POST_ACTION_VERIFIER.value

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

        return ActionName.FINISH.value
