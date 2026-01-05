"""Graph节点处理器 - 封装所有节点执行逻辑（一步一步智能模式）

将节点执行逻辑从主图类中分离，提高代码可维护性和可测试性。

2025-2026 最佳实践：
- 每次请求都重新进行意图识别和路由决策
- 通过 entities 字段存储上下文信息
- Supervisor 根据 entities 智能路由到对应 Agent
"""
import logging
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
import json

from src.multi_agent.state import MultiAgentState

logger = logging.getLogger(__name__)


class GraphNodeHandler:
    """图节点处理器 - 封装所有节点执行逻辑（一步一步智能模式）"""

    def __init__(self, graph_instance):
        """
        初始化节点处理器

        Args:
            graph_instance: MultiAgentGraph实例，用于访问agents和其他资源
        """
        self.graph = graph_instance

    async def intent_recognition_node(self, state: MultiAgentState) -> MultiAgentState:
        """意图识别节点 - 分析用户查询意图并提取实体"""
        try:
            # 从messages中获取最后一条HumanMessage
            question = None
            for msg in reversed(state.messages):
                if isinstance(msg, HumanMessage):
                    question = msg.content
                    break

            if not question or not isinstance(question, str):
                question = state.original_question

            if not question:
                logger.warning("未找到用户问题，跳过意图识别")
                return {"query_intent": None, "original_question": question}

            logger.info(f"🎯【意图识别+实体提取】分析查询: {question}")

            # 执行意图识别（Joint Intent Detection and Slot Filling）
            if not self.graph.intent_classifier:
                return {"query_intent": None, "original_question": question}

            # 使用异步方法提高性能
            intent = await self.graph.intent_classifier.aclassify(question)

            # 提取实体
            existing_entities = state.entities
            entities = {**existing_entities}

            if intent.entities:
                entities_dict = intent.entities.model_dump(exclude_none=True)
                for key, value in entities_dict.items():
                    if value is not None:
                        entities[key] = value

            logger.info(f"📦【实体提取】实体: {entities}")

            intent_dict = intent.model_dump()
            logger.info(f"🎯【意图识别】类型: {intent.intent_type}, 复杂度: {intent.complexity}")

            return {
                "query_intent": intent_dict,
                "original_question": question,
                "entities": entities
            }

        except Exception as e:
            logger.error(f"意图识别节点执行错误: {str(e)}", exc_info=True)
            return {"query_intent": None, "error_message": f"意图识别错误: {str(e)}"}

    async def supervisor_node(
        self, state: MultiAgentState, config: Optional[RunnableConfig] = None
    ) -> MultiAgentState:
        """Supervisor节点 - 路由决策（一步一步智能模式）"""
        try:
            iteration_count = state.iteration_count
            if iteration_count >= self.graph.max_iterations:
                logger.warning(f"达到最大迭代次数 {self.graph.max_iterations}，结束执行")
                return {
                    "next_action": "finish",
                    "routing_reason": f"达到最大迭代次数 {self.graph.max_iterations}"
                }

            routing_decision = await self.graph.supervisor.route(state)

            updated_state = {
                "next_action": routing_decision["next_action"],
                "current_agent": routing_decision.get("selected_agent"),
                "routing_reason": routing_decision.get("routing_reason", ""),
                "iteration_count": iteration_count + 1
            }

            logger.info(f"Supervisor决策: {routing_decision}")
            return updated_state

        except Exception as e:
            logger.error(f"Supervisor节点执行错误: {str(e)}", exc_info=True)
            return {
                "next_action": "finish",
                "error_message": f"Supervisor错误: {str(e)}",
                "routing_reason": f"执行错误: {str(e)}"
            }

    async def _execute_agent_node(
        self, state: MultiAgentState, agent_name: str
    ) -> MultiAgentState:
        """通用Agent节点执行逻辑"""
        try:
            agent = getattr(self.graph, agent_name, None)
            if not agent:
                logger.error(f"{agent_name} 未找到")
                return {
                    "next_action": "finish",
                    "error_message": f"{agent_name} 未找到"
                }

            result = await agent.execute(state)

            updated_state = {
                "messages": state.messages + result.get("messages", []),
                "agent_results": {
                    **state.agent_results,
                    agent_name: result.get("result")
                },
                "agent_history": state.agent_history + [{
                    "agent": agent_name,
                    "result": result.get("result"),
                    "metadata": result.get("metadata", {})
                }]
            }

            logger.info(f"{agent_name} 执行完成")
            return updated_state

        except Exception as e:
            logger.error(f"{agent_name} 节点执行错误: {str(e)}", exc_info=True)
            return {
                "next_action": "finish",
                "error_message": f"{agent_name} 错误: {str(e)}"
            }

    async def rag_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """RAG Agent节点"""
        return await self._execute_agent_node(state, "rag_agent")

    async def chat_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Chat Agent节点"""
        return await self._execute_agent_node(state, "chat_agent")

    async def product_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Product Agent节点（商品搜索）"""
        try:
            product_agent = getattr(self.graph, "product_agent", None)
            if not product_agent:
                logger.error("Product Agent未找到")
                return {"next_action": "finish", "error_message": "Product Agent未找到"}

            result = await product_agent.execute(state)
            updated_state = {
                "messages": result.get("messages", state.messages),
                "current_agent": "product_agent",
                "tools_used": result.get("tools_used", state.tools_used),
                # 设置 conversation_phase 为 product_selecting，表示用户正在选择产品
                "conversation_phase": "product_selecting"
            }

            # 保存产品搜索上下文，用于用户取消后重新发起请求时恢复
            products = self._extract_products_from_result(result, state)
            if products:
                from datetime import datetime
                updated_state["last_product_search_context"] = {
                    "products": products,
                    "search_keyword": state.entities.get("search_keyword"),
                    "quantity": state.entities.get("quantity", 1),
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.info(f"[Product Agent节点] 保存产品搜索上下文: {len(products)} 个产品, keyword={state.entities.get('search_keyword')}")

            return updated_state
        except Exception as e:
            logger.error(f"Product Agent节点执行错误: {str(e)}", exc_info=True)
            return {"next_action": "finish", "error_message": f"Product Agent错误: {str(e)}"}

    def _extract_products_from_result(
        self, result: Dict[str, Any], state: MultiAgentState
    ) -> list:
        """从结果中提取产品列表"""
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, dict) and "products" in data:
                        logger.info(f"从最新的 ToolMessage 中提取到 {len(data['products'])} 个产品")
                        return data["products"]
                except (json.JSONDecodeError, TypeError):
                    continue

        product_result = state.agent_results.get("product_agent", {})
        if isinstance(product_result, dict) and "products" in product_result:
            return product_result["products"]

        return []

    async def order_agent_node(
        self, state: MultiAgentState, config: Optional[RunnableConfig] = None
    ) -> MultiAgentState:
        """Order Agent节点（订单管理，含确认机制）"""
        try:
            order_agent = getattr(self.graph, "order_agent", None)
            if not order_agent:
                logger.error("Order Agent未找到")
                return {"next_action": "finish", "error_message": "Order Agent未找到"}

            session_id = "default"
            if config and "configurable" in config:
                session_id = config["configurable"].get("session_id", "default")

            result = await order_agent.execute(state, session_id=session_id)

            # 传播 conversation_phase（order_agent 可能会更新它）
            # 例如：用户确认订单后会设置为 "order_completed"，开始准备订单时设置为 "order_creating"
            result_phase = result.get("conversation_phase")

            updated_state = {
                "messages": result.get("messages", state.messages),
                "current_agent": "order_agent",
                "confirmation_pending": result.get("confirmation_pending"),
                "tools_used": result.get("tools_used", state.tools_used),
            }

            # 如果 order_agent 返回了 conversation_phase，使用它；否则保持当前 phase
            if result_phase is not None:
                updated_state["conversation_phase"] = result_phase
                logger.info(f"[Order Agent节点] 更新 conversation_phase: {result_phase}")

            return updated_state
        except Exception as e:
            logger.error(f"Order Agent节点执行错误: {str(e)}", exc_info=True)
            return {"next_action": "finish", "error_message": f"Order Agent错误: {str(e)}"}
