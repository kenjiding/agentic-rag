"""Graph节点处理器 - 封装所有节点执行逻辑

将节点执行逻辑从主图类中分离，提高代码可维护性和可测试性。
"""
import logging
from typing import Dict, Any, Optional
from langgraph.errors import GraphInterrupt
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
import json

from src.multi_agent.state import MultiAgentState
from src.multi_agent.task_orchestrator import get_task_orchestrator

logger = logging.getLogger(__name__)


class GraphNodeHandler:
    """图节点处理器 - 封装所有节点执行逻辑"""
    
    def __init__(self, graph_instance):
        """初始化节点处理器
        
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
            # 注意：意图识别只分析当前查询，不包含历史上下文
            # 历史上下文会在后续的 Supervisor 路由和 Agent 执行时使用（从 state.entities 中提取）
            # 这样设计符合单一职责原则：意图识别专注于分析当前查询，上下文理解由后续组件处理
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
        """Supervisor节点 - 路由决策"""
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

            if "task_chain" in routing_decision:
                updated_state["task_chain"] = routing_decision["task_chain"]
                logger.info("任务链已添加到 state，checkpointer 将自动持久化")

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
                "tools_used": result.get("tools_used", state.tools_used)
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

            # 任务链模式：保存结果并继续执行
            task_chain = state.task_chain
            if task_chain:
                products = self._extract_products_from_result(result, state)
                orchestrator = get_task_orchestrator()
                current_index = task_chain.current_step_index
                steps = task_chain.steps

                if current_index < len(steps):
                    updated_step = steps[current_index].model_copy(update={
                        "result_data": {"products": products or []},
                        "status": "completed"
                    })
                    updated_steps = list(steps)
                    updated_steps[current_index] = updated_step
                    task_chain = task_chain.model_copy(update={"steps": updated_steps})

                    # 【核心修改】order_with_search 类型的任务链在 product_search 完成后结束
                    # 等待用户点击"购买"按钮来触发新的订单流程
                    if task_chain.chain_type == "order_with_search":
                        # 任务链完成，等待用户交互
                        updated_state.update({
                            "task_chain": None,  # 清除任务链
                            "next_action": "finish"
                        })
                        logger.info(f"order_with_search 产品搜索完成，找到 {len(products) if products else 0} 个产品。任务链结束，等待用户点击购买按钮")
                    else:
                        # 其他类型的任务链继续执行
                        task_chain = orchestrator.move_to_next_step(task_chain)
                        updated_state.update({
                            "task_chain": task_chain,
                            "next_action": "execute_task_chain"
                        })
                        logger.info(f"产品搜索完成，找到 {len(products) if products else 0} 个产品")

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
            updated_state = {
                "messages": result.get("messages", state.messages),
                "current_agent": "order_agent",
                "confirmation_pending": result.get("confirmation_pending"),
                "tools_used": result.get("tools_used", state.tools_used),
            }

            # 处理任务链
            task_chain = state.task_chain
            if task_chain:
                updated_state = self._handle_order_agent_task_chain(
                    state, task_chain, result, updated_state
                )

            return updated_state
        except Exception as e:
            logger.error(f"Order Agent节点执行错误: {str(e)}", exc_info=True)
            return {"next_action": "finish", "error_message": f"Order Agent错误: {str(e)}"}

    def _handle_order_agent_task_chain(
        self, state: MultiAgentState, task_chain, result: Dict[str, Any], updated_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理Order Agent的任务链逻辑"""
        current_index = task_chain.current_step_index
        steps = task_chain.steps

        if current_index >= len(steps):
            return updated_state

        current_step = steps[current_index]
        step_type = current_step.step_type

        if step_type == "order_creation":
            return self._handle_order_creation_step(
                state, task_chain, current_step, current_index, result, updated_state
            )
        elif result.get("confirmation_pending"):
            # 确认机制已改为使用 interrupt()，这里不再需要设置 wait_for_confirmation
            # confirmation_pending 仅用于状态记录，实际的暂停由 interrupt() 处理
            logger.info("检测到确认请求，应已通过 interrupt() 处理")
            return updated_state
        else:
            return self._handle_other_order_steps(
                state, task_chain, current_step, current_index, result, updated_state
            )

    def _handle_order_creation_step(
        self, state: MultiAgentState, task_chain, current_step, current_index: int, result: Dict[str, Any], updated_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理order_creation步骤"""
        order_info = result.get("order_info") or {}
        updated_step = current_step.model_copy(update={
            "status": "completed",
            "result_data": {
                "order_info": order_info,
                "message": result.get("messages", [])[-1].content if result.get("messages") else ""
            }
        })

        updated_steps = list(task_chain.steps)
        updated_steps[current_index] = updated_step
        updated_task_chain = task_chain.model_copy(update={"steps": updated_steps})

        has_order_info = order_info and order_info.get("can_create")
        logger.info(f"[Order Agent节点] order_creation 步骤结果: has_order_info={has_order_info}")

        if not has_order_info:
            updated_state["task_chain"] = updated_task_chain
            updated_state["next_action"] = "finish"
            logger.info("[Order Agent节点] order_creation 步骤缺少必要信息，保持任务链活跃")
            return updated_state

        orchestrator = get_task_orchestrator()
        updated_task_chain = orchestrator.move_to_next_step(updated_task_chain)
        updated_state["task_chain"] = updated_task_chain

        if updated_task_chain.current_step_index < len(updated_task_chain.steps):
            updated_state["next_action"] = "execute_task_chain"
        else:
            updated_state["task_chain"] = None
            logger.warning("[Order Agent节点] 任务链在 order_creation 后完成，缺少 confirmation 步骤")

        return updated_state

    def _handle_other_order_steps(
        self, state: MultiAgentState, task_chain, current_step, current_index: int, result: Dict[str, Any], updated_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理其他订单步骤"""
        step_type = current_step.step_type
        previous_confirmation_pending = state.confirmation_pending

        if step_type == "confirmation" and previous_confirmation_pending:
            logger.info("用户取消了 confirmation 步骤，清理任务链")
            entities = state.entities.copy()
            entities.pop("selected_product_id", None)  # 只清理选择操作产生的实体
            # 不清理 quantity 和 search_keyword，这些是用户原始意图的一部分
            # 保留 last_product_search_context，用于用户重新发起购买请求时恢复上下文
            updated_state.update({
                "task_chain": None,
                "next_action": "finish",
                "confirmation_pending": None,
                "entities": entities,
                "last_product_search_context": state.last_product_search_context  # 保留搜索上下文
            })
            return updated_state

        orchestrator = get_task_orchestrator()
        updated_steps = list(task_chain.steps)
        updated_steps[current_index] = updated_steps[current_index].model_copy(update={
            "status": "completed",
            "result_data": {"message": result.get("messages", [])[-1].content if result.get("messages") else ""}
        })
        task_chain = task_chain.model_copy(update={"steps": updated_steps})
        task_chain = orchestrator.move_to_next_step(task_chain)
        updated_state["task_chain"] = task_chain

        if task_chain.current_step_index < len(task_chain.steps):
            updated_state["next_action"] = "execute_task_chain"
        else:
            updated_state["task_chain"] = None
            logger.info("任务链已完成")

        return updated_state

    async def task_orchestrator_node(
        self, state: MultiAgentState, config: Optional[RunnableConfig] = None
    ) -> MultiAgentState:
        """Task Orchestrator节点 - 任务编排器"""
        try:
            orchestrator = get_task_orchestrator()
            session_id = "default"
            if config and "configurable" in config:
                session_id = config["configurable"].get("session_id", "default")

            task_chain = state.task_chain
            if task_chain:
                current_index = task_chain.current_step_index
                steps = task_chain.steps

                if current_index < len(steps):
                    current_step = steps[current_index]
                    step_type = current_step.step_type

            result = await orchestrator.execute_current_step(state, session_id)
            updated_state = {
                "task_chain": result.get("task_chain", state.task_chain),
                "confirmation_pending": result.get("confirmation_pending"),
                "next_action": result.get("next_action"),
                "selected_agent": result.get("selected_agent"),
            }
            
            # 【关键修复】如果 result 中包含 messages，需要合并到状态中
            # 这通常发生在任务链完成时，会返回最终的成功消息
            if "messages" in result:
                existing_messages = state.messages or []
                new_messages = result.get("messages", [])
                # 合并消息，避免重复
                existing_ids = {id(msg) if hasattr(msg, 'id') else str(msg) for msg in existing_messages}
                for msg in new_messages:
                    msg_id = id(msg) if hasattr(msg, 'id') else str(msg)
                    if msg_id not in existing_ids:
                        existing_messages.append(msg)
                        existing_ids.add(msg_id)
                updated_state["messages"] = existing_messages
                logger.info(f"[Task Orchestrator节点] 已合并 {len(new_messages)} 条新消息到状态中")

            if updated_state.get("task_chain"):
                logger.info(f"[Task Orchestrator节点] 任务链已更新: current_step_index={updated_state['task_chain'].current_step_index if updated_state['task_chain'] else None}")

            return updated_state

        except GraphInterrupt:
            logger.info(f"[Task Orchestrator节点] GraphInterrupt 被捕获，重新抛出")
            raise
        except Exception as e:
            logger.error(f"Task Orchestrator节点执行错误: {str(e)}", exc_info=True)
            return {
                "next_action": "finish",
                "error_message": f"Task Orchestrator错误: {str(e)}",
                "task_chain": None
            }

