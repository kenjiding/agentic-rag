"""Graph节点处理器 - 封装所有节点执行逻辑（一步一步智能模式）

将节点执行逻辑从主图类中分离，提高代码可维护性和可测试性。

2025-2026 最佳实践：
- 动态节点调用：通过Agent名称动态获取Agent实例
- 注册表驱动：从AgentRegistry获取Agent描述符
- 统一执行逻辑：所有Agent使用统一的执行流程
- 特殊逻辑下放：Agent特定的状态更新逻辑由Agent.execute返回值处理
- interrupt()支持：捕获并转换GraphInterrupt为状态更新
"""
import logging
from typing import Dict, Any, Optional, Callable
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from src.multi_agent.state import MultiAgentState

logger = logging.getLogger(__name__)


class GraphNodeHandler:
    """图节点处理器 - 封装所有节点执行逻辑（一步一步智能模式）

    使用注册表模式，支持动态Agent调用。新增Agent时无需修改此文件。
    """

    def __init__(self, graph_instance):
        """
        初始化节点处理器

        Args:
            graph_instance: MultiAgentGraph实例，用于访问agents和注册表
        """
        self.graph = graph_instance

    def _get_agent(self, agent_name: str):
        """根据名称获取Agent实例

        Args:
            agent_name: Agent名称

        Returns:
            Agent实例，如果不存在返回None
        """
        return getattr(self.graph, agent_name, None)

    async def intent_recognition_node(self, state: MultiAgentState) -> MultiAgentState:
        """意图识别节点 - 分析用户查询意图并提取实体"""
        try:
            # 从messages中获取最后一条HumanMessage
            question = None
            for msg in reversed(state.messages):
                if hasattr(msg, 'content') and msg.content:
                    from langchain_core.messages import HumanMessage
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

            # 提取实体 - 合并新提取的实体到现有实体中
            entities = {**state.entities}

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
        self,
        state: MultiAgentState,
        agent_name: str,
        config: Optional[RunnableConfig] = None
    ) -> MultiAgentState:
        """通用Agent节点执行逻辑

        统一处理所有Agent的执行，特殊逻辑由Agent.execute返回值处理。

        支持 LangGraph 1.x interrupt() 机制：
        - 捕获 GraphInterrupt 异常
        - 转换为 __interrupt__ 状态更新传递给客户端
        - 客户端使用 Command(resume=...) 恢复执行

        Args:
            state: 当前状态
            agent_name: Agent名称
            config: LangGraph配置（用于传递session_id等）

        Returns:
            更新后的状态片段
        """
        try:
            agent = self._get_agent(agent_name)
            if not agent:
                logger.error(f"{agent_name} 未找到")
                return {
                    "next_action": "finish",
                    "error_message": f"{agent_name} 未找到"
                }

            # 获取session_id（用于order_agent等需要会话的Agent）
            session_id = "default"
            if config and "configurable" in config:
                session_id = config["configurable"].get("session_id", "default")

            # 所有Agent统一接受session_id参数
            result = await agent.execute(state, session_id=session_id)

            # Agent 已经在 messages 中添加了所有需要的新消息
            # result["result"] 中的字段只是用于存储数据，不应该再次添加为消息
            additional_messages = result.get("messages", [])
            agent_result = result.get("result")

            # 合并基础状态更新
            updated_state = {
                "messages": state.messages + additional_messages,
                "agent_results": {
                    **state.agent_results,
                    agent_name: agent_result
                },
                "agent_history": state.agent_history + [{
                    "agent": agent_name,
                    "result": agent_result,
                    "metadata": result.get("metadata", {})
                }]
            }

            # 合并Agent返回的所有其他字段（支持Agent自定义状态更新）
            for key, value in result.items():
                if key not in ["messages", "result", "metadata"]:
                    # 特殊处理entities字段：合并而不是覆盖
                    if key == "entities" and isinstance(value, dict) and isinstance(state.entities, dict):
                        updated_state[key] = {**state.entities, **value}
                    else:
                        updated_state[key] = value

            logger.info(f"{agent_name} 执行完成")
            return updated_state

        except GraphInterrupt as e:
            # LangGraph 1.x interrupt() 机制
            # 捕获 interrupt() 调用，转换为状态更新传递给客户端
            # GraphInterrupt 的结构: (Interrupt(value={...}),)
            # 需要从 e.args[0].value 获取实际的值
            interrupt_value = None
            if e.args and len(e.args) > 0:
                interrupt_obj = e.args[0]
                if hasattr(interrupt_obj, 'value'):
                    interrupt_value = interrupt_obj.value
                else:
                    # 如果 args[0] 本身就是值（向后兼容）
                    interrupt_value = interrupt_obj

            # 【关键修复】GraphInterrupt 的 value 可能是 (Interrupt(...),) 这样的 tuple
            # 需要继续解析获取实际的字典值
            if interrupt_value and isinstance(interrupt_value, tuple) and len(interrupt_value) > 0:
                first_element = interrupt_value[0]
                if hasattr(first_element, 'value'):
                    # Interrupt 对象，获取其 value 属性
                    interrupt_value = first_element.value
                elif isinstance(first_element, dict):
                    # 直接是字典
                    interrupt_value = first_element

            # 【关键修复】GraphInterrupt 会被 LangGraph 捕获，不会将返回值包含在 stream 输出中
            # 所以我们需要重新抛出异常，让 LangGraph 处理
            # LangGraph 会将 interrupt 信息保存到 checkpointer，客户端可以通过 get_state() 获取
            raise

        except Exception as e:
            logger.error(f"{agent_name} 节点执行错误: {str(e)}", exc_info=True)
            return {
                "next_action": "finish",
                "error_message": f"{agent_name} 错误: {str(e)}"
            }

    def create_agent_node(self, agent_name: str) -> Callable:
        """创建Agent节点函数（工厂方法）

        根据Agent名称创建对应的节点函数，用于LangGraph图构建。
        新增Agent时无需添加新的节点方法，只需调用此工厂方法即可。

        Args:
            agent_name: Agent名称

        Returns:
            节点函数

        Example:
            graph.add_node("rag_agent", node_handler.create_agent_node("rag_agent"))
            graph.add_node("chat_agent", node_handler.create_agent_node("chat_agent"))
        """
        async def agent_node(state: MultiAgentState, config: Optional[RunnableConfig] = None) -> MultiAgentState:
            return await self._execute_agent_node(state, agent_name, config)

        agent_node.__name__ = f"{agent_name}_node"
        return agent_node

    # ===== 以下为向后兼容保留的方法，内部使用create_agent_node =====

    async def rag_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """RAG Agent节点（向后兼容）"""
        return await self._execute_agent_node(state, "rag_agent")

    async def chat_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Chat Agent节点（向后兼容）"""
        return await self._execute_agent_node(state, "chat_agent")

    async def product_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Product Agent节点（向后兼容）"""
        return await self._execute_agent_node(state, "product_agent")

    async def order_agent_node(
        self, state: MultiAgentState, config: Optional[RunnableConfig] = None
    ) -> MultiAgentState:
        """Order Agent节点（向后兼容）"""
        return await self._execute_agent_node(state, "order_agent", config)
