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
from src.multi_agent.constants import ActionName, MetadataKeys

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

    async def context_manager_node(self, state: MultiAgentState) -> MultiAgentState:
        """上下文管理节点 - 智能提取和压缩上下文

        职责：
        1. 提取当前查询（最后一条HumanMessage）
        2. 使用ContextManager构建上下文摘要
        3. 更新state.context_summary

        Args:
            state: 当前状态

        Returns:
            更新后的状态（包含context_summary）
        """
        try:
            # 1. 提取当前查询（从messages中获取最后一条HumanMessage）
            current_query = None
            for msg in reversed(state.messages):
                if hasattr(msg, 'content') and msg.content:
                    from langchain_core.messages import HumanMessage
                    if isinstance(msg, HumanMessage):
                        current_query = msg.content
                        break

            if not current_query:
                current_query = state.original_question or ""

            # 2. 使用ContextPipeline构建统一上下文与摘要
            context_bundle = await self.graph.context_pipeline.build(
                state=state,
                current_query=current_query
            )

            logger.info(
                f"📊【上下文管理】构建摘要完成: "
                f"{len(context_bundle.short_term_context.get('conversation_history', []))}轮对话, "
                f"{len(context_bundle.short_term_context.get('recent_tool_calls', []))}个工具调用"
            )

            # 3. 更新state
            # 更新上下文缓存元数据（轻量缓存）
            context_cache = {
                "message_count": len(state.messages),
                "history_rounds": len(context_bundle.short_term_context.get("conversation_history", [])),
                "tool_calls_count": len(context_bundle.short_term_context.get("recent_tool_calls", [])),
            }
            context_version = (state.metadata or {}).get("context_version", 0) + 1

            return {
                "context_bundle": context_bundle.model_dump(),
                "original_question": current_query,
                "metadata": {
                    **state.metadata,
                    MetadataKeys.CONTEXT_CACHE.value: context_cache,
                    MetadataKeys.CONTEXT_VERSION.value: context_version,
                    MetadataKeys.CONTEXT_OWNER.value: "context_manager",
                },
            }

        except Exception as e:
            logger.error(f"上下文管理节点执行错误: {str(e)}", exc_info=True)
            # 返回None，避免阻塞流程
            return {
                "context_bundle": None,
                "original_question": state.original_question,
            }

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

            # 【改进】构建增强查询（包含上下文）
            enhanced_query = self._build_enhanced_query(
                current_query=question,
                context_bundle=state.context_bundle
            )

            if enhanced_query != question:
                logger.info(f"📊【意图识别】使用增强查询（包含上下文）")

            # 执行意图识别（Joint Intent Detection and Slot Filling）
            if not self.graph.intent_classifier:
                return {"query_intent": None, "original_question": question}

            # 使用异步方法提高性能（使用增强后的查询）
            intent = await self.graph.intent_classifier.aclassify(enhanced_query)

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
                    "next_action": ActionName.FINISH,
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
                "next_action": ActionName.FINISH,
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
                    "next_action": ActionName.FINISH,
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
                "next_action": ActionName.FINISH,
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

    def _build_enhanced_query(
        self,
        current_query: str,
        context_bundle: Optional[Dict[str, Any]],
        max_history_turns: int = 10
    ) -> str:
        """构建增强的查询（包含上下文）

        将上下文信息附加到当前查询前，帮助意图识别理解历史对话和累积实体。

        Args:
            current_query: 当前用户查询
            context_summary: 上下文摘要
            max_history_turns: 最大显示对话历史轮数，默认10轮

        Returns:
            增强后的查询（包含上下文）
        """
        if not context_bundle:
            return current_query

        # 构建上下文字符串
        context_parts = []

        # 添加对话历史摘要（最近N轮）
        short_term = context_bundle.get("short_term_context", {})
        history = short_term.get("conversation_history", [])
        if history:
            context_parts.append("【最近对话】")
            # 显示最近N轮（由参数控制）
            for idx, turn in enumerate(history[-max_history_turns:], 1):
                if turn.get("human"):
                    human_msg = turn['human']
                    # 限制长度，避免token消耗过大
                    if len(human_msg) > 100:
                        human_msg = human_msg[:100] + "..."
                    context_parts.append(f"  用户: {human_msg}")
                if turn.get("ai"):
                    ai_msg = turn['ai']
                    if len(ai_msg) > 100:
                        ai_msg = ai_msg[:100] + "..."
                    context_parts.append(f"  AI: {ai_msg}")

        # 添加关键实体
        key_entities = short_term.get("key_entities", {})
        if key_entities:
            context_parts.append("\n【当前状态】")
            if key_entities.get("product_id"):
                context_parts.append(f"  已选定产品ID: {key_entities['product_id']}")
            if key_entities.get("product_ids"):
                context_parts.append(f"  已选定产品IDs: {key_entities['product_ids']}")
            if key_entities.get("order_id"):
                context_parts.append(f"  订单ID: {key_entities['order_id']}")

        # 组合上下文和当前查询
        if context_parts:
            context_str = "\n".join(context_parts)
            return f"{context_str}\n\n当前问题: {current_query}"

        return current_query

