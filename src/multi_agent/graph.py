"""Multi-Agent Graph - 多Agent系统主图

本模块使用LangGraph构建多Agent系统的核心工作流。
实现了Supervisor模式，协调多个Agent的执行。

2025-2026 最佳实践：
- 使用LangGraph 1.x最新API
- Supervisor模式实现智能路由
- 清晰的状态管理
- 错误处理和重试机制
- 可扩展的架构设计
"""
import asyncio
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from src.multi_agent.state import MultiAgentState
from src.multi_agent.supervisor import SupervisorAgent
from src.intent import IntentClassifier
from src.multi_agent.agents.base_agent import BaseAgent
from src.multi_agent.agents.rag_agent import RAGAgent
from src.multi_agent.agents.chat_agent import ChatAgent
from src.multi_agent.agents.product_agent import ProductAgent, product_agent_node
from src.multi_agent.agents.order_agent import OrderAgent, order_agent_node
from src.multi_agent.tools.tool_registry import ToolCategory, ToolPermission, ToolRegistry
from src.multi_agent.task_chain_storage import get_task_chain_storage
import logging
from src.tools.web_search import create_web_search_tool

logger = logging.getLogger(__name__)


class MultiAgentGraph:
    """多Agent系统主图
    
    职责：
    1. 初始化所有Agent和工具
    2. 构建LangGraph工作流
    3. 管理状态流转
    4. 协调Agent执行
    
    架构：
    - Supervisor节点：路由决策
    - RAG Agent节点：知识检索
    - Chat Agent节点：一般对话
    - Product Agent节点：商品搜索
    - Order Agent节点：订单管理（含确认机制）
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        agents: Optional[List[BaseAgent]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        rag_persist_directory: str = "./tmp/chroma_db/agentic_rag",
        max_iterations: int = 10,
        init_web_search: bool = True,
        enable_intent_classification: bool = True,
        enable_business_agents: bool = True
    ):
        """
        初始化多Agent图

        Args:
            llm: 语言模型实例
            agents: 自定义Agent列表，如果为None则使用默认Agent
            tool_registry: 工具注册表
            rag_persist_directory: RAG向量数据库持久化目录
            max_iterations: 最大迭代次数
            init_web_search: 是否在初始化时加载web search tools（默认True）
                            如果设置为False，可以稍后调用async_init_web_search_tools()异步加载
            enable_intent_classification: 是否启用意图识别（默认True）
            enable_business_agents: 是否启用业务Agent（商品、订单），默认True
        """
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations
        self._web_search_initialized = False
        self.enable_intent_classification = enable_intent_classification
        self.enable_business_agents = enable_business_agents

        # 延迟加载web search tools，避免阻塞初始化
        # 如果初始化失败，系统仍可正常运行（只是没有web search功能）
        if init_web_search:
            try:
                self._init_web_search_tools()
            except Exception as e:
                logger.warning(f"Web search tools初始化失败，系统将在没有web search功能的情况下运行: {e}")
                self._web_search_initialized = False
        else:
            logger.info("跳过web search tools初始化，可在需要时调用async_init_web_search_tools()异步加载")
        # 初始化Supervisor
        self.supervisor = SupervisorAgent(llm=self.llm)

        # 初始化意图分类器
        if self.enable_intent_classification:
            self.intent_classifier = IntentClassifier(llm=self.llm)
        else:
            self.intent_classifier = None
        
        # 初始化默认Agents（如果未提供）
        if agents is None:
            agents = []
            # 添加RAG Agent
            rag_agent = RAGAgent(
                llm=self.llm,
                persist_directory=rag_persist_directory
            )
            agents.append(rag_agent)

            # 添加Chat Agent
            chat_agent = ChatAgent(
                llm=self.llm,
                tool_registry=self.tool_registry
            )
            agents.append(chat_agent)

            # 添加业务 Agent
            if self.enable_business_agents:
                # Product Agent
                product_agent = ProductAgent(llm=self.llm)
                agents.append(product_agent)

                # Order Agent
                order_agent = OrderAgent(llm=self.llm)
                agents.append(order_agent)
        
        # 注册所有Agents并自动分配工具注册表
        for agent in agents:
            self.supervisor.register_agent(agent)
            # 直接使用agent名称作为属性名，避免重复命名（如rag_agent_agent）
            setattr(self, agent.get_name(), agent)
            
            # 自动为ToolEnabledAgent类型的Agent分配工具注册表
            # 如果Agent已经有tool_registry，则不覆盖
            if hasattr(agent, 'tool_registry') and agent.tool_registry is None:
                agent.tool_registry = self.tool_registry
                # 刷新工具列表
                if hasattr(agent, 'refresh_tools'):
                    agent.refresh_tools()
                logger.info(f"已为Agent {agent.get_name()} 分配工具注册表")
        
        # 构建图
        self.graph = self._build_graph()
    
    def _init_web_search_tools(self):
        """
        初始化web search tools（基于 DDGS）
        如果失败，系统仍可正常运行
        """
        try:
            # 使用 DDGS 创建 web search tools（同步调用）
            web_search_tools = create_web_search_tool()
            
            if web_search_tools:
                for tool in web_search_tools:
                    self.tool_registry.register_tool(
                        name=tool.name,
                        tool=tool,
                        category=ToolCategory.SEARCH,
                        permission=ToolPermission.PUBLIC,
                        allowed_agents=["chat_agent", "rag_agent"]
                    )
                logger.info(f"成功注册 {len(web_search_tools)} 个web search tools（基于 DDGS）")
                self._web_search_initialized = True
            else:
                logger.warning("Web search tools返回为空")
                self._web_search_initialized = False
        except Exception as e:
            logger.warning(f"Web search tools初始化失败: {e}", exc_info=True)
            self._web_search_initialized = False
            # 不重新抛出异常，允许系统在没有web search的情况下运行
    
    async def async_init_web_search_tools(self):
        """
        异步初始化web search tools（基于 DDGS）
        可以在需要时异步调用，不会阻塞
        """
        if self._web_search_initialized:
            logger.info("Web search tools已经初始化")
            return
        
        try:
            # 使用 DDGS 创建 web search tools（同步函数，但在异步上下文中调用）
            web_search_tools = create_web_search_tool()
            if web_search_tools:
                for tool in web_search_tools:
                    self.tool_registry.register_tool(
                        name=tool.name,
                        tool=tool,
                        category=ToolCategory.SEARCH,
                        permission=ToolPermission.PUBLIC,
                        allowed_agents=["chat_agent", "rag_agent"]
                    )
                logger.info(f"成功注册 {len(web_search_tools)} 个web search tools（基于 DDGS）")
                self._web_search_initialized = True
                
                # 刷新所有已注册的agent的工具列表
                for agent in self.supervisor.get_registered_agents():
                    if hasattr(agent, 'refresh_tools'):
                        agent.refresh_tools()
            else:
                logger.warning("Web search tools返回为空")
        except Exception as e:
            logger.warning(f"异步初始化web search tools失败: {e}", exc_info=True)
            self._web_search_initialized = False
    
    def _build_graph(self) -> StateGraph:
        """
        构建LangGraph工作流

        流程: intent_recognition -> supervisor -> [agents | task_orchestrator] -> finish

        新增任务编排支持：
        - task_orchestrator：多步骤任务编排节点
        - 支持复杂的多步骤交互流程（如商品搜索→用户选择→订单创建）

        Returns:
            编译后的图
        """
        # 创建状态图
        graph = StateGraph(MultiAgentState)

        # 添加节点
        if self.intent_classifier:
            graph.add_node("intent_recognition", self._intent_recognition_node)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("task_orchestrator", self._task_orchestrator_node)  # 新增：任务编排节点
        graph.add_node("rag_agent", self._rag_agent_node)
        graph.add_node("chat_agent", self._chat_agent_node)

        # 添加业务 Agent 节点
        if self.enable_business_agents:
            graph.add_node("product_agent", self._product_agent_node)
            graph.add_node("order_agent", self._order_agent_node)

        # 设置入口点 - 意图识别优先
        if self.intent_classifier:
            graph.set_entry_point("intent_recognition")
            # 意图识别后进入Supervisor
            graph.add_edge("intent_recognition", "supervisor")
        else:
            graph.set_entry_point("supervisor")
        
        # 添加条件边：Supervisor根据路由决策选择下一个节点
        route_mapping = {
            "rag_agent": "rag_agent",
            "chat_agent": "chat_agent",
            "task_orchestrator": "task_orchestrator",  # 新增：任务编排
            "finish": END
        }
        if self.enable_business_agents:
            route_mapping["product_agent"] = "product_agent"
            route_mapping["order_agent"] = "order_agent"

        graph.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            route_mapping
        )

        # 新增：Task Orchestrator 条件路由
        orchestrator_route_mapping = {
            "product_agent": "product_agent",
            "order_agent": "order_agent",
            "wait_for_selection": END,  # 暂停，等待用户选择
            "finish": END
        }
        graph.add_conditional_edges(
            "task_orchestrator",
            self._route_after_orchestrator,
            orchestrator_route_mapping
        )
        
        # Agent执行后回到Supervisor（继续路由或结束）
        graph.add_conditional_edges(
            "rag_agent",
            self._route_after_agent,
            {
                "supervisor": "supervisor",  # 如果需要web search，回到Supervisor路由到chat_agent
                "chat_agent": "chat_agent",  # 直接路由到chat_agent（如果RAG失败）
                "finish": END
            }
        )
        
        graph.add_conditional_edges(
            "chat_agent",
            self._route_after_agent,
            {
                "supervisor": "supervisor",
                "finish": END
            }
        )

        # 业务 Agent 执行后的边
        if self.enable_business_agents:
            graph.add_conditional_edges(
                "product_agent",
                self._route_after_agent,
                {
                    "task_orchestrator": "task_orchestrator",  # 新增：任务链模式
                    "supervisor": "supervisor",
                    "finish": END
                }
            )
            graph.add_conditional_edges(
                "order_agent",
                self._route_after_agent,
                {
                    "task_orchestrator": "task_orchestrator",  # 新增：任务链模式
                    "supervisor": "supervisor",
                    "finish": END
                }
            )
        
        # 编译图
        return graph.compile()
    
    async def _intent_recognition_node(self, state: MultiAgentState) -> MultiAgentState:
        """
        意图识别节点 - 分析用户查询意图

        在进入Supervisor之前先进行意图识别，这样可以：
        1. 拆分复杂问题为子查询
        2. 为Supervisor提供更多上下文信息
        3. 优化路由决策

        Args:
            state: 当前状态

        Returns:
            更新后的状态（包含query_intent）
        """
        try:
            # 提取用户问题
            question = state.get("original_question")
            if not question or not isinstance(question, str):
                # 从messages中获取
                for msg in state.get("messages", []):
                    if isinstance(msg, HumanMessage):
                        question = msg.content
                        break

            if not question:
                logger.warning("未找到用户问题，跳过意图识别")
                # 只返回更新的字段，LangGraph会自动合并
                return {
                    "query_intent": None,
                    "original_question": question
                }

            logger.info(f"🎯【意图识别】分析查询: {question}")

            # 检查intent_classifier是否可用
            if self.intent_classifier is None:
                logger.warning("意图分类器未初始化，跳过意图识别")
                return {
                    "query_intent": None,
                    "original_question": question
                }

            # 执行意图识别
            intent = self.intent_classifier.classify(question)

            # 转换为字典格式存储到状态
            intent_dict = intent.model_dump()

            # 打印识别结果
            logger.info(f"🎯【意图识别】类型: {intent.intent_type}, 复杂度: {intent.complexity}")
            if intent.needs_decomposition:
                logger.info(f"🎯【意图识别】需要分解: {intent.decomposition_type}")
                logger.info(f"🎯【意图识别】子查询数: {len(intent.sub_queries)}")
                for sq in intent.sub_queries[:3]:
                    logger.info(f"  - {sq.query[:50]}...")

            # 更新状态 - 只返回需要更新的字段
            # 注意：intent_dict 已经通过 model_dump() 包含了所有字段，包括 sub_queries
            updated_state = {
                "query_intent": intent_dict,
                "original_question": question
            }

            logger.info(f"🎯【意图识别】完成，置信度: {intent.confidence:.2f}")
            return updated_state

        except Exception as e:
            logger.error(f"意图识别节点执行错误: {str(e)}", exc_info=True)
            return {
                "query_intent": None,
                "error_message": f"意图识别错误: {str(e)}"
            }

    async def _supervisor_node(self, state: MultiAgentState, config: Optional[RunnableConfig] = None) -> MultiAgentState:
        """
        Supervisor节点 - 路由决策（生产环境异步版本）

        企业级最佳实践：
        - 使用异步函数提高并发性能
        - 直接使用await调用异步方法，避免事件循环管理
        - LangGraph完全支持异步节点

        Args:
            state: 当前状态
            config: 执行配置，包含 session_id

        Returns:
            更新后的状态
        """
        try:
            # 检查迭代次数
            iteration_count = state.get("iteration_count", 0)
            if iteration_count >= self.max_iterations:
                logger.warning(f"达到最大迭代次数 {self.max_iterations}，结束执行")
                # LangGraph会自动合并状态，只需返回需要更新的字段
                return {
                    "next_action": "finish",
                    "routing_reason": f"达到最大迭代次数 {self.max_iterations}"
                }

            # 调用Supervisor进行路由决策（生产环境：直接使用await）
            routing_decision = await self.supervisor.route(state)

            # 更新状态
            # LangGraph会自动合并状态，只需返回需要更新的字段
            updated_state = {
                "next_action": routing_decision["next_action"],
                "current_agent": routing_decision.get("selected_agent"),
                "routing_reason": routing_decision.get("routing_reason", ""),
                "iteration_count": iteration_count + 1
            }

            # 如果有 task_chain，也需要添加到状态中，并保存到存储
            if "task_chain" in routing_decision:
                updated_state["task_chain"] = routing_decision["task_chain"]

                # 保存任务链到存储，以便用户选择后可以继续执行
                if config and config.get("configurable", {}).get("session_id"):
                    session_id = config["configurable"]["session_id"]
                    storage = get_task_chain_storage()
                    storage.save(session_id, routing_decision["task_chain"])
                    logger.info(f"已保存任务链到存储: session={session_id}")

            logger.info(f"Supervisor决策: {routing_decision}")
            return updated_state

        except Exception as e:
            logger.error(f"Supervisor节点执行错误: {str(e)}", exc_info=True)
            # LangGraph会自动合并状态，只需返回需要更新的字段
            return {
                "next_action": "finish",
                "error_message": f"Supervisor错误: {str(e)}",
                "routing_reason": f"执行错误: {str(e)}"
            }
    
    async def _rag_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """
        RAG Agent节点（生产环境异步版本）
        
        企业级最佳实践：
        - 使用异步函数提高并发性能
        - 直接使用await调用异步方法
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 从实例属性中获取RAG Agent（使用agent的name作为属性名）
            rag_agent = getattr(self, "rag_agent", None)
            if not rag_agent:
                logger.error("RAG Agent未找到")
                # LangGraph会自动合并状态，只需返回需要更新的字段
                return {
                    "next_action": "finish",
                    "error_message": "RAG Agent未找到"
                }
            
            # 执行RAG Agent（生产环境：使用await异步执行）
            result = await rag_agent.execute(state)
            
            # 更新状态
            # LangGraph会自动合并状态，只需返回需要更新的字段
            updated_state = {
                "messages": state["messages"] + result.get("messages", []),
                "agent_results": {
                    **state.get("agent_results", {}),
                    "rag_agent": result.get("result")
                },
                "agent_history": state.get("agent_history", []) + [{
                    "agent": "rag_agent",
                    "result": result.get("result"),
                    "metadata": result.get("metadata", {})
                }]
            }
            
            logger.info("RAG Agent执行完成")
            return updated_state
            
        except Exception as e:
            logger.error(f"RAG Agent节点执行错误: {str(e)}", exc_info=True)
            # LangGraph会自动合并状态，只需返回需要更新的字段
            return {
                "next_action": "finish",
                "error_message": f"RAG Agent错误: {str(e)}"
            }
    
    async def _chat_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """
        Chat Agent节点（生产环境异步版本）
        
        企业级最佳实践：
        - 使用异步函数提高并发性能
        - 直接使用await调用异步方法
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 从实例属性中获取Chat Agent（使用agent的name作为属性名）
            chat_agent = getattr(self, "chat_agent", None)
            if not chat_agent:
                logger.error("Chat Agent未找到")
                # LangGraph会自动合并状态，只需返回需要更新的字段
                return {
                    "next_action": "finish",
                    "error_message": "Chat Agent未找到"
                }
            
            # 执行Chat Agent（生产环境：使用await异步执行）
            result = await chat_agent.execute(state)
            
            # 更新状态
            # LangGraph会自动合并状态，只需返回需要更新的字段
            updated_state = {
                "messages": state["messages"] + result.get("messages", []),
                "agent_results": {
                    **state.get("agent_results", {}),
                    "chat_agent": result.get("result")
                },
                "agent_history": state.get("agent_history", []) + [{
                    "agent": "chat_agent",
                    "result": result.get("result"),
                    "metadata": result.get("metadata", {})
                }]
            }
            
            logger.info("Chat Agent执行完成")
            return updated_state
            
        except Exception as e:
            logger.error(f"Chat Agent节点执行错误: {str(e)}", exc_info=True)
            # LangGraph会自动合并状态，只需返回需要更新的字段
            return {
                "next_action": "finish",
                "error_message": f"Chat Agent错误: {str(e)}"
            }

    async def _product_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Product Agent节点（商品搜索）"""
        try:
            product_agent = getattr(self, "product_agent", None)
            if not product_agent:
                logger.error("Product Agent未找到")
                return {"next_action": "finish", "error_message": "Product Agent未找到"}

            result = product_agent.invoke(state)
            updated_state = {
                "messages": result.get("messages", state.get("messages", [])),
                "current_agent": "product_agent",
                "tools_used": result.get("tools_used", state.get("tools_used", []))
            }

            # 保留context_data
            if state.get("context_data"):
                updated_state["context_data"] = state["context_data"]

            # 任务链模式：保存结果并继续执行
            task_chain = state.get("task_chain")
            if task_chain:
                from src.multi_agent.task_orchestrator import get_task_orchestrator
                products = self._extract_products_from_result(result, state)

                orchestrator = get_task_orchestrator()
                current_index = task_chain["current_step_index"]
                steps = task_chain["steps"]

                if current_index < len(steps):
                    steps[current_index].update({
                        "result_data": {"products": products or []},
                        "status": "completed"
                    })
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

    def _extract_products_from_result(self, result: Dict[str, Any], state: MultiAgentState) -> list:
        """从结果中提取产品列表"""
        import json
        from langchain_core.messages import ToolMessage
        
        # 从 ToolMessage 中提取
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, dict) and "products" in data:
                        return data["products"]
                except (json.JSONDecodeError, TypeError):
                    continue
        
        # 从 agent_results 中提取
        product_result = state.get("agent_results", {}).get("product_agent", {})
        if isinstance(product_result, dict) and "products" in product_result:
            return product_result["products"]
        
        return []

    async def _order_agent_node(self, state: MultiAgentState, config: Optional[RunnableConfig] = None) -> MultiAgentState:
        """Order Agent节点（订单管理，含确认机制）"""
        try:
            order_agent = getattr(self, "order_agent", None)
            if not order_agent:
                logger.error("Order Agent未找到")
                return {"next_action": "finish", "error_message": "Order Agent未找到"}

            session_id = "default"
            if config and "configurable" in config:
                session_id = config["configurable"].get("session_id", "default")

            result = await order_agent.invoke(state, session_id=session_id)
            updated_state = {
                "messages": result.get("messages", state.get("messages", [])),
                "current_agent": "order_agent",
                "confirmation_pending": result.get("confirmation_pending"),
            }

            # 保留context_data
            if state.get("context_data"):
                updated_state["context_data"] = state["context_data"]

            # 任务链模式：处理任务链状态
            task_chain = state.get("task_chain")
            if task_chain:
                current_index = task_chain["current_step_index"]
                steps = task_chain["steps"]
                
                if current_index < len(steps):
                    if result.get("confirmation_pending"):
                        steps[current_index]["status"] = "in_progress"
                    else:
                        from src.multi_agent.task_orchestrator import get_task_orchestrator
                        steps[current_index].update({
                            "status": "completed",
                            "result_data": {"message": result.get("messages", [])[-1].content if result.get("messages") else ""}
                        })
                        task_chain = get_task_orchestrator().move_to_next_step(task_chain)
                        updated_state["task_chain"] = task_chain
                        
                        if task_chain["current_step_index"] < len(task_chain["steps"]):
                            updated_state["next_action"] = "execute_task_chain"
                        else:
                            updated_state["task_chain"] = None

            return updated_state
        except Exception as e:
            logger.error(f"Order Agent节点执行错误: {str(e)}", exc_info=True)
            return {"next_action": "finish", "error_message": f"Order Agent错误: {str(e)}"}

    async def _task_orchestrator_node(
        self,
        state: MultiAgentState,
        config: Optional[RunnableConfig] = None
    ) -> MultiAgentState:
        """
        Task Orchestrator节点 - 任务编排器

        负责执行多步骤任务链的当前步骤，协调多个Agent的执行。

        Args:
            state: 当前状态
            config: 配置信息，包含 session_id

        Returns:
            更新后的状态
        """
        try:
            from src.multi_agent.task_orchestrator import get_task_orchestrator

            # 获取任务编排器
            orchestrator = get_task_orchestrator()

            # 从 config 获取 session_id
            session_id = "default"
            if config and "configurable" in config:
                session_id = config["configurable"].get("session_id", "default")

            # 执行当前步骤
            result = await orchestrator.execute_current_step(state, session_id)

            # 更新状态
            updated_state = {
                "task_chain": result.get("task_chain", state.get("task_chain")),
                "pending_selection": result.get("pending_selection"),
                "next_action": result.get("next_action"),
                "selected_agent": result.get("selected_agent"),
            }
            
            # 传递 context_data（优先使用 result 中的）
            context_data = result.get("context_data") or state.get("context_data")
            if not context_data and result.get("task_chain"):
                context_data = result["task_chain"].get("context_data")
            if context_data:
                updated_state["context_data"] = context_data

            # 如果 task_chain 被更新，保存到存储
            if updated_state.get("task_chain"):
                storage = get_task_chain_storage()
                storage.save(session_id, updated_state["task_chain"])
                logger.info(f"已更新任务链到存储: session={session_id}")

            logger.info(f"Task Orchestrator执行完成: next_action={result.get('next_action')}")
            return updated_state

        except Exception as e:
            logger.error(f"Task Orchestrator节点执行错误: {str(e)}", exc_info=True)
            return {
                "next_action": "finish",
                "error_message": f"Task Orchestrator错误: {str(e)}",
                "task_chain": None  # 清除任务链
            }

    def _route_after_supervisor(self, state: MultiAgentState) -> str:
        """
        Supervisor后的路由决策

        根据Supervisor的决策，路由到相应的Agent、任务编排器或结束。

        新增：支持路由到 task_orchestrator 处理多步骤任务

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        next_action = state.get("next_action", "finish")

        if next_action == "execute_task_chain":
            return "task_orchestrator"  # 新增：路由到任务编排器
        elif next_action == "rag_search":
            return "rag_agent"
        elif next_action == "chat":
            return "chat_agent"
        elif next_action == "product_search" and self.enable_business_agents:
            return "product_agent"
        elif next_action == "order_management" and self.enable_business_agents:
            return "order_agent"
        else:
            return "finish"
    
    def _route_after_agent(self, state: MultiAgentState) -> str:
        """Agent执行后的路由决策"""
        if state.get("error_message") or state.get("iteration_count", 0) >= self.max_iterations:
            return "finish"
        
        # 任务链模式：继续执行任务链
        if state.get("task_chain") and state.get("next_action") == "execute_task_chain":
            if state.get("current_agent") in ["product_agent", "order_agent"]:
                return "task_orchestrator"
        
        # RAG降级：答案质量低时切换到Chat Agent
        current_agent = state.get("current_agent")
        if current_agent == "rag_agent":
            rag_result = state.get("agent_results", {}).get("rag_agent")
            if rag_result:
                answer = rag_result.get("answer", "")
                if (rag_result.get("answer_quality", 0.0) < 0.5 or 
                    not answer or "无法从知识库中找到" in answer):
                    agent_names = [r.get("agent") for r in state.get("agent_history", [])]
                    if "chat_agent" not in agent_names:
                        return "chat_agent"
        
        return "finish"

    def _route_after_orchestrator(self, state: MultiAgentState) -> str:
        """
        Task Orchestrator后的路由决策

        根据任务编排器的执行结果，路由到下一个Agent或暂停/结束。

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        next_action = state.get("next_action", "finish")

        # 如果需要等待用户选择，暂停graph
        if next_action == "wait_for_selection":
            logger.info("等待用户选择，暂停graph")
            return "wait_for_selection"

        # 路由到product_agent
        elif next_action == "product_search" and self.enable_business_agents:
            logger.info("任务链路由到 product_agent")
            return "product_agent"

        # 路由到order_agent
        elif next_action == "order_management" and self.enable_business_agents:
            logger.info("任务链路由到 order_agent")
            return "order_agent"

        # 默认结束
        else:
            logger.info(f"任务链结束: next_action={next_action}")
            return "finish"

    def invoke(self, question: str, config: Optional[Dict[str, Any]] = None) -> MultiAgentState:
        """
        执行查询（同步接口，内部使用异步执行）
        
        企业级最佳实践：
        - 提供同步接口以保持向后兼容
        - 内部使用异步执行以提高性能
        - 使用asyncio.run()在同步上下文中运行异步代码
        
        Args:
            question: 用户问题
            config: 执行配置
            
        Returns:
            最终状态
        """
        import asyncio
        
        # 在同步方法中运行异步代码
        return asyncio.run(self.ainvoke(question, config))
    
    async def ainvoke(
        self,
        question: str,
        config: Optional[Dict[str, Any]] = None,
        session_id: str = "default"
    ) -> MultiAgentState:
        """
        异步执行查询（生产环境推荐）

        企业级最佳实践：
        - 使用异步接口充分利用异步性能优势
        - 支持高并发场景
        - 避免事件循环管理问题

        Args:
            question: 用户问题
            config: 执行配置
            session_id: 用户会话 ID，用于确认机制

        Returns:
            最终状态
        """
        # 创建初始状态
        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=question)],
            "current_agent": None,
            "agent_results": {},
            "agent_history": [],
            "tools_used": [],
            "metadata": {},
            "error_message": None,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "next_action": None,
            "routing_reason": None
        }

        # 执行图（使用异步API）
        if config is None:
            config = {}

        # 设置 recursion_limit 和 session_id
        config.setdefault("recursion_limit", self.max_iterations * 2)
        config.setdefault("configurable", {})
        config["configurable"]["session_id"] = session_id

        final_state = await self.graph.ainvoke(initial_state, config=config)
        return final_state
    
    def stream(self, question: str, config: Optional[Dict[str, Any]] = None):
        """
        流式执行查询（同步接口，内部使用异步执行）
        
        企业级最佳实践：
        - 提供同步接口以保持向后兼容
        - 内部使用异步执行以提高性能
        
        Args:
            question: 用户问题
            config: 执行配置
            
        Yields:
            状态更新
        """
        import asyncio
        
        # 创建初始状态
        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=question)],
            "current_agent": None,
            "agent_results": {},
            "agent_history": [],
            "tools_used": [],
            "metadata": {},
            "error_message": None,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "next_action": None,
            "routing_reason": None
        }
        
        # 流式执行（使用异步API，在同步上下文中运行）
        if config is None:
            config = {"recursion_limit": self.max_iterations * 2}
        
        # 使用asyncio.run()运行异步流
        async def _async_stream():
            async for state_update in self.graph.astream(initial_state, config=config):
                yield state_update
        
        # 在同步方法中运行异步生成器
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_gen = _async_stream()
            while True:
                try:
                    yield loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
    
    async def astream(
        self,
        question: str,
        config: Optional[Dict[str, Any]] = None,
        stream_mode: str = "updates",
        session_id: str = "default"
    ):
        """
        异步流式执行查询（生产环境推荐）

        企业级最佳实践：
        - 使用异步接口充分利用异步性能优势
        - 支持高并发场景

        Args:
            question: 用户问题
            config: 执行配置
            stream_mode: 流式模式，"updates" 返回节点更新，"values" 返回完整状态
            session_id: 用户会话 ID，用于确认机制

        Yields:
            状态更新
        """
        # 创建初始状态
        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=question)],
            "current_agent": None,
            "agent_results": {},
            "agent_history": [],
            "tools_used": [],
            "metadata": {},
            "error_message": None,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "next_action": None,
            "routing_reason": None,
            "task_chain": None,
            "pending_selection": None,
            "context_data": {}
        }

        # 尝试从存储中加载已有的任务链（用于继续多步骤任务）
        storage = get_task_chain_storage()
        existing_task_chain = storage.get(session_id)
        if existing_task_chain:
            logger.info(f"从存储中加载任务链: session={session_id}, chain_id={existing_task_chain.get('chain_id')}")
            initial_state["task_chain"] = existing_task_chain
            initial_state["context_data"] = existing_task_chain.get("context_data", {})

        # 流式执行（使用异步API）
        if config is None:
            config = {}

        # 设置 recursion_limit 和 session_id
        config.setdefault("recursion_limit", self.max_iterations * 2)
        config.setdefault("configurable", {})
        config["configurable"]["session_id"] = session_id

        async for state_update in self.graph.astream(initial_state, config=config, stream_mode=stream_mode):
            yield state_update

