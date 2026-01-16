"""Multi-Agent Graph - 多Agent系统主图（一步一步智能模式）

本模块使用LangGraph构建多Agent系统的核心工作流。
实现了Supervisor模式，协调多个Agent的执行。

2025-2026 最佳实践：
- 使用LangGraph 1.x最新API
- 一步一步智能模式：每次请求都重新进行意图识别和路由决策
- 通过 entities 字段存储上下文信息，实现多轮对话状态管理
- Supervisor 根据 entities 智能路由到对应 Agent
- 清晰的状态管理
- 错误处理和重试机制
- 注册表模式：Agent注册和路由配置自动化
"""
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.language_models import BaseChatModel

from src.multi_agent.state import MultiAgentState
from src.utils.llm_factory import create_llm_for_agent
from src.multi_agent.supervisor import SupervisorAgent
from src.intent import IntentClassifier
from src.multi_agent.agents.base_agent import BaseAgent
from src.multi_agent.agents.rag_agent import RAGAgent
from src.multi_agent.agents.chat_agent import ChatAgent
from src.multi_agent.agents.product_agent import ProductAgent
from src.multi_agent.agents.order_agent import OrderAgent
from src.multi_agent.agents.consultation_agent import ConsultationAgent
from src.multi_agent.tools.tool_registry import ToolRegistry
from src.multi_agent.agent_registry import (
    AgentRegistry,
    AgentDescriptor,
    RouteConfig,
    load_config_from_yaml
)
from src.multi_agent.graph_nodes import GraphNodeHandler
from src.multi_agent.graph_routers import GraphRouter
from src.multi_agent.graph_state_manager import GraphStateManager
from src.multi_agent.graph_tool_initializer import GraphToolInitializer
import logging

logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_AGENTS_CONFIG_PATH = "config/agents.yaml"


class MultiAgentGraph:
    """多Agent系统主图（一步一步智能模式）

    职责：
    1. 初始化所有Agent和工具
    2. 构建LangGraph工作流（使用注册表模式）
    3. 管理状态流转
    4. 协调Agent执行

    架构：
    - 意图识别节点：分析用户意图并提取实体
    - Supervisor节点：根据 entities 智能路由到对应 Agent
    - RAG Agent节点：知识检索
    - Chat Agent节点：一般对话
    - Product Agent节点：商品搜索
    - Order Agent节点：订单管理（含确认机制）

    一步一步智能模式设计：
    - 每次请求都重新进行意图识别和路由决策
    - 通过 entities 字段存储上下文信息
    - Supervisor 根据 entities 智能路由到对应 Agent
    - 不依赖预先定义的任务链

    注册表模式：
    - 所有Agent注册到AgentRegistry
    - 路由配置自动生成
    - 新增Agent只需注册，无需修改图构建代码
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        agents: Optional[List[BaseAgent]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        rag_persist_directory: str = "./tmp/chroma_db/agentic_rag",
        max_iterations: int = 10,
        init_web_search: bool = True,
        enable_intent_classification: bool = True,
        enable_business_agents: bool = True,
        agents_config_path: Optional[str] = None
    ):
        """
        初始化多Agent图

        Args:
            llm: 语言模型实例，如果为None则使用工厂函数创建默认模型
            agents: 自定义Agent列表，如果为None则使用默认Agent
            tool_registry: 工具注册表
            rag_persist_directory: RAG向量数据库持久化目录
            max_iterations: 最大迭代次数
            init_web_search: 是否在初始化时加载web search tools（默认True）
            enable_intent_classification: 是否启用意图识别（默认True）
            enable_business_agents: 是否启用业务Agent（商品、订单），默认True
            agents_config_path: Agent配置文件路径，默认为config/agents.yaml
        """
        self.llm = llm or create_llm_for_agent()
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations
        self.enable_intent_classification = enable_intent_classification
        self.enable_business_agents = enable_business_agents
        self.agents_config_path = agents_config_path or DEFAULT_AGENTS_CONFIG_PATH

        # 初始化工具初始化器
        self.tool_initializer = GraphToolInitializer(self.tool_registry)
        if init_web_search:
            try:
                self.tool_initializer.init_web_search_tools()
            except Exception as e:
                logger.warning(f"Web search tools初始化失败，系统将在没有web search功能的情况下运行: {e}")
        else:
            logger.info("跳过web search tools初始化，可在需要时调用async_init_web_search_tools()异步加载")

        # 初始化Supervisor
        self.supervisor = SupervisorAgent(llm=self.llm)

        # 初始化意图分类器
        self.intent_classifier = IntentClassifier(llm=self.llm) if self.enable_intent_classification else None

        # 初始化上下文管理器
        from src.multi_agent.context_manager import ContextManager
        from src.multi_agent.context_pipeline import ContextPipeline
        self.context_manager = ContextManager(
            max_history_rounds=5,
            max_tool_calls=10
        )
        self.context_pipeline = ContextPipeline(self.context_manager)
        logger.info("已初始化ContextManager（上下文管理器）")

        # 初始化Agent注册表
        self.agent_registry = AgentRegistry()

        # 将注册表设置到Supervisor（用于获取Agent描述）
        self.supervisor.set_agent_registry(self.agent_registry)

        # 初始���处理器（需要在_register_agent_descriptors之前）
        self.node_handler = GraphNodeHandler(self)
        self.router = GraphRouter(self)
        self.state_manager = GraphStateManager(self)

        # 初始化默认Agents（如果未提供）
        if agents is None:
            agents = self._create_default_agents(rag_persist_directory)

        # 注册所有Agents到Supervisor和设置实例属性
        self._register_agents(agents)

        # 从配置文件加载Agent描述并注册到注册表（需要node_handler已初始化）
        self._register_agent_descriptors()

        # 初始化 Checkpointer
        self.checkpointer = MemorySaver()
        logger.info(f"已初始化 MemorySaver checkpointer（内存存储）: id={id(self.checkpointer)}")

        # 构建图
        self.graph = self._build_graph()

        # 输出注册表摘要
        summary = self.agent_registry.get_registry_summary()
        logger.info(f"Agent注册表摘要: {summary}")

    def _create_default_agents(self, rag_persist_directory: str) -> List[BaseAgent]:
        """创建默认Agent列表"""
        agents = []

        # 添加RAG Agent
        agents.append(RAGAgent(llm=self.llm, persist_directory=rag_persist_directory))

        # 添加Chat Agent
        agents.append(ChatAgent(llm=self.llm, tool_registry=self.tool_registry))

        # 添加业务 Agent
        if self.enable_business_agents:
            agents.append(ProductAgent(llm=self.llm))
            agents.append(OrderAgent(llm=self.llm))
            agents.append(ConsultationAgent(llm=self.llm))

        return agents

    def _register_agents(self, agents: List[BaseAgent]):
        """注册所有Agents到Supervisor并设置实例属性"""
        for agent in agents:
            self.supervisor.register_agent(agent)
            setattr(self, agent.get_name(), agent)

            # 自动为ToolEnabledAgent类型的Agent分配工具注册表
            if hasattr(agent, 'tool_registry') and agent.tool_registry is None:
                agent.tool_registry = self.tool_registry
                if hasattr(agent, 'refresh_tools'):
                    agent.refresh_tools()
                logger.info(f"已为Agent {agent.get_name()} 分配工具注册表")

    def _register_agent_descriptors(self):
        """从配置文件加载Agent描述并注册到注册表

        如果配置文件不存在，使用默认配置注册所有已创建的Agent。
        """
        # 尝试从配置文件加载
        config = load_config_from_yaml(self.agents_config_path)

        if config:
            # 从配置文件注册
            for agent_config in config.get("agents", []):
                agent_name = agent_config.get("name")
                agent_enabled = agent_config.get("enabled", True)

                # 检查Agent实例是否存在
                if not hasattr(self, agent_name):
                    if not agent_enabled:
                        logger.info(f"Agent {agent_name} 已禁用且未实例化，跳过")
                        continue
                    logger.warning(f"Agent {agent_name} 未实例化，跳过注册描述符")
                    continue

                # 获取Agent实例
                agent = getattr(self, agent_name)

                # 构建路由配置
                route_config_dict = agent_config.get("route_config", {})
                route_config = RouteConfig(
                    can_handoff_to=route_config_dict.get("can_handoff_to", []),
                    on_complete=route_config_dict.get("on_complete", "finish"),
                    allow_escalation=route_config_dict.get("allow_escalation", False)
                )

                # 创建描述符并注册
                descriptor = AgentDescriptor(
                    name=agent_name,
                    description=agent_config.get("description", agent.get_description()),
                    node=self.node_handler.create_agent_node(agent_name),
                    enabled=agent_enabled,
                    priority=agent_config.get("priority", 0),
                    route_config=route_config,
                    metadata=agent_config.get("metadata", {})
                )

                self.agent_registry.register(descriptor)
                logger.info(f"从配置注册Agent描述符: {agent_name}")

            # 更新全局路由配置
            routing_config = config.get("routing", {})
            if "max_iterations" in routing_config:
                self.max_iterations = routing_config["max_iterations"]
                logger.info(f"从配置更新max_iterations: {self.max_iterations}")

        else:
            # 配置文件不存在，使用默认配置注册
            logger.info("配置文件不存在，使用默认配置注册Agent描述符")
            self._register_default_descriptors()

    def _register_default_descriptors(self):
        """使用默认配置注册所有Agent描述符"""
        # RAG Agent
        self.agent_registry.register(AgentDescriptor(
            name="rag_agent",
            description="知识检索专家 - 从向量数据库中检索相关信息并生成答案",
            node=self.node_handler.create_agent_node("rag_agent"),
            route_config=RouteConfig(
                can_handoff_to=["chat_agent"],
                on_complete="finish",
                allow_escalation=True
            ),
            priority=10
        ))

        # Chat Agent
        self.agent_registry.register(AgentDescriptor(
            name="chat_agent",
            description="通用对话助手 - 处理一般性对话、问候、感谢等交流场景",
            node=self.node_handler.create_agent_node("chat_agent"),
            route_config=RouteConfig(
                can_handoff_to=[],
                on_complete="finish"
            ),
            priority=0
        ))

        # 业务Agents
        if self.enable_business_agents:
            # Product Agent
            self.agent_registry.register(AgentDescriptor(
                name="product_agent",
                description="商品搜索专家 - 处理商品查询、搜索、比价等请求",
                node=self.node_handler.create_agent_node("product_agent"),
                route_config=RouteConfig(
                    can_handoff_to=[],
                    on_complete="finish"
                ),
                priority=20
            ))

            # Order Agent
            self.agent_registry.register(AgentDescriptor(
                name="order_agent",
                description="订单管理专家 - 处理订单查询、取消、创建等操作",
                node=self.node_handler.create_agent_node("order_agent"),
                route_config=RouteConfig(
                    can_handoff_to=[],
                    on_complete="finish"
                ),
                priority=20
            ))

    async def async_init_web_search_tools(self):
        """异步初始化web search tools"""
        await self.tool_initializer.async_init_web_search_tools(
            self.supervisor.get_registered_agents()
        )

    def _setup_graph_routes(self, graph: StateGraph):
        """设置图的路由配置

        使用AgentRegistry自动生成所有路由配置。

        Args:
            graph: LangGraph StateGraph实例
        """
        # Supervisor后的条件路由
        supervisor_routes = self.agent_registry.build_supervisor_routes()
        graph.add_conditional_edges(
            "supervisor",
            self.router.route_after_supervisor,
            supervisor_routes
        )
        logger.info(f"配置Supervisor路由: {list(supervisor_routes.keys())}")

        # 每个Agent的退出路由
        for agent_name in self.agent_registry.get_node_names():
            exit_routes = self.agent_registry.build_agent_exit_routes(agent_name)
            graph.add_conditional_edges(
                agent_name,
                self.router.route_after_agent,
                exit_routes
            )
            logger.info(f"配置{agent_name}退出路由: {list(exit_routes.keys())}")

    def _build_graph(self) -> StateGraph:
        """构建LangGraph工作流（一步一步智能模式）

        使用AgentRegistry自动生成图配置，新增Agent无需修改此方法。
        """
        graph = StateGraph(MultiAgentState)

        # 添加系统节点
        graph.add_node("context_manager", self.node_handler.context_manager_node)  # 新增：上下文管理节点
        graph.add_node("intent_recognition", self.node_handler.intent_recognition_node)
        graph.add_node("supervisor", self.node_handler.supervisor_node)

        # 自动添加所有Agent节点
        for descriptor in self.agent_registry.get_enabled_agents():
            graph.add_node(descriptor.name, descriptor.node)
            logger.info(f"添加Agent节点: {descriptor.name}")

        # 设置入口点（修改：从context_manager开始）
        graph.set_entry_point("context_manager")
        graph.add_edge("context_manager", "intent_recognition")
        graph.add_edge("intent_recognition", "supervisor")

        # 自动配置所有路由
        self._setup_graph_routes(graph)

        return graph.compile(checkpointer=self.checkpointer)

    def invoke(self, question: str, config: Optional[Dict[str, Any]] = None) -> MultiAgentState:
        """执行查询（同步接口，内部使用异步执行）"""
        return asyncio.run(self.ainvoke(question, config))

    async def ainvoke(
        self,
        question: str,
        config: Optional[Dict[str, Any]] = None,
        session_id: str = "default"
    ) -> MultiAgentState:
        """异步执行查询（生产环境推荐）"""
        initial_state = self.state_manager.create_initial_state(question)
        config = self.state_manager.prepare_config(config, session_id)
        final_state = await self.graph.ainvoke(initial_state, config=config)
        return final_state

    def stream(self, question: str, config: Optional[Dict[str, Any]] = None):
        """流式执行查询（同步接口，内部使用异步执行）"""
        initial_state = self.state_manager.create_initial_state(question)
        if config is None:
            config = {"recursion_limit": self.max_iterations * 2}

        async def _async_stream():
            async for state_update in self.graph.astream(initial_state, config=config):
                yield state_update

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
        question: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        stream_mode: str = "updates",
        session_id: str = "default",
        command: Optional[Command] = None
    ):
        """异步流式执行查询（生产环境推荐）"""
        config = self.state_manager.prepare_config(config, session_id)
        logger.info(f"[astream] 设置 config: session_id={session_id}, thread_id={session_id}")

        # 处理 Command(resume=...) 机制
        if command is not None:
            self.state_manager.log_resume_state(config, command)
            logger.info(f"[恢复执行] 开始调用 graph.astream(command, ...)")
            async for state_update in self.graph.astream(command, config=config, stream_mode=stream_mode):
                yield state_update
            logger.info(f"[恢复执行] graph.astream 完成")
            return

        # 获取初始状态（从checkpointer恢复或创建新状态）
        initial_state = self.state_manager.get_initial_state_for_stream(question, config, session_id)

        async for state_update in self.graph.astream(initial_state, config=config, stream_mode=stream_mode):
            yield state_update
