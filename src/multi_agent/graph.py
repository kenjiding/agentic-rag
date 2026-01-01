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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.multi_agent.state import MultiAgentState
from src.multi_agent.supervisor import SupervisorAgent
from src.intent import IntentClassifier
from src.multi_agent.agents.base_agent import BaseAgent
from src.multi_agent.agents.rag_agent import RAGAgent
from src.multi_agent.agents.chat_agent import ChatAgent
from src.multi_agent.agents.product_agent import ProductAgent
from src.multi_agent.agents.order_agent import OrderAgent
from src.multi_agent.tools.tool_registry import ToolRegistry
from src.multi_agent.graph_nodes import GraphNodeHandler
from src.multi_agent.graph_routers import GraphRouter
from src.multi_agent.graph_state_manager import GraphStateManager
from src.multi_agent.graph_tool_initializer import GraphToolInitializer
import logging

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
            enable_intent_classification: 是否启用意图识别（默认True）
            enable_business_agents: 是否启用业务Agent（商品、订单），默认True
        """
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations
        self.enable_intent_classification = enable_intent_classification
        self.enable_business_agents = enable_business_agents

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
        
        # 初始化默认Agents（如果未提供）
        if agents is None:
            agents = self._create_default_agents(rag_persist_directory)
        
        # 注册所有Agents并自动分配工具注册表
        self._register_agents(agents)

        # 初始化 Checkpointer
        self.checkpointer = MemorySaver()
        logger.info(f"已初始化 MemorySaver checkpointer（内存存储）: id={id(self.checkpointer)}")

        # 初始化处理器
        self.node_handler = GraphNodeHandler(self)
        self.router = GraphRouter(self)
        self.state_manager = GraphStateManager(self)

        # 构建图
        self.graph = self._build_graph()
    
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
        
        return agents
    
    def _register_agents(self, agents: List[BaseAgent]):
        """注册所有Agents并自动分配工具注册表"""
        for agent in agents:
            self.supervisor.register_agent(agent)
            setattr(self, agent.get_name(), agent)
            
            # 自动为ToolEnabledAgent类型的Agent分配工具注册表
            if hasattr(agent, 'tool_registry') and agent.tool_registry is None:
                agent.tool_registry = self.tool_registry
                if hasattr(agent, 'refresh_tools'):
                    agent.refresh_tools()
                logger.info(f"已为Agent {agent.get_name()} 分配工具注册表")
    
    async def async_init_web_search_tools(self):
        """异步初始化web search tools"""
        await self.tool_initializer.async_init_web_search_tools(
            self.supervisor.get_registered_agents()
        )
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph工作流"""
        graph = StateGraph(MultiAgentState)

        # 添加节点
        graph.add_node("intent_recognition", self.node_handler.intent_recognition_node)
        graph.add_node("supervisor", self.node_handler.supervisor_node)
        graph.add_node("task_orchestrator", self.node_handler.task_orchestrator_node)
        graph.add_node("rag_agent", self.node_handler.rag_agent_node)
        graph.add_node("chat_agent", self.node_handler.chat_agent_node)

        if self.enable_business_agents:
            graph.add_node("product_agent", self.node_handler.product_agent_node)
            graph.add_node("order_agent", self.node_handler.order_agent_node)

        # 设置入口点
        graph.set_entry_point("intent_recognition")
        graph.add_edge("intent_recognition", "supervisor")
        
        # Supervisor后的条件路由
        route_mapping = {
            "rag_agent": "rag_agent",
            "chat_agent": "chat_agent",
            "task_orchestrator": "task_orchestrator",
            "finish": END
        }
        if self.enable_business_agents:
            route_mapping["product_agent"] = "product_agent"
            route_mapping["order_agent"] = "order_agent"

        graph.add_conditional_edges(
            "supervisor",
            self.router.route_after_supervisor,
            route_mapping
        )

        # Task Orchestrator 条件路由
        orchestrator_route_mapping = {
            "product_agent": "product_agent",
            "order_agent": "order_agent",
            "finish": END
        }
        graph.add_conditional_edges(
            "task_orchestrator",
            self.router.route_after_orchestrator,
            orchestrator_route_mapping
        )
        
        # Agent执行后的条件路由
        agent_route_mapping_rag = {
            "supervisor": "supervisor",
            "chat_agent": "chat_agent",
            "finish": END
        }
        graph.add_conditional_edges("rag_agent", self.router.route_after_agent, agent_route_mapping_rag)
        
        agent_route_mapping_chat = {
            "supervisor": "supervisor",
            "finish": END
        }
        graph.add_conditional_edges("chat_agent", self.router.route_after_agent, agent_route_mapping_chat)

        if self.enable_business_agents:
            agent_route_mapping_product = {
                "task_orchestrator": "task_orchestrator",
                "supervisor": "supervisor",
                "finish": END
            }
            graph.add_conditional_edges("product_agent", self.router.route_after_agent, agent_route_mapping_product)
            
            agent_route_mapping_order = {
                "task_orchestrator": "task_orchestrator",
                "supervisor": "supervisor",
                "finish": END
            }
            graph.add_conditional_edges("order_agent", self.router.route_after_agent, agent_route_mapping_order)
        
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
