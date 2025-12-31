"""Chat Agent - 通用对话Agent

本模块实现了一个通用的对话Agent，可以使用工具进行任务处理。
适用于一般对话、工具调用等场景。
"""
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain.agents import create_agent
from src.multi_agent.agents.base_agent import BaseAgent, ToolEnabledAgent
from src.multi_agent.state import MultiAgentState
from src.multi_agent.tools.tool_registry import ToolRegistry
import logging

logger = logging.getLogger(__name__)


class ChatAgent(ToolEnabledAgent):
    """Chat Agent - 通用对话和工具调用Agent
    
    此Agent提供以下能力：
    1. 一般对话处理
    2. 工具调用（通过ToolRegistry）
    3. 任务执行
    
    使用场景：
    - 一般性对话
    - 需要调用工具的任务
    - 不需要RAG搜索的简单问题
    
    2025-2026 最佳实践：
    - 使用LangChain的create_agent创建工具调用能力
    - 支持动态工具注册
    - 统一的错误处理
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        tool_categories: Optional[List[str]] = None,
        tool_tags: Optional[List[str]] = None
    ):
        """
        初始化Chat Agent
        
        Args:
            llm: 语言模型实例
            tool_registry: 工具注册表，如果提供则Agent可以使用这些工具
            system_prompt: 系统提示词，如果为None则使用默认提示词
            tool_categories: 允许使用的工具类别（企业级功能）
            tool_tags: 允许使用的工具标签（企业级功能）
        """
        super().__init__(
            name="chat_agent",
            llm=llm,
            description="通用对话Agent，可以处理一般性对话和工具调用任务。",
            tool_registry=tool_registry,
            tool_categories=tool_categories,
            tool_tags=tool_tags
        )
        
        self.system_prompt = system_prompt or (
            "Your name is Novid Assistant. You are a professional, courteous, and helpful AI assistant.\n\n"
            "**Greeting Protocol:**\n"
            "When users greet you politely (e.g., '你好', 'Hello', 'Hi', '您好', '早上好', '下午好', '晚上好'), "
            "follow this sequence:\n"
            "1. First, introduce yourself: '您好，我是 Novid Assistant'\n"
            "2. Then, greet them back warmly: '很高兴为您服务' or 'How can I help you today?'\n"
            "3. Finally, address their question or offer assistance\n\n"
            "**Communication Style:**\n"
            "- Always be polite, respectful, and professional\n"
            "- Use courteous language (请, 谢谢, 不客气, 抱歉, etc.)\n"
            "- Show appreciation when users express gratitude\n"
            "- Apologize gracefully when you cannot help or make mistakes\n"
            "- Be patient and understanding with user requests\n\n"
            "**Core Capabilities:**\n"
            "- Answer questions and help with tasks\n"
            "- When RAG system cannot find answers, use available web search tools to find information from the internet\n"
            "- If previous RAG search failed or returned low-quality results, proactively use web search tools to find accurate answers\n"
            "- Provide clear, accurate, and helpful responses"
        )
        self._agent = None  # 延迟初始化
    
    def _get_agent(self):
        """获取或创建LangChain Agent实例"""
        if self._agent is None:
            # 使用ToolEnabledAgent的方法获取可用工具
            tools = self.get_available_tools()

            logger.info(f"ChatAgent 初始化，可用工具数量: {len(tools)}")
            if tools:
                logger.debug(f"可用工具: {[t.name for t in tools]}")

            # 使用 LangChain 1.x 推荐的 create_agent
            # system_prompt 参数可以是 str 或 SystemMessage
            self._agent = create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=self.system_prompt
            )
        return self._agent
    
    async def execute(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        执行对话处理
        
        Args:
            state: 当前的多Agent系统状态
            
        Returns:
            包含以下字段的字典：
            - result: Agent执行结果
            - messages: 新增的消息
            - metadata: 元数据
        """
        try:
            # 获取用户消息
            user_message = None
            for msg in reversed(state.messages):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break
            
            if not user_message:
                return {
                    "result": None,
                    "messages": [AIMessage(content="未找到用户消息")],
                    "metadata": {"error": "未找到用户消息"}
                }
            
            logger.info(f"Chat Agent处理消息: {user_message[:100]}...")
            
            # 检查是否有RAG结果，如果有且质量低，提示使用web search
            rag_result = state.agent_results.get("rag_agent")
            should_use_web_search = False
            if rag_result:
                answer_quality = rag_result.get("answer_quality", 0.0)
                if answer_quality < 0.5:
                    should_use_web_search = True
                    logger.info("检测到RAG答案质量低，Chat Agent将使用web search工具")
            
            # 使用LangChain Agent处理（支持工具调用）
            # LangChain 1.x 最佳实践：create_agent返回的agent可以直接处理状态
            agent = self._get_agent()
            
            # 构建配置
            from langgraph.graph.state import RunnableConfig
            config = RunnableConfig(
                configurable={"thread_id": state.metadata.get("session_id", "default")},
                recursion_limit=50
            )
            
            # LangChain 1.x 最佳实践：
            # create_agent返回的agent是一个Runnable，可以直接处理状态
            # 它会自动处理消息格式，无需手动清理
            try:
                # 构建agent输入状态
                # create_agent期望的输入格式：{"messages": [...]}
                agent_input = {"messages": state.messages}

                # 如果RAG失败，在系统提示中添加搜索引导
                if should_use_web_search:
                    # 更新系统提示，添加web search引导
                    from langchain_core.messages import SystemMessage
                    enhanced_system = self.system_prompt + "\n\n注意：由于之前的RAG搜索结果不佳，请使用web search工具从互联网获取最新信息。"
                    # 在消息列表开头添加增强的系统消息
                    agent_input["messages"] = [
                        SystemMessage(content=enhanced_system)
                    ] + [msg for msg in state.messages if isinstance(msg, HumanMessage)]
                
                # 执行Agent（create_agent返回的agent支持异步调用）
                if hasattr(agent, 'ainvoke'):
                    response = await agent.ainvoke(agent_input, config=config)
                elif hasattr(agent, 'invoke'):
                    # 如果没有异步方法，使用同步调用（在线程中执行）
                    import asyncio
                    response = await asyncio.to_thread(agent.invoke, agent_input, config=config)
                else:
                    raise ValueError("Agent不支持invoke或ainvoke方法")
                
                # create_agent返回的响应格式：包含messages键的字典
                # 处理响应格式
                if isinstance(response, dict):
                    # create_agent返回格式：{"messages": [...]}
                    if "messages" in response:
                        new_messages = response["messages"]
                    else:
                        new_messages = []
                else:
                    # 如果是其他格式（不应该发生）
                    logger.warning(f"Agent返回了意外的格式: {type(response)}")
                    new_messages = []

                # 找到最后一条AI消息
                ai_message = None
                for msg in reversed(new_messages):
                    if isinstance(msg, AIMessage):
                        ai_message = msg
                        break

                if not ai_message:
                    # 如果没有AI消息，使用LLM生成回复
                    logger.warning("Agent未返回AI消息，使用LLM生成回复")
                    llm_response = await self.llm.ainvoke([HumanMessage(content=user_message)])
                    ai_message = llm_response

            except Exception as agent_error:
                # 如果Agent调用失败，降级到直接使用LLM
                logger.warning(f"Agent调用失败: {agent_error}，降级到直接使用LLM", exc_info=True)
                llm_response = await self.llm.ainvoke([HumanMessage(content=user_message)])
                ai_message = llm_response
            
            # 获取可用工具数量
            available_tools = self.get_available_tools()
            tools_count = len(available_tools) if available_tools else 0
            
            result = {
                "result": {"response": ai_message.content},
                "messages": [ai_message],
                "metadata": {
                    "agent": self.name,
                    "tools_available": tools_count,
                    "tool_names": [t.name for t in available_tools] if available_tools else []
                }
            }
            
            logger.info("Chat Agent执行完成")
            return result
            
        except Exception as e:
            logger.error(f"Chat Agent执行错误: {str(e)}", exc_info=True)
            return {
                "result": None,
                "messages": [AIMessage(content=f"处理消息时出现错误: {str(e)}")],
                "metadata": {"error": str(e)},
                "error": str(e)
            }

