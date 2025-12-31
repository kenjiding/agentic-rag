"""基础Agent抽象类 - 2025-2026 企业级最佳实践

本模块定义了所有Agent的基础接口，确保统一的Agent实现规范。
所有Agent都应该继承BaseAgent并实现必要的方法。
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from src.multi_agent.state import MultiAgentState
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Agent基础抽象类
    
    所有Agent都应该继承此类并实现execute方法。
    此类提供了统一的接口和通用功能。
    
    2025-2026 最佳实践：
    - 使用抽象基类定义统一接口
    - 支持依赖注入（LLM、工具等）
    - 提供统一的错误处理机制
    - 支持日志和监控
    
    Attributes:
        name: Agent名称，用于标识和路由
        llm: 语言模型实例
        description: Agent功能描述，用于Supervisor路由决策
    """
    
    def __init__(
        self,
        name: str,
        llm: Optional[ChatOpenAI] = None,
        description: str = ""
    ):
        """
        初始化Agent
        
        Args:
            name: Agent名称，必须唯一
            llm: 语言模型实例，如果为None则使用默认模型
            description: Agent功能描述，用于Supervisor理解何时使用此Agent
        """
        self.name = name
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.description = description
    
    @abstractmethod
    async def execute(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        执行Agent的核心逻辑
        
        这是所有Agent必须实现的方法。它接收当前状态，执行Agent的特定任务，
        并返回更新后的状态信息。
        
        Args:
            state: 当前的多Agent系统状态
            
        Returns:
            包含更新信息的字典，通常包括：
            - result: Agent执行结果
            - messages: 新增的消息（如果有）
            - metadata: 额外的元数据
            
        Raises:
            Exception: 如果执行过程中出现错误
        """
        pass
    
    def get_name(self) -> str:
        """获取Agent名称"""
        return self.name
    
    def get_description(self) -> str:
        """获取Agent描述"""
        return self.description
    
    def validate_state(self, state: MultiAgentState) -> bool:
        """
        验证状态是否有效
        
        Args:
            state: 要验证的状态
            
        Returns:
            如果状态有效返回True，否则返回False
        """
        return hasattr(state, "messages") and isinstance(state.messages, list)


class ToolEnabledAgent(BaseAgent):
    """支持工具的Agent基类
    
    所有需要使用工具的Agent都应该继承此类。
    提供统一的工具访问接口和权限管理。
    
    2025-2026 企业级最佳实践：
    - 统一的工具访问接口
    - 工具权限检查
    - 工具使用监控
    - 工具列表缓存
    """
    
    def __init__(
        self,
        name: str,
        llm=None,
        description: str = "",
        tool_registry=None,
        tool_categories: Optional[List[str]] = None,
        tool_tags: Optional[List[str]] = None
    ):
        """
        初始化支持工具的Agent
        
        Args:
            name: Agent名称
            llm: 语言模型
            description: Agent描述
            tool_registry: 工具注册表
            tool_categories: 允许使用的工具类别
            tool_tags: 允许使用的工具标签
        """
        super().__init__(name=name, llm=llm, description=description)
        self.tool_registry = tool_registry
        self.tool_categories = tool_categories
        self.tool_tags = tool_tags
        self._available_tools = None  # 缓存可用工具
    
    def get_available_tools(self) -> List:
        """
        获取Agent可用的工具列表
        
        Returns:
            可用工具列表
        """
        if self._available_tools is None and self.tool_registry:
            self._available_tools = self.tool_registry.get_tools_for_agent(
                agent_name=self.name,
                categories=self.tool_categories,
                tags=self.tool_tags
            )
        return self._available_tools or []
    
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        调用工具（带权限检查）
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
            
        Raises:
            ValueError: 如果工具注册表未配置
        """
        if not self.tool_registry:
            raise ValueError("工具注册表未配置")
        
        return await self.tool_registry.acall_tool(
            tool_name=tool_name,
            agent_name=self.name,
            **kwargs
        )
    
    def refresh_tools(self):
        """刷新可用工具列表（当工具注册表更新时调用）"""
        self._available_tools = None
        logger.debug(f"Agent {self.name} 的工具列表已刷新")

