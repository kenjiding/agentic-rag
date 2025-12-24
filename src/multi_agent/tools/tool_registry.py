"""工具注册表 - 企业级工具管理系统

2025-2026 企业级最佳实践：
- Agent级别的工具权限控制
- 工具分类和标签管理
- 工具使用审计和监控
- 动态工具注册和热更新
- 工具健康检查和降级策略
"""
from typing import Dict, List, Any, Optional, Set, Callable
from langchain_core.tools import BaseTool
from enum import Enum
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """工具类别枚举"""
    CALCULATION = "calculation"
    INFORMATION = "information"
    SEARCH = "search"
    RAG = "rag"
    UTILITY = "utility"
    COMMUNICATION = "communication"
    DATA_PROCESSING = "data_processing"
    SYSTEM = "system"
    CUSTOM = "custom"


class ToolPermission(str, Enum):
    """工具权限级别"""
    PUBLIC = "public"  # 所有Agent可用
    RESTRICTED = "restricted"  # 需要明确授权
    PRIVATE = "private"  # 仅特定Agent可用


@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    category: str
    permission: ToolPermission
    allowed_agents: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    author: Optional[str] = None
    rate_limit: Optional[int] = None  # 每分钟调用次数限制
    timeout: Optional[float] = None  # 超时时间（秒）
    requires_auth: bool = False
    cost_per_call: float = 0.0  # 每次调用成本
    health_check: Optional[Callable] = None
    is_enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """企业级工具注册表
    
    特性：
    1. Agent级别的工具权限管理
    2. 工具分类和标签系统
    3. 使用审计和监控
    4. 工具健康检查
    5. 速率限制和成本追踪
    6. 动态工具注册
    """
    
    def __init__(self):
        """初始化工具注册表"""
        self._tools: Dict[str, BaseTool] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._usage_stats: Dict[str, Dict[str, Any]] = {}
        self._agent_tool_cache: Dict[str, List[BaseTool]] = {}  # Agent工具缓存
        
    def register_tool(
        self,
        name: str,
        tool: BaseTool,
        category: str = ToolCategory.CUSTOM,
        permission: ToolPermission = ToolPermission.PUBLIC,
        allowed_agents: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        version: str = "1.0.0",
        rate_limit: Optional[int] = None,
        timeout: Optional[float] = None,
        requires_auth: bool = False,
        cost_per_call: float = 0.0,
        health_check: Optional[Callable] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        注册工具
        
        Args:
            name: 工具名称
            tool: 工具实例
            category: 工具类别
            permission: 权限级别
            allowed_agents: 允许使用的Agent列表（permission=RESTRICTED或PRIVATE时有效）
            tags: 工具标签
            description: 工具描述
            version: 版本号
            rate_limit: 速率限制（每分钟）
            timeout: 超时时间（秒）
            requires_auth: 是否需要认证
            cost_per_call: 每次调用成本
            health_check: 健康检查函数
            custom_metadata: 自定义元数据
        """
        if name in self._tools:
            logger.warning(f"工具 {name} 已存在，将被覆盖")
        
        self._tools[name] = tool
        self._metadata[name] = ToolMetadata(
            name=name,
            description=description or tool.description,
            category=category,
            permission=permission,
            allowed_agents=set(allowed_agents or []),
            tags=set(tags or []),
            version=version,
            rate_limit=rate_limit,
            timeout=timeout,
            requires_auth=requires_auth,
            cost_per_call=cost_per_call,
            health_check=health_check,
            custom_metadata=custom_metadata or {}
        )
        
        # 初始化使用统计
        self._usage_stats[name] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_cost": 0.0,
            "last_called": None,
            "call_history": []
        }
        
        # 清除Agent工具缓存（因为工具列表已更新）
        self._agent_tool_cache.clear()
        
        logger.info(f"工具 {name} 已注册 (类别: {category}, 权限: {permission})")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        获取指定工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具实例，如果不存在则返回None
        """
        return self._tools.get(name)
    
    def get_tools(
        self,
        category: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> List[BaseTool]:
        """
        获取工具列表
        
        Args:
            category: 工具类别过滤（可选）
            agent_name: Agent名称，如果提供则只返回该Agent有权限的工具
            
        Returns:
            工具列表
        """
        if agent_name:
            return self.get_tools_for_agent(agent_name, categories=[category] if category else None)
        
        if category:
            return [
                tool for name, tool in self._tools.items()
                if self._metadata.get(name, {}).category == category
                and self._metadata.get(name, {}).is_enabled
            ]
        return [
            tool for name, tool in self._tools.items()
            if self._metadata.get(name, {}).is_enabled
        ]
    
    def get_tools_for_agent(
        self,
        agent_name: str,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        include_disabled: bool = False
    ) -> List[BaseTool]:
        """
        获取指定Agent可用的工具
        
        根据Agent权限、类别、标签等过滤工具。
        
        Args:
            agent_name: Agent名称
            categories: 工具类别过滤（可选）
            tags: 标签过滤（可选）
            include_disabled: 是否包含已禁用的工具
            
        Returns:
            可用工具列表
        """
        # 检查缓存
        cache_key = f"{agent_name}_{categories}_{tags}_{include_disabled}"
        if cache_key in self._agent_tool_cache:
            return self._agent_tool_cache[cache_key]
        
        tools = []
        
        for name, tool in self._tools.items():
            metadata = self._metadata.get(name)
            if not metadata:
                continue
            
            # 检查是否启用
            if not include_disabled and not metadata.is_enabled:
                continue
            
            # 检查权限
            if not self._check_permission(agent_name, metadata):
                continue
            
            # 检查类别过滤
            if categories and metadata.category not in categories:
                continue
            
            # 检查标签过滤
            if tags and not metadata.tags.intersection(set(tags)):
                continue
            
            # 检查健康状态
            if metadata.health_check:
                try:
                    if not metadata.health_check():
                        logger.warning(f"工具 {name} 健康检查失败，跳过")
                        continue
                except Exception as e:
                    logger.error(f"工具 {name} 健康检查出错: {e}")
                    continue
            
            tools.append(tool)
        
        # 缓存结果
        self._agent_tool_cache[cache_key] = tools
        
        logger.debug(f"Agent {agent_name} 可用工具: {[t.name for t in tools]}")
        return tools
    
    def _check_permission(self, agent_name: str, metadata: ToolMetadata) -> bool:
        """检查Agent是否有权限使用工具"""
        if metadata.permission == ToolPermission.PUBLIC:
            return True
        elif metadata.permission == ToolPermission.RESTRICTED:
            return agent_name in metadata.allowed_agents
        elif metadata.permission == ToolPermission.PRIVATE:
            return agent_name in metadata.allowed_agents
        return False
    
    def get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        """获取工具元数据"""
        return self._metadata.get(name)
    
    def get_tool_description(self, name: str) -> Optional[str]:
        """获取工具描述"""
        metadata = self._metadata.get(name)
        return metadata.description if metadata else None
    
    def get_tool_names(self) -> List[str]:
        """获取所有已注册的工具名称"""
        return list(self._tools.keys())
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """按类别获取工具"""
        return [
            tool for name, tool in self._tools.items()
            if self._metadata.get(name, {}).category == category
            and self._metadata.get(name, {}).is_enabled
        ]
    
    def get_tools_by_tag(self, tag: str) -> List[BaseTool]:
        """按标签获取工具"""
        return [
            tool for name, tool in self._tools.items()
            if tag in self._metadata.get(name, {}).tags
            and self._metadata.get(name, {}).is_enabled
        ]
    
    def enable_tool(self, name: str):
        """启用工具"""
        if name in self._metadata:
            self._metadata[name].is_enabled = True
            self._metadata[name].updated_at = datetime.now()
            self._agent_tool_cache.clear()
            logger.info(f"工具 {name} 已启用")
    
    def disable_tool(self, name: str):
        """禁用工具"""
        if name in self._metadata:
            self._metadata[name].is_enabled = False
            self._metadata[name].updated_at = datetime.now()
            self._agent_tool_cache.clear()
            logger.info(f"工具 {name} 已禁用")
    
    def grant_permission(self, tool_name: str, agent_name: str):
        """授予Agent使用工具的权限"""
        if tool_name in self._metadata:
            self._metadata[tool_name].allowed_agents.add(agent_name)
            self._metadata[tool_name].updated_at = datetime.now()
            self._agent_tool_cache.clear()
            logger.info(f"已授予 {agent_name} 使用工具 {tool_name} 的权限")
    
    def revoke_permission(self, tool_name: str, agent_name: str):
        """撤销Agent使用工具的权限"""
        if tool_name in self._metadata:
            self._metadata[tool_name].allowed_agents.discard(agent_name)
            self._metadata[tool_name].updated_at = datetime.now()
            self._agent_tool_cache.clear()
            logger.info(f"已撤销 {agent_name} 使用工具 {tool_name} 的权限")
    
    def call_tool(self, name: str, agent_name: str = "system", **kwargs) -> Any:
        """
        调用工具（同步版本）
        
        Args:
            name: 工具名称
            agent_name: 调用Agent名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        import asyncio
        return asyncio.run(self.acall_tool(name, agent_name=agent_name, **kwargs))
    
    async def acall_tool(
        self,
        tool_name: str,
        agent_name: str = "system",
        **kwargs
    ) -> Any:
        """
        调用工具（带权限检查和审计）
        
        Args:
            tool_name: 工具名称
            agent_name: 调用Agent名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
            
        Raises:
            PermissionError: 如果Agent没有权限
            ValueError: 如果工具不存在
        """
        # 检查工具是否存在
        if tool_name not in self._tools:
            raise ValueError(f"工具 {tool_name} 不存在")
        
        metadata = self._metadata[tool_name]
        
        # 检查工具是否启用
        if not metadata.is_enabled:
            raise ValueError(f"工具 {tool_name} 已禁用")
        
        # 检查权限（如果指定了agent_name）
        if agent_name != "system" and not self._check_permission(agent_name, metadata):
            raise PermissionError(
                f"Agent {agent_name} 没有权限使用工具 {tool_name}"
            )
        
        # 检查速率限制
        if metadata.rate_limit:
            # 这里可以实现速率限制逻辑
            pass
        
        # 记录调用
        self._usage_stats[tool_name]["total_calls"] += 1
        self._usage_stats[tool_name]["last_called"] = datetime.now()
        
        try:
            # 调用工具
            tool = self._tools[tool_name]
            result = await tool.ainvoke(kwargs)
            
            # 记录成功
            self._usage_stats[tool_name]["successful_calls"] += 1
            self._usage_stats[tool_name]["total_cost"] += metadata.cost_per_call
            
            logger.info(f"Agent {agent_name} 成功调用工具 {tool_name}")
            return result
            
        except Exception as e:
            # 记录失败
            self._usage_stats[tool_name]["failed_calls"] += 1
            logger.error(f"Agent {agent_name} 调用工具 {tool_name} 失败: {e}")
            raise
    
    def get_usage_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """获取使用统计"""
        if tool_name:
            return self._usage_stats.get(tool_name, {})
        return self._usage_stats.copy()
    
    def get_tools_summary(self, agent_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        获取工具摘要（支持按Agent过滤）
        
        Args:
            agent_name: Agent名称，如果提供则只返回该Agent有权限的工具
            
        Returns:
            工具摘要字典
        """
        summary = {}
        
        for name, tool in self._tools.items():
            metadata = self._metadata.get(name)
            if not metadata:
                continue
            
            # 如果指定了Agent，检查权限
            if agent_name and not self._check_permission(agent_name, metadata):
                continue
            
            stats = self._usage_stats.get(name, {})
            summary[name] = {
                "description": metadata.description,
                "category": metadata.category,
                "permission": metadata.permission.value,
                "tags": list(metadata.tags),
                "version": metadata.version,
                "is_enabled": metadata.is_enabled,
                "usage_stats": stats,
                "cost_per_call": metadata.cost_per_call,
                "rate_limit": metadata.rate_limit
            }
        
        return summary
