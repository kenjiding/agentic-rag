"""Agent注册表 - 企业级多Agent系统核心组件

本模块实现了Agent注册表模式，用于集中管理Agent的元数据和路由配置。
解决硬编码路由映射问题，实现自动化的图构建。

2025-2026 最佳实践：
- 注册表模式：集中管理Agent元数据
- 描述符模式：统一描述Agent的能力和路由配置
- 自动路由生成：根据配置自动生成LangGraph路由映射
- 配置文件支持：支持YAML配置驱动
- 开闭原则：新增Agent只需注册，无需修改现有代码
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional, Set
from pathlib import Path
from langgraph.graph import END

from src.multi_agent.constants import SystemNodeName

logger = logging.getLogger(__name__)


@dataclass
class RouteConfig:
    """路由配置

    定义Agent完成执行后的路由行为。

    Attributes:
        can_handoff_to: 可以切换到的Agent名称列表
        on_complete: 完成后的默认行为，默认为"finish"结束流程
        allow_escalation: 是否允许升级到chat_agent（降级策略）
    """
    can_handoff_to: List[str] = field(default_factory=list)
    on_complete: str = "finish"
    allow_escalation: bool = False

    def get_exit_mapping(self) -> Dict[str, Any]:
        """生成LangGraph add_conditional_edges使用的路由映射"""
        mapping = {}
        for target in self.can_handoff_to:
            mapping[target] = target
        
        # 支持"supervisor"作为路由目标（一步一步智能模式）
        mapping["supervisor"] = "supervisor"

        # 支持plan_executor作为路由目标（plan-driven智能模式）
        mapping[SystemNodeName.PLAN_EXECUTOR.value] = SystemNodeName.PLAN_EXECUTOR.value

        # 支持post_action_verifier作为路由目标（plan-driven智能模式）
        mapping[SystemNodeName.POST_ACTION_VERIFIER.value] = SystemNodeName.POST_ACTION_VERIFIER.value
        
        if self.on_complete == "finish":
            mapping["finish"] = END
        elif self.on_complete == "supervisor":
            # on_complete为"supervisor"时，supervisor已在上面添加，只需要确保finish也存在
            mapping["finish"] = END
        else:
            mapping[self.on_complete] = self.on_complete
        return mapping


@dataclass
class AgentDescriptor:
    """Agent描述符 - 统一描述Agent的元数据和路由配置

    封装Agent的所有元信息，包括名称、描述、节点函数、路由配置等。
    用于注册表管理和自动图构建。

    Attributes:
        name: Agent名称，必须唯一
        description: Agent功能描述，用于Supervisor路由决策
        node: LangGraph节点函数，接收state返回state更新
        enabled: 是否启用此Agent
        priority: 优先级，数值越大优先级越高
        route_config: 路由配置
        metadata: 额外的元数据
    """
    name: str
    description: str
    node: Callable
    enabled: bool = True
    priority: int = 0
    route_config: RouteConfig = field(default_factory=RouteConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后验证"""
        if not self.name:
            raise ValueError("AgentDescriptor.name 不能为空")
        if not callable(self.node):
            raise ValueError(f"AgentDescriptor.node 必须是可调用对象: {self.name}")

    def get_summary(self) -> str:
        """获取Agent摘要信息"""
        status = "enabled" if self.enabled else "disabled"
        return f"[{status}] {self.name} (priority={self.priority})"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "priority": self.priority,
            "route_config": {
                "can_handoff_to": self.route_config.can_handoff_to,
                "on_complete": self.route_config.on_complete,
                "allow_escalation": self.route_config.allow_escalation,
            },
            "metadata": self.metadata
        }


class AgentRegistry:
    """Agent注册表 - 企业级多Agent系统的核心

    职责：
    1. 注册和管理Agent描述符
    2. 提供Agent查询接口
    3. 自动生成LangGraph路由配置
    4. 支持运行时启用/禁用Agent
    5. 支持从配置文件加载

    设计原则：
    - 单一职责：只负责Agent注册和路由生成
    - 开闭原则：新增Agent无需修改注册表代码
    - 依赖倒置：依赖抽象（AgentDescriptor）而非具体Agent类

    使用示例：
        registry = AgentRegistry()

        # 注册Agent
        registry.register(AgentDescriptor(
            name="rag_agent",
            description="知识检索Agent",
            node=rag_agent_node,
            route_config=RouteConfig(
                can_handoff_to=["chat_agent"],
                allow_escalation=True
            )
        ))

        # 获取路由配置
        supervisor_routes = registry.build_supervisor_routes()
        agent_routes = registry.build_agent_exit_routes("rag_agent")
    """

    # 特殊路由名称常量
    ROUTE_FINISH = "finish"
    ROUTE_SUPERVISOR = "supervisor"

    def __init__(self):
        """初始化注册表"""
        self._agents: Dict[str, AgentDescriptor] = {}
        self._node_names: Set[str] = set()
        logger.info("AgentRegistry 初始化完成")

    def register(self, descriptor: AgentDescriptor) -> None:
        """注册Agent

        Args:
            descriptor: Agent描述符

        Raises:
            ValueError: 如果Agent名称已存在
        """
        if descriptor.name in self._agents:
            raise ValueError(f"Agent '{descriptor.name}' 已注册，请使用不同的名称")

        self._agents[descriptor.name] = descriptor
        self._node_names.add(descriptor.name)
        logger.info(f"注册Agent: {descriptor.get_summary()}")

    def unregister(self, name: str) -> bool:
        """注销Agent

        Args:
            name: Agent名称

        Returns:
            是否成功注销
        """
        if name in self._agents:
            del self._agents[name]
            self._node_names.discard(name)
            logger.info(f"注销Agent: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[AgentDescriptor]:
        """获取Agent描述符

        Args:
            name: Agent名称

        Returns:
            Agent描述符，如果不存在返回None
        """
        return self._agents.get(name)

    def is_enabled(self, name: str) -> bool:
        """检查Agent是否启用

        Args:
            name: Agent名称

        Returns:
            是否启用
        """
        descriptor = self.get(name)
        return descriptor is not None and descriptor.enabled

    def enable(self, name: str) -> bool:
        """启用Agent

        Args:
            name: Agent名称

        Returns:
            是否成功启用
        """
        descriptor = self.get(name)
        if descriptor:
            descriptor.enabled = True
            logger.info(f"启用Agent: {name}")
            return True
        return False

    def disable(self, name: str) -> bool:
        """禁用Agent

        Args:
            name: Agent名称

        Returns:
            是否成功禁用
        """
        descriptor = self.get(name)
        if descriptor:
            descriptor.enabled = False
            logger.info(f"禁用Agent: {name}")
            return True
        return False

    def get_enabled_agents(self, sort_by_priority: bool = True) -> List[AgentDescriptor]:
        """获取所有启用的Agent

        Args:
            sort_by_priority: 是否按优先级排序

        Returns:
            启用的Agent描述符列表
        """
        agents = [a for a in self._agents.values() if a.enabled]
        if sort_by_priority:
            agents.sort(key=lambda x: x.priority, reverse=True)
        return agents

    def get_all_agents(self) -> List[AgentDescriptor]:
        """获取所有Agent（包括禁用的）"""
        return list(self._agents.values())

    def get_node_names(self) -> List[str]:
        """获取所有节点名称（用于图构建）"""
        return list(self._node_names)

    def get_agent_names(self) -> List[str]:
        """获取所有Agent名称"""
        return list(self._agents.keys())

    def get_descriptions_text(self) -> str:
        """获取所有Agent的描述文本（用于Supervisor提示）

        Returns:
            格式化的描述文本
        """
        lines = []
        for agent in self.get_enabled_agents():
            handoff_info = ""
            if agent.route_config.can_handoff_to:
                handoff_info = f" [可切换到: {', '.join(agent.route_config.can_handoff_to)}]"
            lines.append(f"- {agent.name}: {agent.description}{handoff_info}")
        return "\n".join(lines)

    def build_supervisor_routes(self) -> Dict[str, Any]:
        """构建Supervisor后的路由映射

        生成LangGraph add_conditional_edges使用的路由字典。
        所有启用的Agent都可以被Supervisor路由到。

        Returns:
            路由映射字典，格式: {"agent_name": "agent_name", "finish": END}
        """
        routes = {self.ROUTE_FINISH: END}
        for agent in self.get_enabled_agents():
            routes[agent.name] = agent.name
        return routes

    def build_agent_exit_routes(self, agent_name: str) -> Dict[str, Any]:
        """构建Agent完成后的路由映射

        根据Agent的RouteConfig生成退出路由。

        Args:
            agent_name: Agent名称

        Returns:
            路由映射字典
        """
        descriptor = self.get(agent_name)
        if not descriptor:
            logger.warning(f"Agent '{agent_name}' 未找到，使用默认路由")
            return {"finish": END}

        return descriptor.route_config.get_exit_mapping()

    def get_agent_for_node(self, node_name: str) -> Optional[AgentDescriptor]:
        """根据节点名称获取对应的Agent描述符

        Args:
            node_name: 节点名称

        Returns:
            Agent描述符，如果不存在返回None
        """
        return self.get(node_name)

    def load_from_config(self, config: Dict[str, Any]) -> None:
        """从配置字典加载Agent配置

        配置格式：
            {
                "agents": [
                    {
                        "name": "rag_agent",
                        "description": "知识检索",
                        "enabled": true,
                        "priority": 10,
                        "route_config": {
                            "can_handoff_to": ["chat_agent"],
                            "on_complete": "finish",
                            "allow_escalation": true
                        }
                    }
                ]
            }

        Args:
            config: 配置字典

        Note:
            此方法只配置元数据，节点函数需要后续通过register_node关联
        """
        for agent_config in config.get("agents", []):
            name = agent_config.get("name")
            if not name:
                logger.warning("配置中缺少name字段，跳过")
                continue

            route_config_dict = agent_config.get("route_config", {})
            route_config = RouteConfig(
                can_handoff_to=route_config_dict.get("can_handoff_to", []),
                on_complete=route_config_dict.get("on_complete", "finish"),
                allow_escalation=route_config_dict.get("allow_escalation", False)
            )

            # 创建占位描述符（节点函数后续设置）
            descriptor = AgentDescriptor(
                name=name,
                description=agent_config.get("description", ""),
                node=lambda state: {},  # 占位函数
                enabled=agent_config.get("enabled", True),
                priority=agent_config.get("priority", 0),
                route_config=route_config,
                metadata=agent_config.get("metadata", {})
            )

            self.register(descriptor)
            logger.info(f"从配置加载Agent: {name}")

    def update_node(self, name: str, node: Callable) -> bool:
        """更新Agent的节点函数

        用于在从配置加载描述符后，关联实际的节点函数。

        Args:
            name: Agent名称
            node: 节点函数

        Returns:
            是否成功更新
        """
        descriptor = self.get(name)
        if descriptor:
            descriptor.node = node
            logger.info(f"更新Agent节点函数: {name}")
            return True
        logger.warning(f"Agent '{name}' 未找到，无法更新节点函数")
        return False

    def get_registry_summary(self) -> Dict[str, Any]:
        """获取注册表摘要信息

        Returns:
            包含注册表状态的字典
        """
        enabled = [a for a in self._agents.values() if a.enabled]
        disabled = [a for a in self._agents.values() if not a.enabled]

        return {
            "total_agents": len(self._agents),
            "enabled_count": len(enabled),
            "disabled_count": len(disabled),
            "enabled_agents": [a.name for a in enabled],
            "disabled_agents": [a.name for a in disabled],
        }

    def __len__(self) -> int:
        """获取已注册Agent数量"""
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        """检查Agent是否已注册"""
        return name in self._agents

    def __iter__(self):
        """迭代所有Agent描述符"""
        return iter(self._agents.values())


def load_config_from_yaml(file_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置

    Args:
        file_path: YAML文件路径

    Returns:
        配置字典
    """
    import yaml

    path = Path(file_path)
    if not path.exists():
        logger.warning(f"配置文件不存在: {file_path}")
        return {}

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"从YAML文件加载配置: {file_path}")
    return config


def create_registry_from_config(
    config_path: str,
    node_getter: callable
) -> AgentRegistry:
    """从配置文件创建注册表

    Args:
        config_path: 配置文件路径
        node_getter: 根据agent_name获取节点函数的回调

    Returns:
        配置好的AgentRegistry实例

    Example:
        def get_node(agent_name: str):
            if agent_name == "rag_agent":
                return graph_handler.rag_agent_node
            # ...

        registry = create_registry_from_config("config/agents.yaml", get_node)
    """
    config = load_config_from_yaml(config_path)
    registry = AgentRegistry()

    if not config:
        return registry

    for agent_config in config.get("agents", []):
        name = agent_config.get("name")
        if not name:
            continue

        route_config_dict = agent_config.get("route_config", {})
        route_config = RouteConfig(
            can_handoff_to=route_config_dict.get("can_handoff_to", []),
            on_complete=route_config_dict.get("on_complete", "finish"),
            allow_escalation=route_config_dict.get("allow_escalation", False)
        )

        node = node_getter(name)
        if node is None:
            logger.warning(f"无法获取节点函数: {name}，跳过注册")
            continue

        descriptor = AgentDescriptor(
            name=name,
            description=agent_config.get("description", ""),
            node=node,
            enabled=agent_config.get("enabled", True),
            priority=agent_config.get("priority", 0),
            route_config=route_config,
            metadata=agent_config.get("metadata", {})
        )

        registry.register(descriptor)

    logger.info(f"从配置文件创建注册表: {len(registry)} 个Agent")
    return registry
