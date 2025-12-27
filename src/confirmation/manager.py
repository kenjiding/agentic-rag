"""确认机制管理器

ConfirmationManager 是确认机制的核心协调器，负责：
1. 创建待确认操作
2. 存储和检索确认状态
3. 解析用户确认/取消
4. 执行已确认的操作
"""

from typing import Optional, Dict, Any, Callable, Awaitable
from datetime import datetime
import uuid
import logging

from .models import ConfirmationAction, ConfirmationStatus, ConfirmationResult
from .storage import ConfirmationStorage, InMemoryConfirmationStorage
from .exceptions import (
    ConfirmationNotFoundError,
    ConfirmationExpiredError,
    ConfirmationAlreadyResolvedError,
    ExecutorNotFoundError,
)
from src.multi_agent.config import get_keywords_config

logger = logging.getLogger(__name__)

# 类型别名：操作执行器
# 接收 (action_type, action_data) 返回执行结果
ActionExecutor = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


class ConfirmationManager:
    """确认机制管理器

    Usage:
        1. Agent 调用 request_confirmation() 请求用户确认
        2. 前端展示确认对话框
        3. 用户点击确认/取消，调用 resolve_confirmation()
        4. 执行器自动执行对应操作

    Example:
        ```python
        manager = ConfirmationManager()

        # 注册执行器
        manager.register_executor("cancel_order", cancel_order_handler)

        # 创建确认请求
        confirmation = await manager.request_confirmation(
            session_id="user-123",
            action_type="cancel_order",
            action_data={"order_id": 1},
            agent_name="order_agent",
            display_message="确认取消订单 #001？",
        )

        # 用户确认后
        result = await manager.resolve_confirmation(
            confirmation.confirmation_id,
            confirmed=True
        )
        ```
    """

    def __init__(
        self,
        storage: Optional[ConfirmationStorage] = None,
        default_ttl_seconds: int = 300,
    ):
        """初始化确认管理器

        Args:
            storage: 存储后端，默认使用内存存储
            default_ttl_seconds: 默认确认过期时间（秒）
        """
        self._storage = storage or InMemoryConfirmationStorage(default_ttl_seconds)
        self._executors: Dict[str, ActionExecutor] = {}
        self._default_ttl = default_ttl_seconds

        logger.info("ConfirmationManager 初始化完成")

    def register_executor(self, action_type: str, executor: ActionExecutor) -> None:
        """注册操作执行器

        Args:
            action_type: 操作类型标识
            executor: 异步执行函数

        Example:
            ```python
            async def cancel_order_handler(action_type: str, data: dict) -> dict:
                result = await cancel_order(data["order_id"])
                return {"success": True, "message": "订单已取消"}

            manager.register_executor("cancel_order", cancel_order_handler)
            ```
        """
        self._executors[action_type] = executor
        logger.info(f"已注册执行器: {action_type}")

    async def request_confirmation(
        self,
        session_id: str,
        action_type: str,
        action_data: Dict[str, Any],
        agent_name: str,
        display_message: str,
        display_data: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> ConfirmationAction:
        """创建确认请求

        Args:
            session_id: 用户会话标识
            action_type: 操作类型（如 "cancel_order", "create_order"）
            action_data: 执行操作所需的参数
            agent_name: 发起确认的 Agent 名称
            display_message: 展示给用户的确认消息
            display_data: 用于 UI 展示的结构化数据（如订单详情）
            ttl_seconds: 自定义过期时间

        Returns:
            ConfirmationAction 对象

        Note:
            每个会话同时只能有一个待确认操作，
            新请求会自动取消之前的待确认操作
        """
        from datetime import timedelta

        ttl = ttl_seconds or self._default_ttl

        confirmation = ConfirmationAction(
            confirmation_id=str(uuid.uuid4()),
            session_id=session_id,
            action_type=action_type,
            action_data=action_data,
            agent_name=agent_name,
            display_message=display_message,
            display_data=display_data,
            status=ConfirmationStatus.PENDING,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=ttl),
        )

        await self._storage.save(confirmation)

        logger.info(
            f"创建确认请求: id={confirmation.confirmation_id}, "
            f"session={session_id}, type={action_type}"
        )

        return confirmation

    async def get_pending_confirmation(self, session_id: str) -> Optional[ConfirmationAction]:
        """获取会话的待确认操作

        Args:
            session_id: 用户会话标识

        Returns:
            待确认操作，如果没有则返回 None
        """
        return await self._storage.get_pending_by_session(session_id)

    async def get_confirmation(self, confirmation_id: str) -> Optional[ConfirmationAction]:
        """根据 ID 获取确认

        Args:
            confirmation_id: 确认 ID

        Returns:
            确认操作，如果不存在则返回 None
        """
        return await self._storage.get(confirmation_id)

    async def resolve_confirmation(
        self,
        confirmation_id: str,
        confirmed: bool,
    ) -> ConfirmationResult:
        """解析确认

        Args:
            confirmation_id: 确认 ID
            confirmed: True 表示用户确认，False 表示用户取消

        Returns:
            ConfirmationResult 包含执行结果

        Raises:
            ConfirmationNotFoundError: 确认不存在
            ConfirmationExpiredError: 确认已过期
            ConfirmationAlreadyResolvedError: 确认已被处理
        """
        confirmation = await self._storage.get(confirmation_id)

        if not confirmation:
            raise ConfirmationNotFoundError(f"确认 {confirmation_id} 不存在")

        if confirmation.status != ConfirmationStatus.PENDING:
            raise ConfirmationAlreadyResolvedError(
                f"确认 {confirmation_id} 已被处理，状态: {confirmation.status}"
            )

        # 检查过期
        if confirmation.expires_at and datetime.utcnow() > confirmation.expires_at:
            await self._storage.update_status(
                confirmation_id,
                ConfirmationStatus.EXPIRED
            )
            raise ConfirmationExpiredError(f"确认 {confirmation_id} 已过期")

        # 用户取消
        if not confirmed:
            await self._storage.update_status(
                confirmation_id,
                ConfirmationStatus.CANCELLED
            )

            logger.info(f"用户取消确认: {confirmation_id}")

            return ConfirmationResult(
                confirmation_id=confirmation_id,
                status=ConfirmationStatus.CANCELLED,
                action_type=confirmation.action_type,
                action_data=confirmation.action_data,
            )

        # 用户确认，执行操作
        executor = self._executors.get(confirmation.action_type)

        if not executor:
            logger.error(f"未找到执行器: {confirmation.action_type}")

            await self._storage.update_status(
                confirmation_id,
                ConfirmationStatus.CONFIRMED
            )

            return ConfirmationResult(
                confirmation_id=confirmation_id,
                status=ConfirmationStatus.CONFIRMED,
                action_type=confirmation.action_type,
                action_data=confirmation.action_data,
                error=f"未注册执行器: {confirmation.action_type}",
            )

        try:
            logger.info(f"执行确认操作: {confirmation.action_type}, 数据: {confirmation.action_data}")

            execution_result = await executor(
                confirmation.action_type,
                confirmation.action_data
            )

            await self._storage.update_status(
                confirmation_id,
                ConfirmationStatus.CONFIRMED
            )

            logger.info(f"确认操作执行成功: {confirmation_id}")

            return ConfirmationResult(
                confirmation_id=confirmation_id,
                status=ConfirmationStatus.CONFIRMED,
                action_type=confirmation.action_type,
                action_data=confirmation.action_data,
                execution_result=execution_result,
            )

        except Exception as e:
            logger.error(f"执行确认操作失败: {confirmation_id}, 错误: {e}")

            await self._storage.update_status(
                confirmation_id,
                ConfirmationStatus.CONFIRMED  # 仍然标记为已确认，只是执行失败
            )

            return ConfirmationResult(
                confirmation_id=confirmation_id,
                status=ConfirmationStatus.CONFIRMED,
                action_type=confirmation.action_type,
                action_data=confirmation.action_data,
                error=str(e),
            )

    async def check_and_resolve_from_text(
        self,
        session_id: str,
        user_input: str,
    ) -> Optional[ConfirmationResult]:
        """检查并通过文本输入解析确认

        用于处理用户通过文本回复（如"确认"、"取消"）来响应确认请求

        Args:
            session_id: 用户会话标识
            user_input: 用户输入文本

        Returns:
            如果有待确认且用户输入为确认/取消响应，返回 ConfirmationResult
            否则返回 None（表示没有待确认或用户输入不是确认响应）
        """
        confirmation = await self.get_pending_confirmation(session_id)

        if not confirmation:
            return None

        # 解析用户输入
        confirmed = self._parse_user_confirmation(user_input)

        if confirmed is None:
            # 用户输入不是确认响应，可能是新的查询
            # 取消当前的待确认操作
            await self._storage.update_status(
                confirmation.confirmation_id,
                ConfirmationStatus.CANCELLED
            )
            logger.info(
                f"用户输入新查询，自动取消确认: {confirmation.confirmation_id}"
            )
            return None

        return await self.resolve_confirmation(
            confirmation.confirmation_id,
            confirmed
        )

    def _parse_user_confirmation(self, user_input: str) -> Optional[bool]:
        """解析用户输入是否为确认响应

        使用配置化的关键词列表，支持扩展和多语言。

        Args:
            user_input: 用户输入文本

        Returns:
            True: 用户确认
            False: 用户取消
            None: 无法判断（不是确认相关输入）
        """
        user_input_lower = user_input.strip().lower()
        keywords_config = get_keywords_config()

        # 检查确认关键词（使用配置化关键词）
        for keyword in keywords_config.confirm_yes_keywords:
            if keyword.lower() in user_input_lower:
                return True

        # 检查取消关键词（使用配置化关键词）
        for keyword in keywords_config.confirm_no_keywords:
            if keyword.lower() in user_input_lower:
                return False

        return None

    async def cancel_pending(self, session_id: str) -> bool:
        """取消会话的待确认操作

        Args:
            session_id: 用户会话标识

        Returns:
            是否成功取消
        """
        confirmation = await self.get_pending_confirmation(session_id)

        if not confirmation:
            return False

        await self._storage.update_status(
            confirmation.confirmation_id,
            ConfirmationStatus.CANCELLED
        )

        logger.info(f"已取消会话 {session_id} 的待确认操作")
        return True

    async def cleanup_expired(self) -> int:
        """清理过期确认

        Returns:
            清理的确认数量
        """
        return await self._storage.cleanup_expired()


# 单例实例
_confirmation_manager: Optional[ConfirmationManager] = None


def get_confirmation_manager() -> ConfirmationManager:
    """获取 ConfirmationManager 单例

    Returns:
        ConfirmationManager 实例
    """
    global _confirmation_manager

    if _confirmation_manager is None:
        _confirmation_manager = ConfirmationManager()

    return _confirmation_manager


def reset_confirmation_manager() -> None:
    """重置 ConfirmationManager 单例（用于测试）"""
    global _confirmation_manager
    _confirmation_manager = None
