"""选择机制管理器

SelectionManager 是选择机制的核心协调器，负责：
1. 创建选择请求
2. 存储和检索选择状态
3. 解析用户选择
4. 管理选择生命周期

类似于 ConfirmationManager，但用于处理用户选择而非确认。
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import uuid
import logging

from .selection_models import SelectionAction, SelectionStatus, SelectionResult
from .selection_storage import SelectionStorage, InMemorySelectionStorage

logger = logging.getLogger(__name__)


class SelectionManager:
    """选择机制管理器

    Usage:
        ```python
        manager = SelectionManager()

        # 创建选择请求
        selection = await manager.request_selection(
            session_id="user-123",
            selection_type="product",
            options=[
                {"id": "1", "name": "产品A", "price": 100},
                {"id": "2", "name": "产品B", "price": 200}
            ],
            display_message="请选择要购买的产品:",
            metadata={"task_chain_id": "chain-123"}
        )

        # 用户选择后
        result = await manager.resolve_selection(
            selection_id=selection.selection_id,
            selected_option_id="1"
        )
        ```
    """

    def __init__(
        self,
        storage: Optional[SelectionStorage] = None,
        default_ttl_seconds: int = 300,
    ):
        """初始化选择管理器

        Args:
            storage: 存储后端，默认使用内存存储
            default_ttl_seconds: 默认选择过期时间（秒）
        """
        self._storage = storage or InMemorySelectionStorage(default_ttl_seconds)
        self._default_ttl = default_ttl_seconds

        logger.info("SelectionManager 初始化完成")

    async def request_selection(
        self,
        session_id: str,
        selection_type: str,
        options: List[Dict[str, Any]],
        display_message: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> SelectionAction:
        """创建选择请求

        Args:
            session_id: 用户会话标识
            selection_type: 选择类型（如 "product", "address"）
            options: 可选项列表，每个选项是一个字典（需包含唯一的id字段）
            display_message: 展示给用户的提示消息
            metadata: 额外的元数据（如关联的task_chain_id等）
            ttl_seconds: 自定义过期时间

        Returns:
            SelectionAction 对象

        Note:
            每个会话同时只能有一个待选择操作，
            新请求会自动取消之前的待选择操作
        """
        ttl = ttl_seconds or self._default_ttl

        selection = SelectionAction(
            selection_id=str(uuid.uuid4()),
            session_id=session_id,
            selection_type=selection_type,
            options=options,
            display_message=display_message,
            metadata=metadata,
            status=SelectionStatus.PENDING,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=ttl),
        )

        await self._storage.save(selection)

        logger.info(
            f"创建选择请求: id={selection.selection_id}, "
            f"session={session_id}, type={selection_type}, "
            f"options_count={len(options)}"
        )

        return selection

    async def get_pending_selection(self, session_id: str) -> Optional[SelectionAction]:
        """获取会话的待选择操作

        Args:
            session_id: 用户会话标识

        Returns:
            待选择操作，如果没有则返回 None
        """
        return await self._storage.get_pending_by_session(session_id)

    async def get_selection(self, selection_id: str) -> Optional[SelectionAction]:
        """根据 ID 获取选择

        Args:
            selection_id: 选择 ID

        Returns:
            选择操作，如果不存在则返回 None
        """
        return await self._storage.get(selection_id)

    async def resolve_selection(
        self,
        selection_id: str,
        selected_option_id: str,
    ) -> SelectionResult:
        """解析用户选择

        Args:
            selection_id: 选择 ID
            selected_option_id: 用户选择的选项ID

        Returns:
            SelectionResult 包含选择结果

        Raises:
            ValueError: 选择不存在、已过期、或选项ID无效
        """
        selection = await self._storage.get(selection_id)

        if not selection:
            raise ValueError(f"选择 {selection_id} 不存在")

        if selection.status != SelectionStatus.PENDING:
            raise ValueError(
                f"选择 {selection_id} 已被处理，状态: {selection.status}"
            )

        # 检查过期
        if selection.expires_at and datetime.utcnow() > selection.expires_at:
            await self._storage.update_status(
                selection_id,
                SelectionStatus.EXPIRED
            )
            raise ValueError(f"选择 {selection_id} 已过期")

        # 验证选项ID
        selected_option = None
        for option in selection.options:
            if str(option.get("id")) == str(selected_option_id):
                selected_option = option
                break

        if not selected_option:
            raise ValueError(
                f"无效的选项ID: {selected_option_id}，"
                f"可选ID: {[opt.get('id') for opt in selection.options]}"
            )

        # 更新状态
        await self._storage.update_status(
            selection_id,
            SelectionStatus.SELECTED,
            selected_option_id=str(selected_option_id)
        )

        logger.info(
            f"用户选择: selection={selection_id}, "
            f"option={selected_option_id}"
        )

        return SelectionResult(
            selection_id=selection_id,
            status=SelectionStatus.SELECTED,
            selection_type=selection.selection_type,
            selected_option=selected_option,
            metadata=selection.metadata,
        )

    async def cancel_selection(self, selection_id: str) -> SelectionResult:
        """取消选择

        Args:
            selection_id: 选择 ID

        Returns:
            SelectionResult

        Raises:
            ValueError: 选择不存在或已处理
        """
        selection = await self._storage.get(selection_id)

        if not selection:
            raise ValueError(f"选择 {selection_id} 不存在")

        if selection.status != SelectionStatus.PENDING:
            raise ValueError(
                f"选择 {selection_id} 已被处理，状态: {selection.status}"
            )

        await self._storage.update_status(
            selection_id,
            SelectionStatus.CANCELLED
        )

        logger.info(f"用户取消选择: {selection_id}")

        return SelectionResult(
            selection_id=selection_id,
            status=SelectionStatus.CANCELLED,
            selection_type=selection.selection_type,
            metadata=selection.metadata,
        )

    async def cancel_pending(self, session_id: str) -> bool:
        """取消会话的待选择操作

        Args:
            session_id: 用户会话标识

        Returns:
            是否成功取消
        """
        selection = await self.get_pending_selection(session_id)

        if not selection:
            return False

        await self._storage.update_status(
            selection.selection_id,
            SelectionStatus.CANCELLED
        )

        logger.info(f"已取消会话 {session_id} 的待选择操作")
        return True

    async def cleanup_expired(self) -> int:
        """清理过期选择

        Returns:
            清理的选择数量
        """
        return await self._storage.cleanup_expired()


# 单例实例
_selection_manager: Optional[SelectionManager] = None


def get_selection_manager() -> SelectionManager:
    """获取 SelectionManager 单例

    Returns:
        SelectionManager 实例
    """
    global _selection_manager

    if _selection_manager is None:
        _selection_manager = SelectionManager()

    return _selection_manager


def reset_selection_manager() -> None:
    """重置 SelectionManager 单例（用于测试）"""
    global _selection_manager
    _selection_manager = None
