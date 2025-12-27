"""选择机制存储层

提供选择操作的存储接口和实现，复用确认机制的存储模式。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
from datetime import datetime
import logging

from .selection_models import SelectionAction, SelectionStatus

logger = logging.getLogger(__name__)


class SelectionStorage(ABC):
    """选择存储抽象基类"""

    @abstractmethod
    async def save(self, selection: SelectionAction) -> None:
        """保存选择操作

        Args:
            selection: 选择操作对象

        Note:
            每个session只能有一个pending选择，
            新选择会自动取消之前的pending选择
        """
        pass

    @abstractmethod
    async def get(self, selection_id: str) -> Optional[SelectionAction]:
        """根据ID获取选择操作

        Args:
            selection_id: 选择ID

        Returns:
            选择操作对象，如果不存在则返回None
        """
        pass

    @abstractmethod
    async def get_pending_by_session(self, session_id: str) -> Optional[SelectionAction]:
        """获取会话的待选择操作

        Args:
            session_id: 会话ID

        Returns:
            待选择操作，如果没有则返回None
        """
        pass

    @abstractmethod
    async def update_status(
        self,
        selection_id: str,
        status: SelectionStatus,
        selected_option_id: Optional[str] = None
    ) -> None:
        """更新选择状态

        Args:
            selection_id: 选择ID
            status: 新状态
            selected_option_id: 选中的选项ID（仅在selected状态下）
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """清理过期选择

        Returns:
            清理的选择数量
        """
        pass


class InMemorySelectionStorage(SelectionStorage):
    """内存存储实现

    使用字典存储选择操作，适用于单机部署。
    对于分布式部署，建议使用Redis等外部存储。
    """

    def __init__(self, default_ttl_seconds: int = 300):
        """初始化内存存储

        Args:
            default_ttl_seconds: 默认过期时间（秒）
        """
        self._selections: Dict[str, SelectionAction] = {}
        self._session_index: Dict[str, str] = {}  # session_id -> selection_id
        self._default_ttl = default_ttl_seconds

        logger.info(f"InMemorySelectionStorage初始化完成，默认TTL: {default_ttl_seconds}秒")

    async def save(self, selection: SelectionAction) -> None:
        """保存选择操作"""
        # 取消该session之前的pending选择
        if selection.session_id in self._session_index:
            old_selection_id = self._session_index[selection.session_id]
            if old_selection_id in self._selections:
                old_selection = self._selections[old_selection_id]
                if old_selection.status == SelectionStatus.PENDING:
                    await self.update_status(old_selection_id, SelectionStatus.CANCELLED)
                    logger.info(f"自动取消旧选择: {old_selection_id}")

        # 保存新选择
        self._selections[selection.selection_id] = selection

        # 更新索引
        if selection.status == SelectionStatus.PENDING:
            self._session_index[selection.session_id] = selection.selection_id

        logger.info(
            f"保存选择: id={selection.selection_id}, "
            f"session={selection.session_id}, "
            f"type={selection.selection_type}"
        )

    async def get(self, selection_id: str) -> Optional[SelectionAction]:
        """根据ID获取选择操作"""
        return self._selections.get(selection_id)

    async def get_pending_by_session(self, session_id: str) -> Optional[SelectionAction]:
        """获取会话的待选择操作"""
        # 从索引查找
        selection_id = self._session_index.get(session_id)
        if not selection_id:
            return None

        selection = self._selections.get(selection_id)
        if not selection:
            return None

        # 确保是pending状态
        if selection.status != SelectionStatus.PENDING:
            return None

        # 检查是否过期
        if selection.expires_at and datetime.utcnow() > selection.expires_at:
            await self.update_status(selection_id, SelectionStatus.EXPIRED)
            return None

        return selection

    async def update_status(
        self,
        selection_id: str,
        status: SelectionStatus,
        selected_option_id: Optional[str] = None
    ) -> None:
        """更新选择状态"""
        selection = self._selections.get(selection_id)
        if not selection:
            logger.warning(f"选择不存在: {selection_id}")
            return

        selection.status = status
        if selected_option_id:
            selection.selected_option_id = selected_option_id

        # 如果状态不再是pending，从session索引中移除
        if status != SelectionStatus.PENDING:
            if selection.session_id in self._session_index:
                if self._session_index[selection.session_id] == selection_id:
                    del self._session_index[selection.session_id]

        logger.info(
            f"更新选择状态: id={selection_id}, "
            f"status={status}, "
            f"selected_option={selected_option_id}"
        )

    async def cleanup_expired(self) -> int:
        """清理过期选择"""
        now = datetime.utcnow()
        expired_ids = []

        for selection_id, selection in self._selections.items():
            if selection.expires_at and now > selection.expires_at:
                if selection.status == SelectionStatus.PENDING:
                    expired_ids.append(selection_id)

        # 更新状态
        for selection_id in expired_ids:
            await self.update_status(selection_id, SelectionStatus.EXPIRED)

        logger.info(f"清理过期选择: {len(expired_ids)}个")
        return len(expired_ids)
