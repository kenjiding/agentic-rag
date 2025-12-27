"""确认机制存储实现"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
from datetime import datetime, timedelta
import asyncio
import logging

from .models import ConfirmationAction, ConfirmationStatus

logger = logging.getLogger(__name__)


class ConfirmationStorage(ABC):
    """确认存储抽象基类

    定义存储接口，支持不同的存储后端实现
    """

    @abstractmethod
    async def save(self, confirmation: ConfirmationAction) -> None:
        """保存确认操作"""
        pass

    @abstractmethod
    async def get(self, confirmation_id: str) -> Optional[ConfirmationAction]:
        """根据 ID 获取确认"""
        pass

    @abstractmethod
    async def get_pending_by_session(self, session_id: str) -> Optional[ConfirmationAction]:
        """获取会话的待确认操作（每个会话同时只能有一个）"""
        pass

    @abstractmethod
    async def update_status(
        self,
        confirmation_id: str,
        status: ConfirmationStatus,
        resolved_at: Optional[datetime] = None
    ) -> bool:
        """更新确认状态"""
        pass

    @abstractmethod
    async def delete(self, confirmation_id: str) -> bool:
        """删除确认"""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """清理过期确认，返回清理数量"""
        pass


class InMemoryConfirmationStorage(ConfirmationStorage):
    """内存存储实现

    适用于单实例部署，重启后状态丢失
    """

    def __init__(self, default_ttl_seconds: int = 300):
        """初始化内存存储

        Args:
            default_ttl_seconds: 默认过期时间（秒），默认 5 分钟
        """
        self._store: Dict[str, ConfirmationAction] = {}
        self._session_index: Dict[str, str] = {}  # session_id -> confirmation_id
        self._lock = asyncio.Lock()
        self._default_ttl = timedelta(seconds=default_ttl_seconds)

    async def save(self, confirmation: ConfirmationAction) -> None:
        """保存确认操作

        如果会话已有待确认操作，将其标记为已取消
        """
        async with self._lock:
            # 设置过期时间（如果未设置）
            if confirmation.expires_at is None:
                confirmation.expires_at = datetime.utcnow() + self._default_ttl

            # 取消该会话之前的待确认操作
            existing_id = self._session_index.get(confirmation.session_id)
            if existing_id and existing_id in self._store:
                existing = self._store[existing_id]
                if existing.status == ConfirmationStatus.PENDING:
                    existing.status = ConfirmationStatus.CANCELLED
                    existing.resolved_at = datetime.utcnow()
                    logger.info(f"已取消会话 {confirmation.session_id} 之前的确认 {existing_id}")

            # 保存新确认
            self._store[confirmation.confirmation_id] = confirmation

            # 更新会话索引
            if confirmation.status == ConfirmationStatus.PENDING:
                self._session_index[confirmation.session_id] = confirmation.confirmation_id

            logger.info(f"已保存确认 {confirmation.confirmation_id}，会话 {confirmation.session_id}")

    async def get(self, confirmation_id: str) -> Optional[ConfirmationAction]:
        """根据 ID 获取确认"""
        return self._store.get(confirmation_id)

    async def get_pending_by_session(self, session_id: str) -> Optional[ConfirmationAction]:
        """获取会话的待确认操作"""
        async with self._lock:
            confirmation_id = self._session_index.get(session_id)
            if not confirmation_id:
                return None

            confirmation = self._store.get(confirmation_id)
            if not confirmation:
                # 清理无效索引
                del self._session_index[session_id]
                return None

            # 检查状态
            if confirmation.status != ConfirmationStatus.PENDING:
                del self._session_index[session_id]
                return None

            # 检查过期
            if confirmation.expires_at and datetime.utcnow() > confirmation.expires_at:
                confirmation.status = ConfirmationStatus.EXPIRED
                confirmation.resolved_at = datetime.utcnow()
                del self._session_index[session_id]
                logger.info(f"确认 {confirmation_id} 已过期")
                return None

            return confirmation

    async def update_status(
        self,
        confirmation_id: str,
        status: ConfirmationStatus,
        resolved_at: Optional[datetime] = None
    ) -> bool:
        """更新确认状态"""
        async with self._lock:
            confirmation = self._store.get(confirmation_id)
            if not confirmation:
                return False

            confirmation.status = status
            confirmation.resolved_at = resolved_at or datetime.utcnow()

            # 如果不再是 PENDING，从会话索引中移除
            if status != ConfirmationStatus.PENDING:
                if self._session_index.get(confirmation.session_id) == confirmation_id:
                    del self._session_index[confirmation.session_id]

            logger.info(f"确认 {confirmation_id} 状态更新为 {status}")
            return True

    async def delete(self, confirmation_id: str) -> bool:
        """删除确认"""
        async with self._lock:
            confirmation = self._store.get(confirmation_id)
            if not confirmation:
                return False

            # 清理索引
            if self._session_index.get(confirmation.session_id) == confirmation_id:
                del self._session_index[confirmation.session_id]

            del self._store[confirmation_id]
            logger.info(f"已删除确认 {confirmation_id}")
            return True

    async def cleanup_expired(self) -> int:
        """清理过期确认"""
        async with self._lock:
            now = datetime.utcnow()
            expired_ids = []

            for cid, confirmation in self._store.items():
                if confirmation.expires_at and now > confirmation.expires_at:
                    if confirmation.status == ConfirmationStatus.PENDING:
                        confirmation.status = ConfirmationStatus.EXPIRED
                        confirmation.resolved_at = now
                        expired_ids.append(cid)

            # 清理会话索引
            for session_id, cid in list(self._session_index.items()):
                if cid in expired_ids:
                    del self._session_index[session_id]

            if expired_ids:
                logger.info(f"已清理 {len(expired_ids)} 个过期确认")

            return len(expired_ids)

    def get_stats(self) -> Dict:
        """获取存储统计信息（用于调试）"""
        return {
            "total_confirmations": len(self._store),
            "active_sessions": len(self._session_index),
            "pending_count": sum(
                1 for c in self._store.values()
                if c.status == ConfirmationStatus.PENDING
            ),
        }
