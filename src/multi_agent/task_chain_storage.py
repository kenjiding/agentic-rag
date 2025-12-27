"""TaskChain 临时存储

用于跨请求保存任务链状态，使得用户选择后可以继续执行任务链。
"""

from typing import Optional, Dict
from datetime import datetime, timedelta
import logging
from src.multi_agent.state import TaskChain

logger = logging.getLogger(__name__)


class TaskChainStorage:
    """任务链临时存储（内存实现）

    使用 session_id 作为 key 存储任务链状态。
    支持 TTL 过期自动清理。
    """

    def __init__(self, default_ttl_seconds: int = 1800):  # 默认30分钟过期
        self._storage: Dict[str, TaskChain] = {}
        self._expires_at: Dict[str, datetime] = {}
        self._default_ttl = default_ttl_seconds
        logger.info("TaskChainStorage 初始化完成")

    def save(self, session_id: str, task_chain: TaskChain) -> None:
        """保存任务链

        Args:
            session_id: 会话 ID
            task_chain: 任务链对象
        """
        self._storage[session_id] = task_chain
        self._expires_at[session_id] = datetime.utcnow() + timedelta(seconds=self._default_ttl)
        logger.info(f"保存任务链: session={session_id}, chain_id={task_chain.get('chain_id')}")

    def get(self, session_id: str) -> Optional[TaskChain]:
        """获取任务链

        Args:
            session_id: 会话 ID

        Returns:
            任务链对象，如果不存在或已过期则返回 None
        """
        # 检查是否存在
        if session_id not in self._storage:
            return None

        # 检查是否过期
        expires_at = self._expires_at.get(session_id)
        if expires_at and datetime.utcnow() > expires_at:
            logger.info(f"任务链已过期: session={session_id}")
            self.delete(session_id)
            return None

        task_chain = self._storage[session_id]
        logger.info(f"获取任务链: session={session_id}, chain_id={task_chain.get('chain_id')}")
        return task_chain

    def delete(self, session_id: str) -> None:
        """删除任务链

        Args:
            session_id: 会话 ID
        """
        if session_id in self._storage:
            del self._storage[session_id]
        if session_id in self._expires_at:
            del self._expires_at[session_id]
        logger.info(f"删除任务链: session={session_id}")

    def cleanup_expired(self) -> int:
        """清理过期任务链

        Returns:
            清理的数量
        """
        now = datetime.utcnow()
        expired_sessions = [
            session_id
            for session_id, expires_at in self._expires_at.items()
            if expires_at and now > expires_at
        ]

        for session_id in expired_sessions:
            self.delete(session_id)

        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期任务链")

        return len(expired_sessions)


# 全局单例
_task_chain_storage: Optional[TaskChainStorage] = None


def get_task_chain_storage() -> TaskChainStorage:
    """获取 TaskChainStorage 单例

    Returns:
        TaskChainStorage 实例
    """
    global _task_chain_storage

    if _task_chain_storage is None:
        _task_chain_storage = TaskChainStorage()

    return _task_chain_storage


def reset_task_chain_storage() -> None:
    """重置 TaskChainStorage 单例（用于测试）"""
    global _task_chain_storage
    _task_chain_storage = None
