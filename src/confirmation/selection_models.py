"""选择机制数据模型

定义选择机制的核心数据结构，类似于确认机制的models.py。
用于处理需要用户从多个选项中选择的场景，如产品选择、地址选择等。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional


class SelectionStatus(str, Enum):
    """选择状态枚举"""
    PENDING = "pending"  # 等待用户选择
    SELECTED = "selected"  # 用户已选择
    CANCELLED = "cancelled"  # 用户已取消
    EXPIRED = "expired"  # 已过期


@dataclass
class SelectionAction:
    """选择操作数据模型

    用于存储一个待用户选择的操作，包含选择的选项列表和相关元数据。

    Attributes:
        selection_id: 选择操作的唯一标识符
        session_id: 用户会话标识
        selection_type: 选择类型（如 "product", "address"）
        options: 可选项列表，每个选项是一个字典（通常包含id、name等字段）
        display_message: 展示给用户的提示消息
        metadata: 额外的元数据（如关联的task_chain_id等）
        status: 选择状态
        created_at: 创建时间
        expires_at: 过期时间
        selected_option_id: 用户选择的选项ID（仅在selected状态下有值）
    """
    selection_id: str
    session_id: str
    selection_type: str
    options: List[Dict[str, Any]]
    display_message: str
    metadata: Optional[Dict[str, Any]] = None
    status: SelectionStatus = SelectionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    selected_option_id: Optional[str] = None


@dataclass
class SelectionResult:
    """选择结果数据模型

    用于返回选择操作的执行结果。

    Attributes:
        selection_id: 选择操作的唯一标识符
        status: 最终状态
        selection_type: 选择类型
        selected_option: 用户选择的选项完整数据（如果selected状态）
        metadata: 原始metadata
        error: 错误信息（如果有）
    """
    selection_id: str
    status: SelectionStatus
    selection_type: str
    selected_option: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
