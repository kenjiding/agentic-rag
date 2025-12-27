"""确认机制数据模型"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ConfirmationStatus(str, Enum):
    """确认状态枚举"""
    PENDING = "pending"       # 等待确认
    CONFIRMED = "confirmed"   # 已确认
    CANCELLED = "cancelled"   # 已取消
    EXPIRED = "expired"       # 已过期


class ConfirmationAction(BaseModel):
    """待确认操作

    存储一个需要用户确认的操作的所有信息
    """

    # 唯一标识
    confirmation_id: str = Field(..., description="唯一确认 ID")
    session_id: str = Field(..., description="所属会话 ID")

    # 操作详情
    action_type: str = Field(..., description="操作类型，如 'cancel_order', 'create_order'")
    action_data: Dict[str, Any] = Field(default_factory=dict, description="执行操作所需的参数")

    # 上下文
    agent_name: str = Field(..., description="发起确认的 Agent 名称")
    display_message: str = Field(..., description="展示给用户的确认消息")
    display_data: Optional[Dict[str, Any]] = Field(default=None, description="用于 UI 展示的结构化数据")

    # 状态
    status: ConfirmationStatus = Field(default=ConfirmationStatus.PENDING)

    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(default=None, description="过期时间")
    resolved_at: Optional[datetime] = Field(default=None, description="解决时间")

    class Config:
        use_enum_values = True


class ConfirmationResult(BaseModel):
    """确认解决结果

    确认被处理后的结果信息
    """

    confirmation_id: str
    status: ConfirmationStatus
    action_type: str
    action_data: Dict[str, Any]
    execution_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        use_enum_values = True


class ConfirmationRequest(BaseModel):
    """创建确认请求的参数"""

    session_id: str
    action_type: str
    action_data: Dict[str, Any]
    agent_name: str
    display_message: str
    display_data: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = None
