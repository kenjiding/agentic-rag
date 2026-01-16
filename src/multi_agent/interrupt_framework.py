"""通用的 Interrupt 处理框架

LangGraph 1.x interrupt() 机制的通用封装：
- 支持多种 interrupt 场景（确认、选择、输入等）
- 使用枚举和类型化数据，避免魔术字符串
- 易于扩展新的 interrupt 类型

## 架构设计原则

1. **枚举优于魔术字符串**：使用 InterruptType 枚举
2. **类型化数据**：使用 Pydantic 模型验证数据
3. **单一职责**：每个 InterruptType 有对应的处理逻辑
4. **开闭原则**：新增类型不需要修改核心代码

## 使用方式

```python
# Agent 中需要 interrupt 时
from src.multi_agent.interrupt_framework import create_interrupt, InterruptType

interrupt_data = create_interrupt(
    InterruptType.CONFIRMATION,
    action_type="cancel_order",
    action_data={"order_id": 123},
    display_message="确认取消订单？"
)
return interrupt(interrupt_data)

# 恢复执行时
resume_value = {"confirmed": True}
```
"""
from enum import Enum
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class InterruptType(str, Enum):
    """Interrupt 类型枚举

    扩展方法：
    1. 添加新的枚举值
    2. 在 create_interrupt 中添加对应的创建逻辑
    3. 在 Agent 中处理 resume 值时判断类型
    """
    # 确认类型：用户需要确认是/否
    CONFIRMATION = "confirmation"
    # 选择类型：用户需要从多个选项中选择
    SELECTION = "selection"
    # 输入类型：用户需要提供输入信息
    INPUT = "input"


@dataclass
class InterruptData:
    """通用的 Interrupt 数据结构

    Attributes:
        interrupt_type: Interrupt 类型（使用枚举，避免魔术字符串）
        action_type: 具体的操作类型（如 "cancel_order", "create_order"）
        action_data: 执行操作所需的数据
        display_message: 展示给用户的消息
        display_data: 用于 UI 展示的结构化数据
        metadata: 额外的元数据（扩展字段）
    """
    interrupt_type: InterruptType
    action_type: str
    action_data: Dict[str, Any]
    display_message: str
    display_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于传递给 interrupt()）"""
        return {
            "interrupt_type": self.interrupt_type.value,
            "action_type": self.action_type,
            "action_data": self.action_data,
            "display_message": self.display_message,
            "display_data": self.display_data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterruptData":
        """从字典创建实例"""
        if "interrupt_type" not in data:
            raise ValueError("interrupt_type is required")
        interrupt_type = data.get("interrupt_type")
        if isinstance(interrupt_type, str):
            interrupt_type = InterruptType(interrupt_type)
        return cls(
            interrupt_type=interrupt_type,
            action_type=data.get("action_type", ""),
            action_data=data.get("action_data", {}),
            display_message=data.get("display_message", ""),
            display_data=data.get("display_data"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class InterruptResume:
    """Interrupt 恢复数据结构

    用户响应后的 resume 数据
    """
    confirmed: Optional[bool] = None  # 用于 CONFIRMATION 类型
    selected: Optional[Any] = None     # 用于 SELECTION 类型
    input_value: Optional[str] = None  # 用于 INPUT 类型
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {"metadata": self.metadata}
        if self.confirmed is not None:
            result["confirmed"] = self.confirmed
        if self.selected is not None:
            result["selected"] = self.selected
        if self.input_value is not None:
            result["input_value"] = self.input_value
        return result


def create_interrupt(
    interrupt_type: Union[InterruptType, str],
    action_type: str,
    action_data: Dict[str, Any],
    display_message: str,
    display_data: Optional[Dict[str, Any]] = None,
    **metadata
) -> Dict[str, Any]:
    """创建 interrupt 数据（工厂函数）

    Args:
        interrupt_type: Interrupt 类型（枚举或字符串）
        action_type: 具体的操作类型
        action_data: 执行操作所需的数据
        display_message: 展示给用户的消息
        display_data: 用于 UI 展示的结构化数据
        **metadata: 额外的元数据

    Returns:
        可传递给 interrupt() 的字典

    Example:
        ```python
        interrupt_data = create_interrupt(
            InterruptType.CONFIRMATION,
            action_type="cancel_order",
            action_data={"order_id": 123},
            display_message="确认取消订单？"
        )
        return interrupt(interrupt_data)
        ```
    """
    if isinstance(interrupt_type, str):
        interrupt_type = InterruptType(interrupt_type)

    data = InterruptData(
        interrupt_type=interrupt_type,
        action_type=action_type,
        action_data=action_data,
        display_message=display_message,
        display_data=display_data,
        metadata=metadata,
    )
    return data.to_dict()


def create_confirmation_interrupt(
    action_type: str,
    action_data: Dict[str, Any],
    display_message: str,
    display_data: Optional[Dict[str, Any]] = None,
    **metadata
) -> Dict[str, Any]:
    """创建确认类型的 interrupt（便捷函数）

    Args:
        action_type: 具体的操作类型（如 "cancel_order", "create_order"）
        action_data: 执行操作所需的数据
        display_message: 展示给用户的消息
        display_data: 用于 UI 展示的结构化数据
        **metadata: 额外的元数据

    Returns:
        可传递给 interrupt() 的字典

    Example:
        ```python
        interrupt_data = create_confirmation_interrupt(
            action_type="cancel_order",
            action_data={"order_id": 123, "user_phone": "13800138000"},
            display_message="确认取消订单 #001？"
        )
        return interrupt(interrupt_data)
        ```
    """
    return create_interrupt(
        InterruptType.CONFIRMATION,
        action_type=action_type,
        action_data=action_data,
        display_message=display_message,
        display_data=display_data,
        **metadata
    )


def is_resume_confirm(resume_value: Any) -> bool:
    """检查 resume 值是否表示确认（用于 CONFIRMATION 类型）

    Args:
        resume_value: interrupt() 返回的 resume 值

    Returns:
        True 表示确认，False 表示取消/拒绝，None 表示无法判断
    """
    if isinstance(resume_value, dict):
        return resume_value.get("confirmed")
    return resume_value


class InterruptState:
    """Interrupt 状态管理（在 state 中使用的标记）

    使用类而不是魔术字符串来管理状态中的 interrupt 标记。
    """
    # 状态键名常量（避免魔术字符串）
    INTERRUPT_DATA = "interrupt_data"
    INTERRUPT_RESOLVED = "interrupt_resolved"

    @staticmethod
    def set_interrupt_data(state_dict: Dict[str, Any], interrupt_data: Dict[str, Any]) -> None:
        """在 state 中设置 interrupt 数据"""
        state_dict[InterruptState.INTERRUPT_DATA] = interrupt_data

    @staticmethod
    def get_interrupt_data(state_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从 state 中获取 interrupt 数据"""
        return state_dict.get(InterruptState.INTERRUPT_DATA)

    @staticmethod
    def has_pending_interrupt(state_dict: Dict[str, Any]) -> bool:
        """检查 state 中是否有待处理的 interrupt"""
        return InterruptState.INTERRUPT_DATA in state_dict

    @staticmethod
    def clear_interrupt_data(state_dict: Dict[str, Any]) -> None:
        """从 state 中清除 interrupt 数据"""
        state_dict[InterruptState.INTERRUPT_DATA] = None

    @staticmethod
    def mark_interrupt_resolved(state_dict: Dict[str, Any], result: Dict[str, Any]) -> None:
        """标记 interrupt 已解决"""
        state_dict[InterruptState.INTERRUPT_RESOLVED] = result
        InterruptState.clear_interrupt_data(state_dict)

    @staticmethod
    def get_interrupt_type(state_dict: Dict[str, Any]) -> Optional[InterruptType]:
        """获取 state 中 interrupt 的类型"""
        data = InterruptState.get_interrupt_data(state_dict)
        if data:
            if "interrupt_type" not in data:
                raise ValueError("interrupt_type is required in interrupt data")
            interrupt_type = data.get("interrupt_type")
            if isinstance(interrupt_type, str):
                return InterruptType(interrupt_type)
        return None
