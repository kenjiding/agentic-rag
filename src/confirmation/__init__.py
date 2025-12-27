"""通用确认机制模块

提供跨请求的确认状态管理，支持：
- 订单创建/取消确认
- 退款申请确认
- 其他需要用户确认的操作

Usage:
    ```python
    from src.confirmation import get_confirmation_manager, ConfirmationManager

    # 获取单例管理器
    manager = get_confirmation_manager()

    # 注册执行器
    async def cancel_order_handler(action_type: str, data: dict) -> dict:
        # 执行取消订单逻辑
        return {"success": True, "message": "订单已取消"}

    manager.register_executor("cancel_order", cancel_order_handler)

    # 创建确认请求
    confirmation = await manager.request_confirmation(
        session_id="user-123",
        action_type="cancel_order",
        action_data={"order_id": 1, "user_phone": "13800138000"},
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

from .models import (
    ConfirmationAction,
    ConfirmationResult,
    ConfirmationStatus,
    ConfirmationRequest,
)
from .storage import (
    ConfirmationStorage,
    InMemoryConfirmationStorage,
)
from .manager import (
    ConfirmationManager,
    get_confirmation_manager,
    reset_confirmation_manager,
)
from .exceptions import (
    ConfirmationError,
    ConfirmationNotFoundError,
    ConfirmationExpiredError,
    ConfirmationAlreadyResolvedError,
    ExecutorNotFoundError,
)

__all__ = [
    # 模型
    "ConfirmationAction",
    "ConfirmationResult",
    "ConfirmationStatus",
    "ConfirmationRequest",
    # 存储
    "ConfirmationStorage",
    "InMemoryConfirmationStorage",
    # 管理器
    "ConfirmationManager",
    "get_confirmation_manager",
    "reset_confirmation_manager",
    # 异常
    "ConfirmationError",
    "ConfirmationNotFoundError",
    "ConfirmationExpiredError",
    "ConfirmationAlreadyResolvedError",
    "ExecutorNotFoundError",
]
