"""确认操作执行器"""
import json
import logging
from typing import Dict, Any
from src.confirmation import ConfirmationManager

logger = logging.getLogger(__name__)


async def cancel_order_executor(action_type: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
    """执行订单取消"""
    from src.tools.order_tools import confirm_cancel_order
    result = confirm_cancel_order.invoke(action_data)
    if isinstance(result, str):
        return json.loads(result)
    return result


async def create_order_executor(action_type: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
    """执行订单创建"""
    from src.tools.order_tools import confirm_create_order
    result = confirm_create_order.invoke(action_data)
    if isinstance(result, str):
        return json.loads(result)
    return result


def register_confirmation_executors(manager: ConfirmationManager) -> None:
    """注册确认操作执行器"""
    manager.register_executor("cancel_order", cancel_order_executor)
    manager.register_executor("create_order", create_order_executor)
    logger.info("已注册订单确认执行器")

