"""确认操作执行器"""
import json
import logging
from typing import Dict, Any
from src.confirmation import ConfirmationManager

logger = logging.getLogger(__name__)


async def cancel_order_executor(action_type: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
    """执行订单取消"""
    logger.info(f"🔧 [EXECUTOR_CANCEL] 开始执行订单取消")
    logger.info(f"🔧 [EXECUTOR_CANCEL] action_type: {action_type}")
    logger.info(f"🔧 [EXECUTOR_CANCEL] action_data: {action_data}")
    
    from src.tools.order_tools import confirm_cancel_order
    result = confirm_cancel_order.invoke(action_data)
    
    if isinstance(result, str):
        parsed_result = json.loads(result)
    else:
        parsed_result = result
    
    # 【关键日志】记录执行结果
    logger.info(f"🔧 [EXECUTOR_CANCEL] 执行结果: success={parsed_result.get('success')}, order_status={parsed_result.get('order_status')}")
    
    return parsed_result


async def create_order_executor(action_type: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
    """执行订单创建"""
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"🔧 [EXECUTOR] 开始执行订单创建")
    logger.info(f"🔧 [EXECUTOR] action_type: {action_type}")
    logger.info(f"🔧 [EXECUTOR] action_data: {action_data}")
    if 'user_id' in action_data:
        logger.info(f"🔧 [EXECUTOR] 用户ID: '{action_data['user_id']}' (类型: {type(action_data['user_id']).__name__})")
    if 'items' in action_data:
        logger.info(f"🔧 [EXECUTOR] 商品列表: {action_data['items']}")

    from src.tools.order_tools import confirm_create_order
    result = confirm_create_order.invoke(action_data)

    logger.info(f"🔧 [EXECUTOR] 订单创建结果: {result[:200] if isinstance(result, str) else result}")

    if isinstance(result, str):
        return json.loads(result)
    return result


def register_confirmation_executors(manager: ConfirmationManager) -> None:
    """注册确认操作执行器"""
    manager.register_executor("cancel_order", cancel_order_executor)
    manager.register_executor("create_order", create_order_executor)
    logger.info("已注册订单确认执行器")

