"""确认机制单元测试

直接测试 ConfirmationManager 的功能，不依赖 LLM 行为。
"""

import os
import sys
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.confirmation import (
    get_confirmation_manager,
    reset_confirmation_manager,
    ConfirmationStatus,
)


async def test_confirmation_manager():
    """测试 ConfirmationManager 核心功能"""
    print("\n" + "=" * 60)
    print("  ConfirmationManager 单元测试")
    print("=" * 60)

    # 重置确认管理器
    reset_confirmation_manager()
    manager = get_confirmation_manager()

    # 注册执行器
    async def mock_create_order_executor(action_type: str, action_data: dict) -> dict:
        """模拟订单创建执行器"""
        print(f"  [执行器] 创建订单: {action_data}")
        return {
            "text": f"订单创建成功！订单号: ORD-12345",
            "success": True,
            "order_id": "ORD-12345",
        }

    async def mock_cancel_order_executor(action_type: str, action_data: dict) -> dict:
        """模拟订单取消执行器"""
        print(f"  [执行器] 取消订单: {action_data}")
        return {
            "text": f"订单已取消",
            "success": True,
        }

    manager.register_executor("create_order", mock_create_order_executor)
    manager.register_executor("cancel_order", mock_cancel_order_executor)

    session_id = "test-session-001"

    # 测试 1: 创建确认请求
    print("\n--- 测试 1: 创建确认请求 ---")
    confirmation = await manager.request_confirmation(
        session_id=session_id,
        action_type="create_order",
        action_data={
            "user_phone": "13800138000",
            "items": [{"product_id": 1, "quantity": 2}],
        },
        agent_name="order_agent",
        display_message="确认创建订单？\n- 商品: 测试商品 x 2\n- 总金额: ¥199.00",
        display_data={
            "items": [{"name": "测试商品", "quantity": 2, "subtotal": 199.00}],
            "total_amount": 199.00,
        },
    )

    print(f"  ✅ 确认请求已创建")
    print(f"     ID: {confirmation.confirmation_id}")
    print(f"     操作类型: {confirmation.action_type}")
    print(f"     状态: {confirmation.status}")

    # 测试 2: 获取待确认操作
    print("\n--- 测试 2: 获取待确认操作 ---")
    pending = await manager.get_pending_confirmation(session_id)
    if pending:
        print(f"  ✅ 找到待确认操作: {pending.confirmation_id}")
        assert pending.confirmation_id == confirmation.confirmation_id
    else:
        print("  ❌ 未找到待确认操作")
        return False

    # 测试 3: 通过文本输入解析确认
    print("\n--- 测试 3: 通过文本输入解析确认 ---")

    # 3a: 测试确认关键词
    print("  3a: 测试确认关键词 '确认'")
    result = await manager.check_and_resolve_from_text(session_id, "确认")
    if result:
        print(f"     ✅ 确认解析成功")
        print(f"        状态: {result.status}")
        print(f"        执行结果: {result.execution_result}")
        assert result.status == ConfirmationStatus.CONFIRMED
    else:
        print("     ❌ 确认解析失败")
        return False

    # 测试 4: 验证确认后没有待确认操作
    print("\n--- 测试 4: 验证确认后状态 ---")
    pending_after = await manager.get_pending_confirmation(session_id)
    if pending_after is None:
        print("  ✅ 确认后无待确认操作")
    else:
        print(f"  ❌ 仍有待确认操作: {pending_after.confirmation_id}")
        return False

    # 测试 5: 测试取消流程
    print("\n--- 测试 5: 测试取消流程 ---")

    # 创建新的确认请求
    confirmation2 = await manager.request_confirmation(
        session_id=session_id,
        action_type="cancel_order",
        action_data={"order_id": 1, "user_phone": "13800138000"},
        agent_name="order_agent",
        display_message="确认取消订单 #001？",
    )
    print(f"  创建取消订单确认请求: {confirmation2.confirmation_id}")

    # 用户取消
    result2 = await manager.check_and_resolve_from_text(session_id, "不要了")
    if result2 and result2.status == ConfirmationStatus.CANCELLED:
        print("  ✅ 取消操作成功")
    else:
        print("  ❌ 取消操作失败")
        return False

    # 测试 6: 测试直接解析确认
    print("\n--- 测试 6: 测试直接解析确认 API ---")

    confirmation3 = await manager.request_confirmation(
        session_id=session_id,
        action_type="create_order",
        action_data={"user_phone": "13900139000", "items": []},
        agent_name="order_agent",
        display_message="测试直接确认",
    )

    result3 = await manager.resolve_confirmation(
        confirmation3.confirmation_id,
        confirmed=True
    )
    print(f"  ✅ 直接确认解析成功")
    print(f"     状态: {result3.status}")

    # 测试 7: 测试过期功能
    print("\n--- 测试 7: 测试超时创建 ---")
    confirmation4 = await manager.request_confirmation(
        session_id="timeout-session",
        action_type="create_order",
        action_data={},
        agent_name="order_agent",
        display_message="测试超时",
        ttl_seconds=1,  # 1 秒过期
    )
    print(f"  创建 1 秒过期的确认请求: {confirmation4.confirmation_id}")
    print(f"  过期时间: {confirmation4.expires_at}")

    print("\n" + "=" * 60)
    print("  ✅ 所有测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_confirmation_manager())
    exit(0 if success else 1)
