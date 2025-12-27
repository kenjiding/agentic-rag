"""端到端确认机制测试"""

import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.confirmation import get_confirmation_manager, reset_confirmation_manager
from src.multi_agent.graph import MultiAgentGraph
from src.api.server import _register_confirmation_executors


async def test_e2e_confirmation():
    """测试完整的确认流程"""

    print("="*60)
    print("  端到端确认机制测试")
    print("="*60)
    print()

    # 重置并获取确认管理器
    reset_confirmation_manager()
    manager = get_confirmation_manager()

    # 注册执行器
    _register_confirmation_executors(manager)
    print("✓ 执行器已注册")
    print()

    # 创建图
    graph = MultiAgentGraph(
        enable_business_agents=True,
        enable_intent_classification=False,
        max_iterations=10,
    )

    session_id = "test-user-e2e"

    # 第一轮：创建订单请求
    print("--- 第一轮：创建订单请求 ---")
    query1 = "我要购买 product_id=1 的商品，数量2个，我的手机号是 13800138000"

    print(f"用户输入: {query1}")
    print()

    # 执行第一轮
    result1 = await graph.ainvoke(query1, session_id=session_id)

    # 打印消息
    messages = result1.get("messages", [])
    if messages:
        last_msg = messages[-1]
        print(f"AI回复: {last_msg.content if hasattr(last_msg, 'content') else last_msg}")
        print()

    # 检查确认
    confirmation_pending = result1.get("confirmation_pending")
    if confirmation_pending:
        print(f"✓ 确认请求已创建:")
        print(f"  ID: {confirmation_pending['confirmation_id']}")
        print(f"  类型: {confirmation_pending['action_type']}")
        print(f"  消息: {confirmation_pending['display_message']}")
        print()
    else:
        print("✗ 没有确认请求")
        return

    # 验证 ConfirmationManager 中也有这个确认
    pending_in_manager = await manager.get_pending_confirmation(session_id)
    if pending_in_manager:
        print(f"✓ ConfirmationManager 中存在待确认操作: {pending_in_manager.confirmation_id}")
        print()
    else:
        print("✗ ConfirmationManager 中没有待确认操作")
        return

    # 第二轮：用户通过文本确认
    print("--- 第二轮：用户通过文本确认 ---")
    query2 = "确认"

    print(f"用户输入: {query2}")
    print()

    # 执行第二轮
    result2 = await graph.ainvoke(query2, session_id=session_id)

    # 打印消息
    messages2 = result2.get("messages", [])
    if messages2:
        last_msg2 = messages2[-1]
        print(f"AI回复: {last_msg2.content if hasattr(last_msg2, 'content') else last_msg2}")
        print()
    else:
        print("✗ 没有收到 AI 回复")
        print()

    # 检查确认状态
    pending_after = await manager.get_pending_confirmation(session_id)
    if pending_after:
        print(f"✗ 仍有待确认操作: {pending_after.confirmation_id}")
    else:
        print(f"✓ 确认已处理，无待确认操作")
        print()

    # 检查确认结果
    confirmation_id = confirmation_pending['confirmation_id']
    confirmation_obj = await manager.get_confirmation(confirmation_id)
    if confirmation_obj:
        print(f"✓ 确认状态: {confirmation_obj.status}")
        print()

    print("="*60)
    print("  测试完成")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_e2e_confirmation())
