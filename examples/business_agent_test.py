"""业务 Agent 集成测试

测试电商客服多 Agent 系统的业务功能：
- 商品搜索
- 订单查询
- 订单取消（含确认机制）
- 下单流程（含确认机制）
"""

import os
import sys
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from src.multi_agent.graph import MultiAgentGraph
from src.db.engine import test_connection
from src.confirmation import get_confirmation_manager, reset_confirmation_manager


def print_separator(title: str = ""):
    """打印分隔符"""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def test_db_connection():
    """测试数据库连接"""
    print_separator("测试数据库连接")

    if test_connection():
        print("✅ 数据库连接成功")
        return True
    else:
        print("❌ 数据库连接失败")
        return False


def test_seed_data():
    """测试数据生成"""
    print_separator("生成测试数据")

    try:
        seed_all(drop_existing=False)
        print("✅ 测试数据生成成功")
        return True
    except Exception as e:
        print(f"❌ 测试数据生成失败: {e}")
        return False


def test_product_search():
    """测试商品搜索"""
    print_separator("CASE 1: 商品搜索")

    # 初始化图（启用业务 Agent）
    graph = MultiAgentGraph(
        enable_business_agents=True,
        enable_intent_classification=False,  # 简化测试，跳过意图识别
        max_iterations=5,
    )

    # 测试查询
    queries = [
      "帮我搜索10个 西门子和danfoss的产品"
        # "帮我找 2000 元以下的智能手机，要评价好的",
        # "华为的笔记本电脑有哪些",
        # "推荐一款性价比高的手机",
    ]

    for query in queries:
        print(f"\n🔍 用户问题: {query}")
        print("-" * 40)

        try:
            result = asyncio.run(graph.ainvoke(query))

            # 打印最终回复
            messages = result.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n🤖 Agent回复:\n{msg.content}")
                    break

            # 打印路由信息
            current_agent = result.get("current_agent")
            routing_reason = result.get("routing_reason")
            print(f"\n📊 路由信息: Agent={current_agent}, Reason={routing_reason}")

        except Exception as e:
            print(f"❌ 执行出错: {e}")


def test_order_query():
    """测试订单查询"""
    print_separator("CASE 2: 订单查询")

    graph = MultiAgentGraph(
        enable_business_agents=True,
        enable_intent_classification=False,
        max_iterations=5,
    )

    query = "我的手机号是 13800138000，查询我的订单"
    print(f"\n🔍 用户问题: {query}")
    print("-" * 40)

    try:
        result = asyncio.run(graph.ainvoke(query))

        # 打印最终回复
        messages = result.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\n🤖 Agent回复:\n{msg.content}")
                break

        # 打印路由信息
        current_agent = result.get("current_agent")
        print(f"\n📊 路由的Agent: {current_agent}")

    except Exception as e:
        print(f"❌ 执行出错: {e}")


def test_order_cancel_with_confirmation():
    """测试订单取消（含确认机制）"""
    print_separator("CASE 3: 订单取消（含确认）")

    graph = MultiAgentGraph(
        enable_business_agents=True,
        enable_intent_classification=False,
        max_iterations=10,
    )

    # 模拟两轮对话
    conversation = [
        "取消订单 1，我的手机号是 13800138000",
        "确认",
    ]

    state = None
    for i, user_input in enumerate(conversation, 1):
        print(f"\n🔍 用户问题 (第{i}轮): {user_input}")
        print("-" * 40)

        try:
            if state is None:
                result = asyncio.run(graph.ainvoke(user_input))
            else:
                # 使用上一次的状态继续对话
                result = asyncio.run(graph.ainvoke(user_input))

            # 打印最终回复
            messages = result.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n🤖 Agent回复:\n{msg.content}")
                    break

            # 检查确认状态
            confirmation = result.get("confirmation_pending")
            if confirmation:
                print(f"\n⚠️ 等待确认: {confirmation}")

            state = result

        except Exception as e:
            print(f"❌ 执行出错: {e}")
            break


async def test_order_create_with_confirmation_async():
    """测试订单创建（含确认机制）- 使用 ConfirmationManager"""
    print_separator("CASE 4: 订单创建（含确认）")

    # 重置确认管理器（确保干净的测试环境）
    reset_confirmation_manager()
    manager = get_confirmation_manager()

    # 注册执行器
    from src.api.server import _register_confirmation_executors
    _register_confirmation_executors(manager)

    graph = MultiAgentGraph(
        enable_business_agents=True,
        enable_intent_classification=False,
        max_iterations=10,
    )

    session_id = "test-session-order-create"

    # 第一轮：发起订单创建
    print("\n🔍 用户问题 (第1轮): 我要下单，购买 1 号商品 2 件，我的手机号是 13800138000")
    print("-" * 40)

    try:
        result1 = await graph.ainvoke(
            "我要下单，购买 1 号商品 2 件，我的手机号是 13800138000",
            session_id=session_id
        )

        # 打印第一��回复
        messages = result1.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\n🤖 Agent回复:\n{msg.content}")
                break

        # 检查确认状态
        confirmation_pending = result1.get("confirmation_pending")
        if confirmation_pending:
            print(f"\n⚠️ 等待确认: {confirmation_pending}")

        # 验证确认请求已创建
        pending = await manager.get_pending_confirmation(session_id)
        if pending:
            print(f"\n✅ 确认请求已创建: {pending.confirmation_id}")
            print(f"   操作类型: {pending.action_type}")
            print(f"   显示消息: {pending.display_message}")

            # 第二轮：用户确认
            print("\n🔍 用户问题 (第2轮): 确认")
            print("-" * 40)

            # 解析确认
            resolve_result = await manager.resolve_confirmation(
                pending.confirmation_id,
                confirmed=True
            )

            print(f"\n✅ 确认解析结果:")
            print(f"   状态: {resolve_result.status}")
            print(f"   执行结果: {resolve_result.execution_result}")

            if resolve_result.error:
                print(f"   错误: {resolve_result.error}")
            else:
                print("\n✅ 订单创建确认流程测试通过!")

            # 验证没有更多待确认操作
            pending_after = await manager.get_pending_confirmation(session_id)
            if pending_after is None:
                print("✅ 确认状态已正确清除")
            else:
                print(f"⚠️ 仍有待确认操作: {pending_after}")
        else:
            print("\n⚠️ 未找到待确认操作（可能工具未调用 prepare_create_order）")

    except Exception as e:
        print(f"❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()


def test_order_create_with_confirmation():
    """测试订单创建（含确认机制）- 同步包装"""
    asyncio.run(test_order_create_with_confirmation_async())


def test_supervisor_routing():
    """测试 Supervisor 路由决策"""
    print_separator("CASE 5: Supervisor 路由决策")

    graph = MultiAgentGraph(
        enable_business_agents=True,
        enable_intent_classification=False,
        max_iterations=3,
    )

    test_cases = [
        ("2000元以下的手机", "product_agent"),
        ("我的订单", "order_agent"),
        ("今天天气怎么样", "chat_agent"),
    ]

    for query, expected_agent in test_cases:
        print(f"\n🔍 用户问题: {query}")
        print(f"🎯 预期路由: {expected_agent}")
        print("-" * 40)

        try:
            result = asyncio.run(graph.ainvoke(query))

            actual_agent = result.get("current_agent")
            routing_reason = result.get("routing_reason")

            status = "✅" if actual_agent == expected_agent else "⚠️"
            print(f"{status} 实际路由: {actual_agent}")
            print(f"   路由原因: {routing_reason}")

        except Exception as e:
            print(f"❌ 执行出错: {e}")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("  电商客服多 Agent 系统 - 业务功能集成测试")
    print("=" * 60)

    # 测试数据库连接
    if not test_db_connection():
        print("\n⚠️  数据库连接失败，部分测试可能无法执行")
        print("   请检查 DATABASE_URL 配置")

    # 生成测试数据
    # test_seed_data()  # 可选：如果需要重新生成数据

    # 运行测试用例
    test_cases = [
        ("商品搜索", test_product_search),
        # ("订单查询", test_order_query),
        # ("订单取消（含确认）", test_order_cancel_with_confirmation),
        # ("订单创建（含确认）", test_order_create_with_confirmation),
        # ("Supervisor 路由决策", test_supervisor_routing),
    ]

    results = {}
    for name, test_func in test_cases:
        try:
            test_func()
            results[name] = "✅ 通过"
        except Exception as e:
            results[name] = f"❌ 失败: {e}"

    # 打印测试结果汇总
    print_separator("测试结果汇总")
    for name, result in results.items():
        print(f"{result} - {name}")

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_order_create_with_confirmation()
