"""测试订单流程 - 验证多步骤任务链是否正常工作"""

import asyncio
import json
import logging
from src.multi_agent.graph import MultiAgentGraph

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_order_flow():
    """测试完整的订单流程：搜索商品 -> 选择 -> 创建订单"""

    print("\n" + "=" * 80)
    print("测试场景：用户说'我要下单，购买 西门子商品 2 件，我的手机号是 13800138000'")
    print("=" * 80 + "\n")

    # 初始化MultiAgentGraph
    print("正在初始化MultiAgentGraph...")
    graph = MultiAgentGraph(
        init_web_search=False,  # 跳过web search初始化以加快测试
        enable_intent_classification=True,
        enable_business_agents=True
    )

    question = "我要下单，购买 西门子商品 2 件，我的手机号是 13800138000"
    session_id = "test_session_001"

    print(f"\n{'=' * 80}")
    print(f"用户输入: {question}")
    print(f"会话ID: {session_id}")
    print(f"{'=' * 80}\n")

    # 使用流式输出观察每一步
    print("\n开始执行任务链...\n")

    step_count = 0
    async for state_update in graph.astream(question, session_id=session_id, stream_mode="updates"):
        step_count += 1
        print(f"\n--- 步骤 {step_count} ---")

        for node_name, node_data in state_update.items():
            print(f"节点: {node_name}")

            # 显示关键信息
            if isinstance(node_data, dict):
                if "next_action" in node_data:
                    print(f"  下一步行动: {node_data['next_action']}")
                if "current_agent" in node_data:
                    print(f"  当前Agent: {node_data['current_agent']}")
                if "routing_reason" in node_data:
                    print(f"  路由原因: {node_data['routing_reason']}")

                # 显示任务链信息
                if "task_chain" in node_data and node_data["task_chain"]:
                    task_chain = node_data["task_chain"]
                    print(f"  任务链ID: {task_chain.get('chain_id')}")
                    print(f"  任务链类型: {task_chain.get('chain_type')}")
                    print(f"  当前步骤: {task_chain.get('current_step_index')}/{len(task_chain.get('steps', []))}")

                    current_step_index = task_chain.get('current_step_index', 0)
                    steps = task_chain.get('steps', [])
                    if current_step_index < len(steps):
                        current_step = steps[current_step_index]
                        print(f"  当前步骤类型: {current_step.get('step_type')}")
                        print(f"  当前步骤状态: {current_step.get('status')}")

                    # 显示上下文数据
                    context_data = task_chain.get('context_data', {})
                    if context_data:
                        print(f"  上下文数据: {context_data}")

                # 显示待选择信息
                if "pending_selection" in node_data and node_data["pending_selection"]:
                    pending = node_data["pending_selection"]
                    print(f"  待选择: {pending.get('selection_type')}")
                    print(f"  选项数量: {len(pending.get('options', []))}")
                    if pending.get('options'):
                        print(f"  可选商品:")
                        for i, option in enumerate(pending['options'][:3], 1):
                            print(f"    {i}. {option.get('name')} - ¥{option.get('price')}")

                # 显示消息
                if "messages" in node_data and node_data["messages"]:
                    from langchain_core.messages import AIMessage, ToolMessage
                    messages = node_data["messages"]
                    last_msg = messages[-1] if messages else None

                    if isinstance(last_msg, AIMessage):
                        print(f"  AI回复: {last_msg.content[:200]}")
                    elif isinstance(last_msg, ToolMessage):
                        try:
                            tool_result = json.loads(last_msg.content)
                            if "products" in tool_result:
                                products = tool_result["products"]
                                print(f"  工具返回: 找到{len(products)}个商品")
                                for i, p in enumerate(products[:3], 1):
                                    print(f"    {i}. {p.get('name')} - ¥{p.get('price')}")
                        except:
                            print(f"  工具返回: {last_msg.content[:200]}")

    print(f"\n{'=' * 80}")
    print("任务链执行完毕")
    print(f"{'=' * 80}\n")

    # 总结
    print("\n测试总结:")
    print("1. 系统应该已经检测到这是一个多步骤任务（order_with_search）")
    print("2. 系统应该已经自动搜索了西门子商品")
    print("3. 系统应该已经返回了商品列表供用户选择")
    print("4. 前端应该展示产品选择UI，而不是纯文本")
    print("\n如果看到产品列表，说明修复成功！✅")
    print("如果还是看到询问文本，说明还有问题需要修复 ❌\n")


if __name__ == "__main__":
    asyncio.run(test_order_flow())
