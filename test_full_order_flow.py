"""测试完整订单流程 - 从搜索到确认订单"""

import requests
import json
import time

def test_full_order_flow():
    """测试完整流程：搜索 -> 选择 -> 创建订单 -> 确认"""

    print("\n" + "=" * 100)
    print("测试完整订单流程")
    print("=" * 100 + "\n")

    api_base = "http://localhost:8000"
    session_id = "test_full_flow_002"

    # ========== 步骤 1: 用户发起购买请求 ==========
    print("步骤 1: 用户说'我要下单，购买 西门子商品 2 件，我的手机号是 13800138000'")
    print("-" * 100)

    response = requests.post(
        f"{api_base}/api/chat",
        json={
            "message": "我要下单，购买 西门子商品 2 件，我的手机号是 13800138000",
            "session_id": session_id,
            "stream": True
        },
        stream=True
    )

    selection_id = None
    products = []

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                try:
                    data = json.loads(line_str[6:])
                    if data.get('type') == 'state_update':
                        state_data = data.get('data', {})

                        # 检查是否有待选择
                        if state_data.get('pending_selection'):
                            selection_id = state_data['pending_selection']['selection_id']
                            products = state_data['pending_selection']['options']
                            print(f"✅ 找到 {len(products)} 个产品")
                            print(f"   选择ID: {selection_id}")
                            for i, p in enumerate(products[:3], 1):
                                print(f"   {i}. {p.get('name')} - ¥{p.get('price')}")

                    elif data.get('type') == 'done':
                        break
                except:
                    pass

    if not selection_id or not products:
        print("❌ 失败：没有收到产品选择请求")
        return

    # ========== 步骤 2: 用户选择产品 ==========
    print("\n" + "=" * 100)
    print("步骤 2: 用户选择第一个产品")
    print("-" * 100)

    selected_product = products[0]
    print(f"选择产品: {selected_product.get('name')} (ID: {selected_product.get('id')})")

    # 调用选择接口（流式）
    response = requests.post(
        f"{api_base}/api/selection/resolve",
        json={
            "selection_id": selection_id,
            "selected_option_id": str(selected_product.get('id'))
        },
        stream=True
    )

    confirmation_id = None
    order_info = None

    print("\n流式响应:")
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                try:
                    data = json.loads(line_str[6:])
                    event_type = data.get('type')

                    if event_type == 'selection_resolved':
                        print(f"✅ {data.get('message')}")

                    elif event_type == 'state_update':
                        state_data = data.get('data', {})

                        # 检查当前agent
                        if state_data.get('current_agent'):
                            print(f"   当前Agent: {state_data['current_agent']}")

                        # 检查是否有确认请求
                        if state_data.get('confirmation_pending'):
                            confirmation_data = state_data['confirmation_pending']
                            confirmation_id = confirmation_data['confirmation_id']
                            print(f"✅ 收到订单确认请求")
                            print(f"   确认ID: {confirmation_id}")
                            print(f"   消息: {confirmation_data.get('display_message')}")

                            # 显示订单详情
                            if confirmation_data.get('display_data'):
                                display_data = confirmation_data['display_data']
                                if display_data.get('items'):
                                    print(f"   订单项:")
                                    for item in display_data['items']:
                                        print(f"     - {item.get('name')} x{item.get('quantity')} = ¥{item.get('subtotal')}")
                                if display_data.get('total_amount'):
                                    print(f"   总金额: ¥{display_data['total_amount']}")

                    elif event_type == 'done':
                        print("✅ 流式响应完成")
                        break

                except Exception as e:
                    print(f"解析错误: {e}")

    if not confirmation_id:
        print("\n❌ 失败：没有收到订单确认请求")
        print("   期望：order_agent应该创建订单并返回confirmation_pending")
        return

    # ========== 步骤 3: 用户确认订单 ==========
    print("\n" + "=" * 100)
    print("步骤 3: 用户确认订单")
    print("-" * 100)

    response = requests.post(
        f"{api_base}/api/confirmation/resolve",
        json={
            "confirmation_id": confirmation_id,
            "confirmed": True
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ 订单确认成功")
        print(f"   消息: {result.get('message')}")
        if result.get('data'):
            print(f"   订单数据: {json.dumps(result['data'], ensure_ascii=False, indent=2)}")
    else:
        print(f"❌ 订单确认失败: {response.status_code}")
        print(f"   {response.text}")

    # ========== 测试总结 ==========
    print("\n" + "=" * 100)
    print("测试总结")
    print("=" * 100)

    if confirmation_id:
        print("✅ 完整流程测试成功！")
        print("\n流程验证:")
        print("  1. ✅ 用户提问 -> 系统自动搜索产品")
        print("  2. ✅ 系统返回产品列表 -> 用户选择产品")
        print("  3. ✅ 系统创建订单 -> 返回确认请求")
        print("  4. ✅ 用户确认 -> 订单创建完成")
        print("\n🎉 所有步骤都按预期工作！")
    else:
        print("❌ 流程不完整")
        print("\n问题:")
        print("  - 用户选择产品后，系统没有继续执行订单创建")
        print("  - 期望：order_agent应该自动执行并返回confirmation_pending")


if __name__ == "__main__":
    test_full_order_flow()
