"""测试API流程 - 验证多步骤任务链是否正常工作"""

import requests
import json
import time

def test_order_flow():
    """测试完整的订单流程"""

    print("\n" + "=" * 80)
    print("测试场景：用户说'我要下单，购买 西门子商品 2 件，我的手机号是 13800138000'")
    print("=" * 80 + "\n")

    # API配置
    api_url = "http://localhost:8000/api/chat"

    # 测试请求
    request_data = {
        "message": "我要下单，购买 西门子商品 2 件，我的手机号是 13800138000",
        "session_id": "test_session_001",
        "stream": True
    }

    print(f"发送请求到: {api_url}")
    print(f"请求数据: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
    print("\n" + "=" * 80)
    print("流式响应:")
    print("=" * 80 + "\n")

    try:
        # 发送流式请求
        response = requests.post(api_url, json=request_data, stream=True)

        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
            return

        # 读取流式响应
        step_count = 0
        has_products = False
        has_pending_selection = False

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # 去掉 'data: ' 前缀
                    try:
                        data = json.loads(data_str)

                        if data.get('type') == 'state_update':
                            step_count += 1
                            print(f"\n--- 步骤 {step_count} ---")

                            state_data = data.get('data', {})

                            # 显示执行步骤
                            if 'execution_steps' in state_data:
                                print(f"执行步骤: {' -> '.join(state_data['execution_steps'])}")

                            # 显示当前agent
                            if 'current_agent' in state_data:
                                print(f"当前Agent: {state_data['current_agent']}")

                            # 显示响应类型
                            if 'response_type' in state_data:
                                print(f"响应类型: {state_data['response_type']}")

                                # 检查是否有产品列表
                                if state_data['response_type'] == 'product_list':
                                    has_products = True
                                    products = state_data.get('response_data', {}).get('products', [])
                                    print(f"✅ 找到产品列表，共 {len(products)} 个商品:")
                                    for i, product in enumerate(products[:5], 1):
                                        print(f"  {i}. {product.get('name')} - ¥{product.get('price')}")

                            # 显示待选择
                            if 'pending_selection' in state_data and state_data['pending_selection']:
                                has_pending_selection = True
                                pending = state_data['pending_selection']
                                print(f"✅ 待用户选择: {pending.get('display_message')}")
                                options = pending.get('options', [])
                                print(f"   可选项数量: {len(options)}")
                                for i, option in enumerate(options[:3], 1):
                                    print(f"   {i}. {option.get('name')} - ¥{option.get('price')}")

                            # 显示内容
                            if 'content' in state_data:
                                content = state_data['content']
                                if content:
                                    print(f"回复内容: {content[:200]}")

                        elif data.get('type') == 'done':
                            print("\n" + "=" * 80)
                            print("✅ 流式响应完成")
                            print("=" * 80)

                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}")
                        print(f"原始数据: {data_str}")

        # 测试总结
        print("\n" + "=" * 80)
        print("测试���结:")
        print("=" * 80)

        if has_products:
            print("✅ 成功：系统返回了产品列表")
        else:
            print("❌ 失败：没有返回产品列表")

        if has_pending_selection:
            print("✅ 成功：创建了待选择操作")
        else:
            print("❌ 失败：没有创建待选择操作")

        if has_products and has_pending_selection:
            print("\n🎉 完美！修复成功，前端应该能够显示产品选择UI了！")
        elif has_products:
            print("\n⚠️  部分成功：有产品列表，但缺少待选择操作")
        else:
            print("\n❌ 修复失败：系统仍然只返回文本，没有产品列表")

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器")
        print("请确保API服务器正在运行: python -m uvicorn src.api.server:app --reload")
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_order_flow()
