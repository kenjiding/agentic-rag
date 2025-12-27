"""测试前端渲染 - 验证是否只渲染一次商品列表"""

import requests
import json

def test_frontend_rendering():
    """测试前端应该只看到一次商品列表渲染"""

    print("\n" + "=" * 100)
    print("测试：验证前端只渲染一次商品列表")
    print("=" * 100 + "\n")

    api_base = "http://localhost:8000"
    session_id = "test_frontend_rendering_001"

    request_data = {
        "message": "我要下单，购买 西门子商品 2 件，我的手机号是 13800138000",
        "session_id": session_id,
        "stream": True
    }

    print("发送请求...\n")

    response = requests.post(f"{api_base}/api/chat", json=request_data, stream=True)

    if response.status_code != 200:
        print(f"❌ 请求失败: {response.status_code}")
        return

    product_list_count = 0
    selection_count = 0

    step_number = 0

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]
                try:
                    data = json.loads(data_str)

                    if data.get('type') == 'state_update':
                        step_number += 1
                        state_data = data.get('data', {})
                        response_type = state_data.get('response_type')
                        has_response_data_products = bool(state_data.get('response_data', {}).get('products'))
                        has_pending_selection = bool(state_data.get('pending_selection'))

                        # 检查是否有product_list类型
                        if response_type == 'product_list':
                            product_list_count += 1
                            print(f"步骤 {step_number}: ❌ response_type='product_list'")
                            if has_response_data_products:
                                print(f"         response_data.products 有数据")

                        # 检查是否有selection类型
                        if response_type == 'selection':
                            selection_count += 1
                            print(f"步骤 {step_number}: ✅ response_type='selection'")
                            if has_pending_selection:
                                options = state_data['pending_selection'].get('options', [])
                                print(f"         pending_selection.options 有 {len(options)} 个商品")
                            if has_response_data_products:
                                print(f"         ⚠️  WARNING: response_data.products 仍然有数据（不应该）")

                    elif data.get('type') == 'done':
                        break

                except json.JSONDecodeError:
                    pass

    print("\n" + "=" * 100)
    print("测试结果:")
    print("=" * 100)
    print(f"response_type='product_list' 的次数: {product_list_count}")
    print(f"response_type='selection' 的次数: {selection_count}")
    print()

    if product_list_count == 0 and selection_count == 1:
        print("✅ 完美！前端只会看到一次商品列表（在selection中）")
        print("   前端应该：")
        print("   1. 不渲染单独的ProductGrid")
        print("   2. 只渲染ProductSelectionDialog（包含产品列表）")
    elif product_list_count == 1 and selection_count == 1:
        print("⚠️  前端会看到两次商品列表")
        print("   1. 第一次：ProductGrid（来自product_agent）")
        print("   2. 第二次：ProductSelectionDialog（来自task_orchestrator）")
        print()
        print("   建议：前端应该在收到selection类型时，隐藏之前的product_list渲染")
    else:
        print(f"❌ 异常情况：product_list={product_list_count}, selection={selection_count}")


if __name__ == "__main__":
    test_frontend_rendering()
