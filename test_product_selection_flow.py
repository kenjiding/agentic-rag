#!/usr/bin/env python3
"""Test the complete product selection flow

Flow:
1. User: "帮我购买3个西门子产品"
2. System searches and shows products
3. User selects product_id=1
4. System creates order
5. System asks for confirmation
"""
import asyncio
import json
import requests
import time

BASE_URL = "http://localhost:8000"

async def test_flow():
    print("=" * 60)
    print("测试商品选择完整流程")
    print("=" * 60)

    # Step 1: Send initial request
    print("\n[步骤 1] 发送请求: 帮我购买3个西门子产品")
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": "帮我购买3个西门子产品", "session_id": "test_flow_123"}
    )

    if response.status_code != 200:
        print(f"错误: {response.status_code} - {response.text}")
        return

    data = response.json()
    print(f"响应: {json.dumps(data, ensure_ascii=False, indent=2)}")

    # Extract thread_id for resuming
    thread_id = data.get("thread_id")
    print(f"\nthread_id: {thread_id}")

    # Check if there's a pending selection
    if "requires_selection" in data and data["requires_selection"]:
        print("\n[步骤 2] 等待用户选择商品")

        selection_id = data.get("selection_id")
        options = data.get("options", [])

        print(f"selection_id: {selection_id}")
        print(f"商品列表: {len(options)} 个")

        if not options:
            print("错误: 没有可选商品")
            return

        # Select first product
        selected = options[0]
        print(f"\n[步骤 3] 用户选择商品: {selected.get('name', 'N/A')} (id={selected.get('product_id')})")

        # Step 3: Resolve selection
        resolve_response = requests.post(
            f"{BASE_URL}/api/selection/resolve",
            json={
                "session_id": "test_flow_123",
                "thread_id": thread_id,
                "selection_id": selection_id,
                "selected_value": json.dumps({"product_id": selected.get("product_id")})
            }
        )

        if resolve_response.status_code != 200:
            print(f"错误: {resolve_response.status_code} - {resolve_response.text}")
            return

        resolve_data = resolve_response.json()
        print(f"\n[步骤 4] 选择后的响应:")
        print(json.dumps(resolve_data, ensure_ascii=False, indent=2))

        # Check if confirmation is required
        if "requires_confirmation" in resolve_data and resolve_data["requires_confirmation"]:
            print("\n[步骤 5] 需要用户确认订单")

            confirmation_id = resolve_data.get("confirmation_id")
            print(f"confirmation_id: {confirmation_id}")

            # Confirm the order
            print("\n[步骤 6] 用户确认订单")

            confirm_response = requests.post(
                f"{BASE_URL}/api/confirmation/resolve",
                json={
                    "session_id": "test_flow_123",
                    "thread_id": thread_id,
                    "confirmation_id": confirmation_id,
                    "approved": True
                }
            )

            if confirm_response.status_code != 200:
                print(f"错误: {confirm_response.status_code} - {confirm_response.text}")
                return

            confirm_data = confirm_response.json()
            print(f"\n[步骤 7] 最终响应:")
            print(json.dumps(confirm_data, ensure_ascii=False, indent=2))

            print("\n" + "=" * 60)
            print("流程测试完成!")
            print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_flow())
