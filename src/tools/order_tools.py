"""订单管理工具

提供订单查询、取消、创建功能。
"""

from typing import Annotated, Optional, List
from decimal import Decimal

from langchain_core.tools import tool
from pydantic import Field

from src.db.engine import get_db_session
from src.db.crud import (
    get_order_by_id,
    get_order_by_number,
    get_user_orders,
    create_order as create_order_db,
    cancel_order as cancel_order_db,
    get_product_by_id,
)
from src.db.models import Order, OrderItem
from src.schema.business_models import (
    OrderDisplay,
    OrderCreateItem,
    ConfirmationRequest,
)


@tool
def query_user_orders(
    user_phone: Annotated[
        str,
        Field(
            description="用户手机号（必填）",
            examples=["13800138000", "13900139000"]
        )
    ],
    status: Annotated[
        Optional[str],
        Field(
            default=None,
            description="订单状态筛选，可选值: pending(待支付)/paid(已支付)/shipped(已发货)/delivered(已收货)/cancelled(已取消)",
            examples=["pending", "paid", "delivered"]
        )
    ] = None,
    limit: Annotated[
        int,
        Field(
            default=10,
            description="返回结果数量限制",
            examples=[10, 20, 50]
        )
    ] = 10,
) -> str:
    """查询用户订单列表

    参数说明:
    - user_phone: 用户手机号（必填）
    - status: 订单状态筛选，可选值: pending(待支付)/paid(已支付)/shipped(已发货)/delivered(已收货)/cancelled(已取消)
    - limit: 返回结果数量限制，默认10条

    使用场景:
    - "我的订单" → 需要提供 user_phone
    - "查看待支付的订单" → status='pending'
    - "已完成的订单" → status='delivered'

    Returns:
        订单列表的格式化文本
    """
    try:
        with get_db_session() as db:
            # 直接使用手机号查询订单
            orders = get_user_orders(db, user_phone, status=status, limit=limit)

            if not orders:
                status_msg = f"(状态: {status})" if status else ""
                return f"📋 手机号 {user_phone} 暂无订单{status_msg}"

            # 格式化结果
            result_lines = [
                f"📋 手机号 {user_phone} 的订单 (共{len(orders)}个):\n"
            ]

            for i, order in enumerate(orders, 1):
                display = OrderDisplay.from_db(order)
                result_lines.append(f"{i}. {display.format_text()}\n")

            return "\n".join(result_lines)

    except Exception as e:
        return f"❌ 查询订单时出错: {str(e)}"


@tool
def query_order_detail(
    order_id: Annotated[
        Optional[int],
        Field(
            default=None,
            description="订单ID（二选一）",
            examples=[1, 2, 100]
        )
    ] = None,
    order_number: Annotated[
        Optional[str],
        Field(
            default=None,
            description="订单号，如 ORD123456（二选一）",
            examples=["ORD123456", "ORD789012"]
        )
    ] = None,
) -> str:
    """查询订单详细信息

    参数说明:
    - order_id: 订单ID（二选一）
    - order_number: 订单号，如 ORD123456（二选一）

    使用场景:
    - "查询订单123的详情" → order_id=123
    - "订单ORD123456的详情" → order_number='ORD123456'

    Returns:
        订单详细信息
    """
    try:
        with get_db_session() as db:
            order = None
            if order_id:
                order = get_order_by_id(db, order_id)
            elif order_number:
                order = get_order_by_number(db, order_number)

            if not order:
                return f"❌ 未找到订单 (ID: {order_id}, 订单号: {order_number})"

            display = OrderDisplay.from_db(order)

            # 状态映射
            status_emoji = {
                "pending": "⏳ 待支付",
                "paid": "💰 已支付",
                "shipped": "🚚 已发货",
                "delivered": "✅ 已收货",
                "cancelled": "❌ 已取消",
            }.get(display.status, display.status)

            result = [
                f"📋 订单详情",
                f"━━━━━━━━━━━━━━━━━━━━",
                f"🔢 订单号: {display.order_number}",
                f"🆔 订单ID: {display.id}",
                f"📊 状态: {status_emoji}",
                f"💰 总金额: ¥{display.total_amount:.2f}",
                f"📅 创建时间: {display.created_at}",
                f"\n📦 商品清单:",
            ]

            for item in display.items:
                result.append(
                    f"   • {item.get('product_name', 'Unknown')} "
                    f"x {item['quantity']} = ¥{item['subtotal']:.2f}"
                )

            result.append(f"━━━━━━━━━━━━━━━━━━━━")
            result.append(f"💡 订单ID: {display.id} (用于取消订单)")

            return "\n".join(result)

    except Exception as e:
        return f"❌ 查询订单详情时出错: {str(e)}"


@tool
def prepare_cancel_order(
    order_id: Annotated[
        int,
        Field(
            description="要取消的订单ID",
            examples=[1, 2, 100]
        )
    ],
    user_phone: Annotated[
        str,
        Field(
            description="用户手机号（用于验证权限）",
            examples=["13800138000", "13900139000"]
        )
    ],
    reason: Annotated[
        Optional[str],
        Field(
            default=None,
            description="取消原因（可选）",
            examples=["不需要了", "买错了"]
        )
    ] = None,
) -> str:
    """准备取消订单 - 返回确认信息

    注意: 此工具不会直接取消订单，而是返回确认信息。
    用户确认后，需要调用 confirm_cancel_order 工具完成取消。

    参数说明:
    - order_id: 要取消的订单ID
    - user_phone: 用户手机号（用于验证权限）
    - reason: 取消原因（可选）

    Returns:
        确认信息
    """
    try:
        with get_db_session() as db:
            # 查询订单
            order = get_order_by_id(db, order_id)
            if not order:
                return f"❌ 未找到ID为 {order_id} 的订单"

            # 验证用户（现在 user_id 是字符串类型的手机号）
            if order.user_id != user_phone:
                return f"❌ 无权取消此订单（订单属于用户 {order.user_id}）"

            # 检查订单状态
            if order.status == "cancelled":
                return f"⚠️ 订单 {order.order_id} 已经是取消状态"

            if order.status not in ["pending", "paid"]:
                return f"⚠️ 订单 {order.order_id} 的状态为 {order.status}，无法取消"

            display = OrderDisplay.from_db(order)

            # 返回确认信息
            result = [
                f"⚠️ 确认取消订单",
                f"━━━━━━━━━━━━━━━━━━━━",
                f"🔢 订单号: {display.order_number}",
                f"💰 金额: ¥{display.total_amount:.2f}",
                f"📊 状态: {display.status}",
            ]

            if reason:
                result.append(f"📝 取消原因: {reason}")

            result.append(f"\n⚠️ 请确认：您确定要取消此订单吗？")
            result.append(f"   如果确认，请回复'确认'或'是'。")

            return "\n".join(result)

    except Exception as e:
        return f"❌ 准备取消订单时出错: {str(e)}"


@tool
def confirm_cancel_order(
    order_id: Annotated[
        int,
        Field(
            description="要取消的订单ID",
            examples=[1, 2, 100]
        )
    ],
    user_phone: Annotated[
        str,
        Field(
            description="用户手机号（用于验证权限）",
            examples=["13800138000", "13900139000"]
        )
    ],
) -> str:
    """确认取消订单 - 执行实际的取消操作

    注意: 应该先调用 prepare_cancel_order 让用户确认后再调用此工具。

    参数说明:
    - order_id: 要取消的订单ID
    - user_phone: 用户手机号（用于验证权限）

    Returns:
        取消结果
    """
    try:
        with get_db_session() as db:
            # 验证权限
            order = get_order_by_id(db, order_id)
            if not order:
                return f"❌ 未找到ID为 {order_id} 的订单"

            if order.user_id != user_phone:
                return f"❌ 无权取消此订单"

            # 执行取消
            order = cancel_order_db(db, order_id)

            return f"✅ 订单 {order.order_id} 已成功取消"

    except ValueError as e:
        return f"⚠️ 取消失败: {str(e)}"
    except Exception as e:
        return f"❌ 取消订单时出错: {str(e)}"


@tool
def prepare_create_order(
    user_phone: Annotated[
        str,
        Field(
            description="用户手机号",
            examples=["13800138000", "13900139000"]
        )
    ],
    items: Annotated[
        str,
        Field(
            description="商品列表（JSON格式字符串），如: [{\"product_id\": 1, \"quantity\": 2}]",
            examples=['[{"product_id": 1, "quantity": 2}]', '[{"product_id": 5, "quantity": 1}]']
        )
    ],
    notes: Annotated[
        Optional[str],
        Field(
            default=None,
            description="订单备注（可选）",
            examples=["请尽快发货", "送到门口"]
        )
    ] = None,
) -> str:
    """准备创建订单 - 返回确认信息

    注意: 此工具不会直接创建订单，而是验证并返回确认信息。
    用户确认后，需要调用 confirm_create_order 工具完成创建。

    参数说明:
    - user_phone: 用户手机号
    - items: 商品列表（JSON格式字符串），如: [{"product_id": 1, "quantity": 2}]
    - notes: 订单备注（可选）

    Returns:
        确认信息，包含订单预览
    """
    try:
        import json

        # 解析商品列表
        try:
            items_data = json.loads(items)
        except json.JSONDecodeError:
            return "❌ 商品列表格式错误，请使用JSON格式: [{\"product_id\": 1, \"quantity\": 2}]"

        with get_db_session() as db:
            # 验证商品并计算金额
            total_amount = Decimal("0")
            items_preview = []

            for item in items_data:
                product = get_product_by_id(db, item["product_id"])
                if not product:
                    return f"❌ 未找到ID为 {item['product_id']} 的商品"

                stock = product.stock or 0
                if stock < item["quantity"]:
                    return f"❌ 商品 {product.name} 库存不足 (库存: {stock}, 需要: {item['quantity']})"

                price = product.price or Decimal("0")
                subtotal = price * item["quantity"]
                total_amount += subtotal

                items_preview.append({
                    "name": product.name,
                    "quantity": item["quantity"],
                    "price": float(price),
                    "subtotal": float(subtotal),
                })

            # 返回确认信息
            result = [
                f"🛒 确认订单信息",
                f"━━━━━━━━━━━━━━━━━━━━",
                f"👤 用户手机号: {user_phone}",
                f"📦 商品清单:",
            ]

            for item in items_preview:
                result.append(
                    f"   • {item['name']} x {item['quantity']} = ¥{item['subtotal']:.2f}"
                )

            result.extend([
                f"💰 总金额: ¥{float(total_amount):.2f}",
            ])

            if notes:
                result.append(f"📝 备注: {notes}")

            result.extend([
                f"━━━━━━━━━━━━━━━━━━━━",
                f"⚠️ 请确认：是否创建此订单？",
                f"   如果确认，请回复'确认'或'是'。",
            ])

            return "\n".join(result)

    except Exception as e:
        return f"❌ 准备创建订单时出错: {str(e)}"


@tool
def confirm_create_order(
    user_phone: Annotated[
        str,
        Field(
            description="用户手机号",
            examples=["13800138000", "13900139000"]
        )
    ],
    items: Annotated[
        str,
        Field(
            description="商品列表（JSON格式字符串）",
            examples=['[{"product_id": 1, "quantity": 2}]', '[{"product_id": 5, "quantity": 1}]']
        )
    ],
    notes: Annotated[
        Optional[str],
        Field(
            default=None,
            description="订单备注（可选）",
            examples=["请尽快发货", "送到门口"]
        )
    ] = None,
) -> str:
    """确认创建订单 - 执行实际的创建操作

    注意: 应该先调用 prepare_create_order 让用户确认后再调用此工具。

    参数说明:
    - user_phone: 用户手机号
    - items: 商品列表（JSON格式字符串）
    - notes: 订单备注（可选）

    Returns:
        创建结果
    """
    try:
        import json

        # 解析商品列表
        items_data = json.loads(items)

        with get_db_session() as db:
            # 创建订单
            order = create_order_db(
                db,
                user_phone=user_phone,
                items=items_data,
                notes=notes,
            )

            return (
                f"✅ 订单创建成功！\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🔢 订单号: {order.order_id}\n"
                f"🆔 订单ID: {order.id}\n"
                f"💰 金额: ¥{float(order.total_amount):.2f}\n"
                f"📊 状态: {order.status}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 请保存订单号以便查询"
            )

    except ValueError as e:
        return f"⚠️ 创建失败: {str(e)}"
    except Exception as e:
        return f"❌ 创建订单时出错: {str(e)}"


@tool
def update_order_status(
    order_id: Annotated[
        int,
        Field(
            description="订单ID",
            examples=[1, 2, 100]
        )
    ],
    status: Annotated[
        str,
        Field(
            description="新状态，可选值: pending/paid/shipped/delivered/cancelled",
            examples=["paid", "shipped", "delivered"]
        )
    ],
) -> str:
    """更新订单状态（管理员功能）

    参数说明:
    - order_id: 订单ID
    - status: 新状态，可选值: pending/paid/shipped/delivered/cancelled

    Returns:
        更新结果
    """
    try:
        from src.db.crud import update_order_status as update_status_db

        valid_statuses = ["pending", "paid", "shipped", "delivered", "cancelled"]
        if status not in valid_statuses:
            return f"❌ 无效的状态，可选值: {', '.join(valid_statuses)}"

        with get_db_session() as db:
            order = update_status_db(db, order_id, status)
            if not order:
                return f"❌ 未找到ID为 {order_id} 的订单"

            return f"✅ 订单 {order.order_id} 状态已更新为 {status}"

    except Exception as e:
        return f"❌ 更新订单状态时出错: {str(e)}"


def get_order_tools() -> list:
    """获取所有订单工具"""
    return [
        query_user_orders,
        query_order_detail,
        prepare_cancel_order,
        confirm_cancel_order,
        prepare_create_order,
        confirm_create_order,
        update_order_status,
    ]
