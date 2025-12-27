"""订单管理工具

提供订单查询、取消、创建功能。
返回 JSON 格式，包含人类可读文本和结构化订单数据。
"""

import json
from typing import Annotated, Optional
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
    """查询用���订单列表

    返回 JSON 格式，包含人类可读文本和结构化订单数据。
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        # 添加调试日志
        logger.info(f"🔍 [ORDER_QUERY] 开始查询订单")
        logger.info(f"🔍 [ORDER_QUERY] 手机号参数: '{user_phone}' (类型: {type(user_phone).__name__}, 长度: {len(user_phone)})")
        logger.info(f"🔍 [ORDER_QUERY] 状态筛选: {status}, 限制数量: {limit}")

        with get_db_session() as db:
            # 先查询所有订单看看数据库中有什么
            from src.db.models import Order as OrderModel
            all_orders = db.query(OrderModel).limit(20).all()
            logger.info(f"🔍 [ORDER_QUERY] 数据库中最近20个订单:")
            for order in all_orders:
                logger.info(f"  - 订单ID: {order.id}, 手机号: '{order.user_id}', 订单号: {order.order_id}, 状态: {order.status}")

            # 执行用户订单查询
            orders = get_user_orders(db, user_phone, status=status, limit=limit)

            # 添加调试日志
            logger.info(f"🔍 [ORDER_QUERY] 查询结果: 找到 {len(orders)} 个订单")

            # 构建结构化订单数据
            orders_data = []
            for order in orders:
                order_items = [
                    {
                        "product_name": item.product.name if item.product else "未知商品",
                        "quantity": item.quantity,
                        "subtotal": float(item.price * item.quantity),
                    }
                    for item in order.order_items  # 修复：items -> order_items
                ]
                orders_data.append({
                    "id": order.id,
                    "order_number": order.order_id,
                    "status": order.status,
                    "total_amount": float(order.total_amount) if order.total_amount else 0,
                    "created_at": order.created_at.isoformat() if order.created_at else None,
                    "items": order_items,
                })

            # 生成人类可读文本
            if not orders:
                status_msg = f"(状态: {status})" if status else ""
                text = f"手机号 {user_phone} 暂无订单{status_msg}"
            else:
                result_lines = [f"手机号 {user_phone} 的订单 (共{len(orders)}个):\n"]
                for i, order in enumerate(orders, 1):
                    display = OrderDisplay.from_db(order)
                    result_lines.append(f"{i}. {display.format_text()}\n")
                text = "\n".join(result_lines)

            return json.dumps({
                "text": text,
                "orders": orders_data
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "text": f"查询订单时出错: {str(e)}",
            "orders": []
        }, ensure_ascii=False)


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

    返回 JSON 格式，包含人类可读文本和结构化订单数据。
    """
    try:
        with get_db_session() as db:
            order = None
            if order_id:
                order = get_order_by_id(db, order_id)
            elif order_number:
                order = get_order_by_number(db, order_number)

            if not order:
                return json.dumps({
                    "text": f"未找到订单 (ID: {order_id}, 订单号: {order_number})",
                    "order": None
                }, ensure_ascii=False)

            display = OrderDisplay.from_db(order)

            # 构建结构化订单数据
            order_items = [
                {
                    "product_name": item.product_name,
                    "quantity": item.quantity,
                    "subtotal": float(item.price * item.quantity),
                }
                for item in order.items
            ]

            order_data = {
                "id": order.id,
                "order_number": order.order_id,
                "status": order.status,
                "total_amount": float(order.total_amount) if order.total_amount else 0,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "items": order_items,
            }

            # 生成人类可读文本
            status_emoji = {
                "pending": "⏳ 待支付",
                "paid": "💰 已支付",
                "shipped": "🚚 已发货",
                "delivered": "✅ 已收货",
                "cancelled": "❌ 已取消",
            }.get(display.status, display.status)

            text_parts = [
                f"订单详情",
                f"订单号: {display.order_number}",
                f"订单ID: {display.id}",
                f"状态: {status_emoji}",
                f"总金额: ¥{display.total_amount:.2f}",
                f"创建时间: {display.created_at}",
                f"商品清单:",
            ]
            for item in display.items:
                text_parts.append(f"  • {item.get('product_name', 'Unknown')} x {item['quantity']} = ¥{item['subtotal']:.2f}")

            text = "\n".join(text_parts)

            return json.dumps({
                "text": text,
                "order": order_data
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "text": f"查询订单详情时出错: {str(e)}",
            "order": None
        }, ensure_ascii=False)


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
    """准备取消订单 - 返回确认信息（JSON格式）"""
    try:
        with get_db_session() as db:
            order = get_order_by_id(db, order_id)
            if not order:
                return json.dumps({
                    "text": f"未找到ID为 {order_id} 的订单",
                    "can_cancel": False
                }, ensure_ascii=False)

            if order.user_id != user_phone:
                return json.dumps({
                    "text": f"无权取消此订单（订单属于用户 {order.user_id}）",
                    "can_cancel": False
                }, ensure_ascii=False)

            if order.status == "cancelled":
                return json.dumps({
                    "text": f"订单 {order.order_id} 已经是取消状态",
                    "can_cancel": False
                }, ensure_ascii=False)

            if order.status not in ["pending", "paid"]:
                return json.dumps({
                    "text": f"订单 {order.order_id} 的状态为 {order.status}，无法取消",
                    "can_cancel": False
                }, ensure_ascii=False)

            display = OrderDisplay.from_db(order)

            text_lines = [
                f"确认取消订单",
                f"订单号: {display.order_number}",
                f"金额: ¥{display.total_amount:.2f}",
                f"状态: {display.status}",
            ]
            if reason:
                text_lines.append(f"取消原因: {reason}")
            text_lines.append(f"请确认：您确定要取消此订单吗？")

            return json.dumps({
                "text": "\n".join(text_lines),
                "can_cancel": True,
                "order_id": order_id
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "text": f"准备取消订单时出错: {str(e)}",
            "can_cancel": False
        }, ensure_ascii=False)


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
    """确认取消订单 - 执行实际的取消操作（JSON格式）"""
    try:
        with get_db_session() as db:
            order = get_order_by_id(db, order_id)
            if not order:
                return json.dumps({
                    "text": f"未找到ID为 {order_id} 的订单",
                    "success": False
                }, ensure_ascii=False)

            if order.user_id != user_phone:
                return json.dumps({
                    "text": "无权取消此订单",
                    "success": False
                }, ensure_ascii=False)

            order = cancel_order_db(db, order_id)
            return json.dumps({
                "text": f"订单 {order.order_id} 已成功取消",
                "success": True,
                "order_id": order.id
            }, ensure_ascii=False)

    except ValueError as e:
        return json.dumps({
            "text": f"取消失败: {str(e)}",
            "success": False
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "text": f"取消订单时出错: {str(e)}",
            "success": False
        }, ensure_ascii=False)


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
            description="订���备注（可选）",
            examples=["请尽快发货", "送到门口"]
        )
    ] = None,
) -> str:
    """准备创建订单 - 返回确认信息（JSON格式）"""
    try:
        # 解析商品列表
        try:
            items_data = json.loads(items)
        except json.JSONDecodeError:
            return json.dumps({
                "text": "商品列表格式错误，请使用JSON格式",
                "can_create": False
            }, ensure_ascii=False)

        with get_db_session() as db:
            total_amount = Decimal("0")
            items_preview = []

            for item in items_data:
                product = get_product_by_id(db, item["product_id"])
                if not product:
                    return json.dumps({
                        "text": f"未找到ID为 {item['product_id']} 的商品",
                        "can_create": False
                    }, ensure_ascii=False)

                stock = product.stock or 0
                if stock < item["quantity"]:
                    return json.dumps({
                        "text": f"商品 {product.name} 库存不足 (库存: {stock}, 需要: {item['quantity']})",
                        "can_create": False
                    }, ensure_ascii=False)

                price = product.price or Decimal("0")
                subtotal = price * item["quantity"]
                total_amount += subtotal

                items_preview.append({
                    "product_id": product.id,
                    "name": product.name,
                    "quantity": item["quantity"],
                    "price": float(price),
                    "subtotal": float(subtotal),
                })

            text_lines = [
                f"确认订单信息",
                f"用户手机号: {user_phone}",
                f"商品清单:",
            ]
            for item in items_preview:
                text_lines.append(f"  • {item['name']} x {item['quantity']} = ¥{item['subtotal']:.2f}")
            text_lines.append(f"总金额: ¥{float(total_amount):.2f}")
            if notes:
                text_lines.append(f"备注: {notes}")
            text_lines.append(f"请确认：是否创建此订单？")

            return json.dumps({
                "text": "\n".join(text_lines),
                "can_create": True,
                "user_phone": user_phone,
                "items": items_preview,
                "total_amount": float(total_amount),
                "notes": notes
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "text": f"准备创建订单时出错: {str(e)}",
            "can_create": False
        }, ensure_ascii=False)


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
    """确认创建订单 - 执行实际的创建操作（JSON格式）"""
    import logging
    logger = logging.getLogger(__name__)

    try:
        items_data = json.loads(items)

        # 添加调试日志
        logger.info(f"✅ [ORDER_CREATE] 开始创建订单")
        logger.info(f"✅ [ORDER_CREATE] 手机号参数: '{user_phone}' (类型: {type(user_phone).__name__}, 长度: {len(user_phone)})")
        logger.info(f"✅ [ORDER_CREATE] 商品列表: {items}")
        logger.info(f"✅ [ORDER_CREATE] 备注: {notes}")

        with get_db_session() as db:
            order = create_order_db(
                db,
                user_phone=user_phone,
                items=items_data,
                notes=notes,
            )

            # 添加调试日志
            logger.info(f"✅ [ORDER_CREATE] 订单创建成功!")
            logger.info(f"  - 订单ID: {order.id}")
            logger.info(f"  - 订单号: {order.order_id}")
            logger.info(f"  - 保存的手机号: '{order.user_id}'")
            logger.info(f"  - 总金额: {order.total_amount}")
            logger.info(f"  - 状态: {order.status}")

            return json.dumps({
                "text": f"订单创建成功！订单号: {order.order_id}, 金额: ¥{float(order.total_amount):.2f}",
                "success": True,
                "order_id": order.id,
                "order_number": order.order_id,
                "total_amount": float(order.total_amount) if order.total_amount else 0
            }, ensure_ascii=False)

    except ValueError as e:
        return json.dumps({
            "text": f"创建失败: {str(e)}",
            "success": False
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "text": f"创建订单时出错: {str(e)}",
            "success": False
        }, ensure_ascii=False)


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
    """更新订单状态（管理员功能，JSON格式）"""
    try:
        from src.db.crud import update_order_status as update_status_db

        valid_statuses = ["pending", "paid", "shipped", "delivered", "cancelled"]
        if status not in valid_statuses:
            return json.dumps({
                "text": f"无效的状态，可选值: {', '.join(valid_statuses)}",
                "success": False
            }, ensure_ascii=False)

        with get_db_session() as db:
            order = update_status_db(db, order_id, status)
            if not order:
                return json.dumps({
                    "text": f"未找到ID为 {order_id} 的订单",
                    "success": False
                }, ensure_ascii=False)

            return json.dumps({
                "text": f"订单 {order.order_id} 状态已更新为 {status}",
                "success": True
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "text": f"更新订单状态时出错: {str(e)}",
            "success": False
        }, ensure_ascii=False)


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
