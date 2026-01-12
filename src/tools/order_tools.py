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
def query_order(
    user_id: Annotated[
        str,
        Field(
            description="用户ID（session_id，必填）",
            examples=["default", "session_123"]
        )
    ],
    order_id: Annotated[
        Optional[str],
        Field(
            default=None,
            description="订单ID（可选），支持订单号格式（如ORD123456）或纯数字字符串（如'123'）。如果提供，优先查询特定订单并验证权限；如果不提供，查询用户所有订单",
            examples=["ORD123456", "ORD789012", "123", "456"]
        )
    ] = None,
    status: Annotated[
        Optional[str],
        Field(
            default=None,
            description="订单状态筛选（仅在查询所有订单时生效），可选值: pending(待支付)/paid(已支付)/shipped(已发货)/delivered(已收货)/cancelled(已取消)",
            examples=["pending", "paid", "delivered"]
        )
    ] = None,
    limit: Annotated[
        int,
        Field(
            default=20,
            description="返回结果数量限制（仅在查询所有订单时生效）",
            examples=[10, 20, 50]
        )
    ] = 20,
) -> str:
    """查询订单（统一接口）

    统一的订单查询接口：
    - 如果提供了 order_id：优先查询特定订单详情，并验证订单归属权限
    - 如果没有提供 order_id：查询用户所有订单列表
    
    返回 JSON 格式，包含人类可读文本和结构化订单数据。
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"🔍 [ORDER_QUERY] 开始查询订单: user_id={user_id}, order_id={order_id}")

        with get_db_session() as db:
            # 如果提供了 order_id，优先查询特定订单详情
            if order_id:
                logger.info(f"🔍 [ORDER_QUERY] 查询特定订单: order_id={order_id}")
                
                # 判断order_id是纯数字还是包含字母的订单号
                if order_id.isdigit():
                    # 纯数字，作为数据库主键ID查询
                    order = get_order_by_id(db, int(order_id), refresh=True)
                else:
                    # 包含字母，作为订单号（order_id字段）查询
                    order = get_order_by_number(db, order_id)

                if not order:
                    logger.warning(f"🔍 [ORDER_QUERY] 未找到订单: order_id={order_id}")
                    return json.dumps({
                        "text": f"未找到订单 (订单ID: {order_id})，请确认订单号是否正确。",
                        "orders": [],
                        "order": None
                    }, ensure_ascii=False)

                # 权限验证：验证订单是否属于该用户
                if order.user_id != user_id:
                    logger.warning(f"🔒 [ORDER_QUERY] 权限验证失败: order_id={order_id}, order.user_id={order.user_id}, request.user_id={user_id}")
                    return json.dumps({
                        "text": f"您没有权限查看订单 {order_id}，该订单不属于您。",
                        "orders": [],
                        "order": None
                    }, ensure_ascii=False)

                logger.info(f"✅ [ORDER_QUERY] 权限验证通过: order_id={order_id}")

                display = OrderDisplay.from_db(order)

                # 构建结构化订单数据
                order_items = []
                logger.info(f"🔍 [ORDER_QUERY] 订单包含 {len(order.order_items)} 个订单项")
                for idx, item in enumerate(order.order_items):
                    logger.info(f"🔍 [ORDER_QUERY] 订单项 {idx+1}: order_item_id={item.id}, product_id={item.product_id}, quantity={item.quantity}, price={item.price}")
                    
                    # 检查 product 关系是否正确加载
                    if item.product:
                        logger.info(f"🔍 [ORDER_QUERY] 订单项 {idx+1} 的产品信息: product_id={item.product.id}, name={item.product.name}, price={item.product.price}")
                    else:
                        logger.warning(f"🔍 [ORDER_QUERY] 订单项 {idx+1} 的 product 关系未加载或为 None! product_id={item.product_id}")
                    
                    product_images = []
                    if item.product and item.product.images:
                        if isinstance(item.product.images, list):
                            product_images = item.product.images
                        elif isinstance(item.product.images, dict):
                            product_images = [v for v in item.product.images.values() if isinstance(v, str)]
                    
                    product_name = item.product.name if item.product else "未知商品"
                    logger.info(f"🔍 [ORDER_QUERY] 订单项 {idx+1} 最终显示: product_name={product_name}, quantity={item.quantity}, subtotal={float(item.price * item.quantity):.2f}")
                    
                    order_items.append({
                        "product_id": item.product_id,  # 添加 product_id 用于调试
                        "product_name": product_name,
                        "quantity": item.quantity,
                        "subtotal": float(item.price * item.quantity),
                        "product_images": product_images,
                    })

                order_data = {
                    "id": order.id,
                    "order_number": order.order_id,
                    "status": order.status,
                    "total_amount": float(order.total_amount) if order.total_amount else 0,
                    "created_at": order.created_at.isoformat() if order.created_at else None,
                    "items": order_items,
                }

                # 生成人类可读文本（单个订单详情）
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

                logger.info(f"📋 [ORDER_QUERY] 查询订单详情完成: id={order.id}, order_number={order.order_id}, status={order.status}")

                return json.dumps({
                    "text": text,
                    "orders": [order_data],  # 包装成列表，保持结构一致
                    "order": order_data
                }, ensure_ascii=False)

            # 如果没有提供 order_id，查询用户所有订单列表
            logger.info(f"🔍 [ORDER_QUERY] 查询用户所有订单: user_id={user_id}, status={status}, limit={limit}")
            orders = get_user_orders(db, user_id, status=status, limit=limit)

            logger.info(f"🔍 [ORDER_QUERY] 查询结果: 找到 {len(orders)} 个订单")

            # 构建结构化订单数据
            orders_data = []
            for order in orders:
                order_items = []
                for item in order.order_items:
                    product_images = []
                    if item.product and item.product.images:
                        if isinstance(item.product.images, list):
                            product_images = item.product.images
                        elif isinstance(item.product.images, dict):
                            product_images = [v for v in item.product.images.values() if isinstance(v, str)]
                    
                    order_items.append({
                        "product_name": item.product.name if item.product else "未知商品",
                        "quantity": item.quantity,
                        "subtotal": float(item.price * item.quantity),
                        "product_images": product_images,
                    })
                order_data_item = {
                    "id": order.id,
                    "order_number": order.order_id,
                    "status": order.status,
                    "total_amount": float(order.total_amount) if order.total_amount else 0,
                    "created_at": order.created_at.isoformat() if order.created_at else None,
                    "items": order_items,
                }
                orders_data.append(order_data_item)
                logger.info(f"📋 [ORDER_QUERY] 构建订单数据: id={order.id}, order_number={order.order_id}, status={order.status}")

            # 生成人类可读文本
            if not orders:
                status_msg = f"(状态: {status})" if status else ""
                text = f"暂无订单{status_msg}"
            else:
                result_lines = [f"您的订单 (共{len(orders)}个):\n"]
                for i, order in enumerate(orders, 1):
                    display = OrderDisplay.from_db(order)
                    result_lines.append(f"{i}. {display.format_text()}\n")
                text = "\n".join(result_lines)

            logger.info(f"📋 [ORDER_QUERY] 返回订单数据: 共{len(orders_data)}个订单")

            return json.dumps({
                "text": text,
                "orders": orders_data,
                "order": None
            }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"❌ [ORDER_QUERY] 查询订单时出错: {e}", exc_info=True)
        return json.dumps({
            "text": f"查询订单时出错: {str(e)}",
            "orders": [],
            "order": None
        }, ensure_ascii=False)


@tool
def prepare_cancel_order(
    order_id: Annotated[
        str,
        Field(
            description="要取消的订单ID（订单号，字符串格式，如ORD123456或纯数字字符串如'123'）",
            examples=["ORD123456", "ORD789012", "123", "456"]
        )
    ],
    user_id: Annotated[
        str,
        Field(
            description="用户ID（session_id，用于验证权限）",
            examples=["default", "session_123"]
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
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"🔍 [PREPARE_CANCEL] 开始准备取消订单: order_id={order_id}, user_id={user_id}")
        
        with get_db_session() as db:
            # 判断order_id是纯数字还是包含字母的订单号
            if order_id.isdigit():
                # 纯数字，作为数据库主键ID查询
                order = get_order_by_id(db, int(order_id), refresh=True)
            else:
                # 包含字母，作为订单号（order_id字段）查询
                order = get_order_by_number(db, order_id)
            
            if not order:
                logger.warning(f"🔍 [PREPARE_CANCEL] 未找到订单: order_id={order_id}")
                return json.dumps({
                    "text": f"未找到订单 {order_id}，请确认订单号是否正确。",
                    "can_cancel": False
                }, ensure_ascii=False)

            if order.user_id != user_id:
                logger.warning(f"🔒 [PREPARE_CANCEL] 权限验证失败: order_id={order_id}, order.user_id={order.user_id}, request.user_id={user_id}")
                return json.dumps({
                    "text": f"无权取消此订单（订单属于其他用户）",
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

            # 返回订单号（业务标识），而不是数据库主键ID
            logger.info(f"✅ [PREPARE_CANCEL] 准备取消订单成功: order_number={order.order_id}")
            return json.dumps({
                "text": "\n".join(text_lines),
                "can_cancel": True,
                "order_id": order.order_id  # 返回订单号（字符串）
            }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"❌ [PREPARE_CANCEL] 准备取消订单时出错: {e}", exc_info=True)
        return json.dumps({
            "text": f"准备取消订单时出错: {str(e)}",
            "can_cancel": False
        }, ensure_ascii=False)


@tool
def confirm_cancel_order(
    order_id: Annotated[
        str,
        Field(
            description="要取消的订单ID（订单号，字符串格式，如ORD123456或纯数字字符串如'123'）",
            examples=["ORD123456", "ORD789012", "123", "456"]
        )
    ],
    user_id: Annotated[
        str,
        Field(
            description="用户ID（session_id，用于验证权限）",
            examples=["default", "session_123"]
        )
    ],
) -> str:
    """确认取消订单 - 执行实际的取消操作（JSON格式）"""
    import logging
    logger_cancel = logging.getLogger(__name__)
    
    try:
        logger_cancel.info(f"🚫 [CANCEL_ORDER] 开始取消订单: order_id={order_id}, user_id={user_id}")
        
        with get_db_session() as db:
            # 判断order_id是纯数字还是包含字母的订单号
            if order_id.isdigit():
                # 纯数字，作为数据库主键ID查询
                order_before = get_order_by_id(db, int(order_id))
            else:
                # 包含字母，作为订单号（order_id字段）查询
                order_before = get_order_by_number(db, order_id)
            
            if order_before:
                logger_cancel.info(f"🚫 [CANCEL_ORDER] 取消前状态: order_id={order_before.id}, status={order_before.status}, order_number={order_before.order_id}")
                # 获取数据库主键ID，用于调用cancel_order_db
                db_order_id = order_before.id
            else:
                logger_cancel.warning(f"🚫 [CANCEL_ORDER] 取消前未找到订单: order_id={order_id}")
                return json.dumps({
                    "text": f"未找到订单 {order_id}，请确认订单号是否正确。",
                    "success": False
                }, ensure_ascii=False)
            
            # 使用数据库主键ID查询订单（确保获取最新状态）
            order = get_order_by_id(db, db_order_id, refresh=True)
            if not order:
                logger_cancel.error(f"🚫 [CANCEL_ORDER] 未找到订单: order_id={order_id}, db_id={db_order_id}")
                return json.dumps({
                    "text": f"未找到订单 {order_id}",
                    "success": False
                }, ensure_ascii=False)

            if order.user_id != user_id:
                logger_cancel.warning(f"🚫 [CANCEL_ORDER] 权限验证失败: order.user_id={order.user_id}, user_id={user_id}")
                return json.dumps({
                    "text": "无权取消此订单",
                    "success": False
                }, ensure_ascii=False)

            # 【关键日志】执行取消操作（使用数据库主键ID）
            logger_cancel.info(f"🚫 [CANCEL_ORDER] 执行取消操作: order_number={order.order_id}, db_id={order.id}, 当前状态={order.status}")
            order = cancel_order_db(db, db_order_id)
            if not order:
                logger_cancel.error(f"🚫 [CANCEL_ORDER] 取消操作失败: order_id={order_id}, db_id={db_order_id}")
                return json.dumps({
                    "text": f"无法取消订单 {order_id}",
                    "success": False
                }, ensure_ascii=False)
            
            # 【关键日志】取消后的状态
            logger_cancel.info(f"🚫 [CANCEL_ORDER] 取消后状态: order_id={order.id}, status={order.status}, order_number={order.order_id}")
            
            # 上下文管理器会自动提交事务
            # 为了确保获取最新状态，在提交后重新查询（使用新的查询会从数据库读取最新数据）
            # 由于事务会在退出 with 块时提交，这里返回的对象状态应该是正确的
            
            result_data = {
                "text": f"订单 {order.order_id} 已成功取消",
                "success": True,
                "order_id": order.order_id,  # 返回订单号（业务标识）
                "order_status": order.status  # 明确返回状态，用于前端显示
            }
            
            logger_cancel.info(f"🚫 [CANCEL_ORDER] 返回结果: success={result_data['success']}, order_status={result_data['order_status']}")
            
            return json.dumps(result_data, ensure_ascii=False)

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
    user_id: Annotated[
        str,
        Field(
            description="用户ID（session_id）",
            examples=["default", "session_123"]
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

                # 处理产品图片
                product_images = []
                if product.images:
                    if isinstance(product.images, list):
                        product_images = product.images
                    elif isinstance(product.images, dict):
                        product_images = [v for v in product.images.values() if isinstance(v, str)]

                items_preview.append({
                    "product_id": product.id,
                    "name": product.name,
                    "quantity": item["quantity"],
                    "price": float(price),
                    "subtotal": float(subtotal),
                    "product_images": product_images,
                })

            text_lines = [
                f"确认订单信息",
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
                "user_id": user_id,
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
    user_id: Annotated[
        str,
        Field(
            description="用户ID（session_id）",
            examples=["default", "session_123"]
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
        logger.info(f"✅ [ORDER_CREATE] 用户ID参数: '{user_id}' (类型: {type(user_id).__name__}, 长度: {len(user_id)})")
        logger.info(f"✅ [ORDER_CREATE] 商品列表: {items}")
        logger.info(f"✅ [ORDER_CREATE] 备注: {notes}")

        with get_db_session() as db:
            order = create_order_db(
                db,
                user_id=user_id,  # 使用 user_id (session_id) 作为用户标识
                items=items_data,
                notes=notes,
            )

            # 添加调试日志
            logger.info(f"✅ [ORDER_CREATE] 订单创建成功!")
            logger.info(f"  - 订单ID: {order.id}")
            logger.info(f"  - 订单号: {order.order_id}")
            logger.info(f"  - 保存的用户ID: '{order.user_id}'")
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
        query_order,  # 统一的订单查询工具
        prepare_cancel_order,
        confirm_cancel_order,
        prepare_create_order,
        confirm_create_order,
        update_order_status,
    ]
