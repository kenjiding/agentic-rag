"""CRUD 操作封装 - 适配现有数据库

提供对数据库表的增删改查操作。
"""

from decimal import Decimal
from typing import List, Optional
import random

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, joinedload

from .models import (
    Brand,
    MainCategory,
    SubCategory,
    Product,
    Order,
    OrderItem,
)


# ============== Brand CRUD ==============

def get_brand_by_name(db: Session, name: str) -> Optional[Brand]:
    """根据名称获取品牌"""
    return db.execute(
        select(Brand).where(Brand.name == name)
    ).scalar_one_or_none()


# ============== Category CRUD ==============

def get_main_category_by_name(db: Session, name: str) -> Optional[MainCategory]:
    """根据名称获取主分类"""
    return db.execute(
        select(MainCategory).where(MainCategory.name == name)
    ).scalar_one_or_none()


def get_sub_category_by_name(db: Session, name: str) -> Optional[SubCategory]:
    """根据名称获取子分类"""
    return db.execute(
        select(SubCategory).where(SubCategory.name == name)
    ).scalar_one_or_none()


# ============== Product CRUD ==============

def search_products(
    db: Session,
    name: Optional[str] = None,
    category: Optional[str] = None,
    sub_category: Optional[str] = None,
    brand: Optional[str] = None,
    price_min: Optional[Decimal] = None,
    price_max: Optional[Decimal] = None,
    min_rating: Optional[float] = None,
    in_stock_only: bool = False,
    special_only: bool = False,
    limit: int = 10,
) -> List[Product]:
    """搜索商品（支持多条件组合）

    Args:
        db: 数据库会话
        name: 商品名称或型号关键词（模糊搜索）
        category: 主分类名称
        sub_category: 子分类名称
        brand: 品牌名称
        price_min: 最低价格
        price_max: 最高价格
        min_rating: 最低评分
        in_stock_only: 仅显示有货商品
        special_only: 仅显示特价商品
        limit: 返回数量限制

    Returns:
        商品列表
    """
    query = select(Product).options(
        joinedload(Product.brand),
        joinedload(Product.main_category),
        joinedload(Product.sub_category),
    )

    conditions = []

    if name:
        conditions.append(
            or_(
                Product.name.ilike(f"%{name}%"),
                Product.model_number.ilike(f"%{name}%"),
                Product.features.ilike(f"%{name}%"),
                Product.description.ilike(f"%{name}%"),
            )
        )

    if category:
        query = query.join(MainCategory, Product.main_category_id == MainCategory.id)
        conditions.append(MainCategory.name == category)

    if sub_category:
        query = query.join(SubCategory, Product.sub_category_id == SubCategory.id)
        conditions.append(SubCategory.name == sub_category)

    if brand:
        query = query.join(Brand, Product.brand_id == Brand.id)
        conditions.append(Brand.name == brand)

    if price_min is not None:
        conditions.append(Product.price >= price_min)

    if price_max is not None:
        conditions.append(Product.price <= price_max)

    # 评分筛选需要通过 features 字段模糊匹配（数据库中没有单独的 rating 字段）
    if min_rating is not None:
        # 将评分转换为整数范围进行模糊匹配
        rating_threshold = int(min_rating)
        conditions.append(
            or_(
                Product.features.like(f"%评分:%{rating_threshold}%%"),
                Product.features.like(f"%评分:%{rating_threshold + 1}%%"),
                Product.features.like(f"%评分:5%"),  # 最高评分
            )
        )

    if in_stock_only:
        conditions.append(
            or_(Product.stock > 0, Product.stock.is_(None))
        )

    if special_only:
        conditions.append(Product.special == True)

    if conditions:
        query = query.where(and_(*conditions))

    query = query.order_by(Product.id).limit(limit)

    result = db.execute(query).unique().scalars().all()
    return list(result)


def get_product_by_id(db: Session, product_id: int) -> Optional[Product]:
    """根据 ID 获取商品"""
    return db.execute(
        select(Product)
        .options(
            joinedload(Product.brand),
            joinedload(Product.main_category),
            joinedload(Product.sub_category),
        )
        .where(Product.id == product_id)
    ).unique().scalar_one_or_none()


# ============== Order CRUD ==============

def get_order_by_id(db: Session, order_id: int, refresh: bool = False) -> Optional[Order]:
    """根据 ID 获取订单
    
    Args:
        db: 数据库会话
        order_id: 订单 ID
        refresh: 是否强制从数据库刷新对象（用于确保获取最新状态）
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # SQLAlchemy 2.0+: 使用 joinedload 加载集合关系时需要调用 unique() 去重
    order = db.execute(
        select(Order)
        .options(joinedload(Order.order_items).joinedload(OrderItem.product))
        .where(Order.id == order_id)
    ).unique().scalar_one_or_none()
    
    if order:
        logger.info(f"💾 [DB_GET_ORDER] 查询订单: order_id={order_id}, 找到订单, status={order.status}, refresh={refresh}")
    else:
        logger.warning(f"💾 [DB_GET_ORDER] 查询订单: order_id={order_id}, 未找到订单")
    
    # 如果需要强制刷新（例如在状态更新后查询）
    if order and refresh:
        # 先过期对象，然后刷新，确保从数据库重新加载
        old_status = order.status
        db.expire(order)
        db.refresh(order)
        logger.info(f"💾 [DB_GET_ORDER] 刷新订单: order_id={order_id}, 刷新前status={old_status}, 刷新后status={order.status}")
    
    return order


def get_order_by_number(db: Session, order_number: str) -> Optional[Order]:
    """根据订单号获取订单"""
    # SQLAlchemy 2.0+: 使用 joinedload 加载集合关系时需要调用 unique() 去重
    return db.execute(
        select(Order)
        .options(joinedload(Order.order_items))
        .where(Order.order_id == order_number)
    ).unique().scalar_one_or_none()


def get_user_orders(
    db: Session,
    user_phone: str,
    status: Optional[str] = None,
    limit: int = 20,
) -> List[Order]:
    """获取用户订单列表

    Args:
        db: 数据库会话
        user_phone: 用户手机号（作为 user_id）
        status: 订单状态筛选
        limit: 返回数量限制

    Returns:
        订单列表
    """
    import logging
    logger = logging.getLogger(__name__)
    
    query = select(Order).options(
        joinedload(Order.order_items).joinedload(OrderItem.product),
    ).where(Order.user_id == user_phone)

    if status:
        query = query.where(Order.status == status)

    query = query.order_by(Order.created_at.desc()).limit(limit)

    # 使用 unique() 去重，因为 joinedload 会产生重复行
    result = db.execute(query).unique().scalars().all()
    orders = list(result)
    
    # 【关键日志】记录从数据库查询到的订单状态
    logger.info(f"💾 [DB_QUERY] 查询用户订单: user_phone={user_phone}, status_filter={status}, 找到{len(orders)}个订单")
    for order in orders:
        logger.info(f"  - 订单ID: {order.id}, 订单号: {order.order_id}, 状态: {order.status}")
    
    return orders


def create_order(
    db: Session,
    user_phone: str,
    items: List[dict],  # [{"product_id": 1, "quantity": 2}]
    notes: Optional[str] = None,
) -> Order:
    """创建订单

    Args:
        db: 数据库会话
        user_phone: 用户手机号（作为 user_id）
        items: 订单明细列表
        notes: 订单备注（当前数据库不存储，忽略）

    Returns:
        创建的订单对象
    """
    # 生成订单号
    order_number = f"ORD{random.randint(100000, 999999)}"

    # 计算总金额
    total_amount = Decimal("0")
    order_items = []

    for item in items:
        product = get_product_by_id(db, item["product_id"])
        if not product:
            raise ValueError(f"Product {item['product_id']} not found")

        if product.stock and product.stock < item["quantity"]:
            raise ValueError(f"Product {product.name} insufficient stock")

        price = product.price or Decimal("0")
        subtotal = price * item["quantity"]
        total_amount += subtotal

        order_items.append(
            OrderItem(
                product_id=product.id,
                quantity=item["quantity"],
                price=price,
            )
        )

    # 创建订单
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    order = Order(
        order_id=order_number,
        user_id=user_phone,
        total_amount=total_amount,
        status="pending",
        created_at=now,
        updated_at=now,
        order_items=order_items,
    )
    db.add(order)
    db.flush()

    return order


def cancel_order(db: Session, order_id: int) -> Optional[Order]:
    """取消订单

    Args:
        db: 数据库会话
        order_id: 订单 ID

    Returns:
        取消后的订单对象，失败返回 None
    """
    import logging
    logger = logging.getLogger(__name__)
    
    order = get_order_by_id(db, order_id)
    if not order:
        logger.warning(f"💾 [DB_CANCEL] 未找到订单: order_id={order_id}")
        return None

    # 【关键日志】取消前的状态
    logger.info(f"💾 [DB_CANCEL] 取消前: order_id={order.id}, status={order.status}, order_number={order.order_id}")

    # 只有 pending 状态的订单可以取消
    if order.status != "pending":
        logger.error(f"💾 [DB_CANCEL] 订单状态不允许取消: order_id={order.id}, status={order.status}")
        raise ValueError(f"Cannot cancel order with status {order.status}")

    # 更新状态和更新时间
    from datetime import datetime, timezone
    old_status = order.status
    order.status = "cancelled"
    order.updated_at = datetime.now(timezone.utc)
    
    # 【关键日志】更新后的状态（flush前）
    logger.info(f"💾 [DB_CANCEL] 更新状态: order_id={order.id}, {old_status} -> {order.status}")
    
    # flush 确保更改被保存到当前会话
    db.flush()
    # refresh 确保对象状态与数据库同步
    db.refresh(order)
    
    # 【关键日志】刷新后的状态
    logger.info(f"💾 [DB_CANCEL] 刷新后: order_id={order.id}, status={order.status}, order_number={order.order_id}")
    
    return order


def update_order_status(db: Session, order_id: int, status: str) -> Optional[Order]:
    """更新订单状态

    Args:
        db: 数据库会话
        order_id: 订单 ID
        status: 新状态

    Returns:
        更新后的订单对象
    """
    order = get_order_by_id(db, order_id)
    if not order:
        return None

    order.status = status
    db.flush()
    return order
