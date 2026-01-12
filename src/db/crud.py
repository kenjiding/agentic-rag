"""CRUD 操作封装 - 适配现有数据库

提供对数据库表的增删改查操作。
当USE_TEST_DATA=True时，支持从JSON文件读取测试数据。
"""

import os
import re
from decimal import Decimal
from typing import List, Optional, Union, Dict, Any
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

# 延迟导入测试数据加载器，避免模块级别导入时环境变量未加载的问题
_test_data_loader_imported = False
_load_test_data = None
_get_test_product_by_id = None
_search_test_products = None
_is_use_test_data = None


def _ensure_test_data_loader_imported():
    """确保测试数据加载器已导入（延迟导入，在运行时动态检查）"""
    global _test_data_loader_imported, _load_test_data, _get_test_product_by_id, _search_test_products, _is_use_test_data
    
    if not _test_data_loader_imported:
        from .test_data_loader import (
            load_test_data,
            get_test_product_by_id,
            search_test_products,
            is_use_test_data
        )
        _load_test_data = load_test_data
        _get_test_product_by_id = get_test_product_by_id
        _search_test_products = search_test_products
        _is_use_test_data = is_use_test_data
        _test_data_loader_imported = True


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

def _normalize_string_for_search(text: str) -> str:
    """标准化字符串用于搜索比较（企业级最佳实践）
    
    从源头解决问题：在搜索时对关键词和数据库字段都进行标准化处理，
    而不是生成多个变体。这样可以处理空格、大小写等格式差异。
    
    标准化规则：
    1. 去除所有空格（处理"华为Mate 60 Pro" vs "华为 Mate 60 Pro"）
    2. 转换为小写（处理大小写差异）
    3. 去除首尾空白
    
    注意：这是应用层的标准化，真正的企业级方案应该：
    - 使用PostgreSQL的pg_trgm扩展进行模糊匹配
    - 使用全文搜索引擎（Elasticsearch、Meilisearch等）
    - 在数据入库时进行标准化处理
    
    Args:
        text: 原始文本
        
    Returns:
        标准化后的文本
    """
    if not text:
        return ""
    # 去除所有空格并转换为小写
    return text.replace(' ', '').replace('\t', '').replace('\n', '').lower().strip()


def search_products(
    db: Optional[Session],
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
) -> List[Union[Product, Any]]:
    """搜索商品（支持多条件组合）

    当USE_TEST_DATA=True时，从JSON文件读取测试数据（db参数可忽略）。
    否则从数据库查询真实数据。

    Args:
        db: 数据库会话（测试数据模式下可为None）
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
        商品列表（Product对象或类似Product的对象）
    """
    # 运行时动态检查是否使用测试数据
    _ensure_test_data_loader_imported()
    if _is_use_test_data():
        # 从测试数据搜索
        test_products = _search_test_products(
            name=name,
            category=category,
            sub_category=sub_category,
            brand=brand,
            price_min=float(price_min) if price_min else None,
            price_max=float(price_max) if price_max else None,
            min_rating=min_rating,
            in_stock_only=in_stock_only,
            special_only=special_only,
            limit=limit
        )
        
        # 转换为类似Product的对象结构
        from types import SimpleNamespace
        products = []
        for test_product in test_products:
            product = SimpleNamespace()
            product.id = test_product["id"]
            product.name = test_product["name"]
            product.model_number = test_product.get("model_number")
            product.price = Decimal(str(test_product.get("price", 0)))
            product.stock = test_product.get("stock", 0)
            product.description = test_product.get("description")
            product.features = test_product.get("features")
            product.images = test_product.get("images")
            product.special = test_product.get("special", False)
            product.specifications = test_product.get("specifications")
            product.semantic_tags = test_product.get("semantic_tags")
            product.rating = test_product.get("rating", 0.0)
            
            # 模拟brand关系
            brand_obj = SimpleNamespace()
            brand_obj.name = test_product.get("brand")
            product.brand = brand_obj
            
            # 模拟category关系
            main_category_obj = SimpleNamespace()
            main_category_obj.name = test_product.get("main_category")
            product.main_category = main_category_obj
            
            sub_category_obj = SimpleNamespace()
            sub_category_obj.name = test_product.get("sub_category")
            product.sub_category = sub_category_obj
            
            products.append(product)
        
        return products
    else:
        if db is None:
            raise ValueError("数据库会话不能为None（真实数据模式）")
        # 从数据库查询
        query = select(Product).options(
            joinedload(Product.brand),
            joinedload(Product.main_category),
            joinedload(Product.sub_category),
        )

        conditions = []

        if name:
            # 企业级最佳实践：使用PostgreSQL的字符串函数在查询时标准化比较
            # 对搜索关键词和数据库字段都去除所有空格后比较，从源头解决格式差异问题
            # 这样"华为Mate 60 Pro"和"华为 Mate 60 Pro"都能匹配
            from sqlalchemy import func
            
            # 标准化搜索关键词（去除所有空格、转换为小写）
            normalized_name = _normalize_string_for_search(name)
            
            # 使用PostgreSQL的regexp_replace函数去除所有空白字符（包括空格、制表符、换行符等）
            # 然后转换为小写进行比较，确保能匹配不同格式的产品名称
            # regexp_replace(column, '\s+', '', 'g') 会替换所有连续空白字符
            def normalize_column(column):
                """标准化数据库字段：去除所有空白字符并转换为小写"""
                return func.lower(func.regexp_replace(column, r'\s+', '', 'g'))
            
            name_conditions = [
                normalize_column(Product.name).like(f"%{normalized_name}%"),
                normalize_column(Product.model_number).like(f"%{normalized_name}%"),
                normalize_column(Product.features).like(f"%{normalized_name}%"),
                normalize_column(Product.description).like(f"%{normalized_name}%"),
            ]
            
            conditions.append(or_(*name_conditions))

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


def get_product_by_id(db: Optional[Session], product_id: int) -> Optional[Union[Product, Any]]:
    """根据 ID 获取商品
    
    当USE_TEST_DATA=True时，从JSON文件读取测试数据（db参数可忽略）。
    否则从数据库查询真实数据。
    
    Args:
        db: 数据库会话（测试数据模式下可为None）
        product_id: 产品ID
    """
    # 运行时动态检查是否使用测试数据
    _ensure_test_data_loader_imported()
    if _is_use_test_data():
        # 从测试数据加载
        test_product = _get_test_product_by_id(product_id)
        if test_product:
            # 转换为类似Product的对象结构（使用SimpleNamespace模拟）
            from types import SimpleNamespace
            product = SimpleNamespace()
            product.id = test_product["id"]
            product.name = test_product["name"]
            product.model_number = test_product.get("model_number")
            product.price = Decimal(str(test_product.get("price", 0)))
            product.stock = test_product.get("stock", 0)
            product.description = test_product.get("description")
            product.features = test_product.get("features")
            product.images = test_product.get("images")
            product.special = test_product.get("special", False)
            product.specifications = test_product.get("specifications")
            product.semantic_tags = test_product.get("semantic_tags")
            
            # 模拟brand关系
            brand = SimpleNamespace()
            brand.name = test_product.get("brand")
            product.brand = brand
            
            # 模拟category关系
            main_category = SimpleNamespace()
            main_category.name = test_product.get("main_category")
            product.main_category = main_category
            
            sub_category = SimpleNamespace()
            sub_category.name = test_product.get("sub_category")
            product.sub_category = sub_category
            
            # 添加rating属性（从features解析或直接使用）
            product.rating = test_product.get("rating", 0.0)
            
            return product
        return None
    else:
        # 从数据库查询
        if db is None:
            raise ValueError("数据库会话不能为None（真实数据模式）")
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
    """根据订单号获取订单
    
    Args:
        db: 数据库会话
        order_number: 订单号（如 ORD123456）
    
    Returns:
        订单对象，如果未找到返回 None
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # SQLAlchemy 2.0+: 使用 joinedload 加载集合关系时需要调用 unique() 去重
    # 【关键修复】必须同时加载 order_items 和 product 关系，否则查询订单时无法获取正确的产品信息
    order = db.execute(
        select(Order)
        .options(joinedload(Order.order_items).joinedload(OrderItem.product))
        .where(Order.order_id == order_number)
    ).unique().scalar_one_or_none()
    
    if order:
        logger.info(f"💾 [DB_GET_ORDER_BY_NUMBER] 查询订单: order_number={order_number}, 找到订单, id={order.id}, status={order.status}")
        # 记录订单项信息，用于调试
        for item in order.order_items:
            product_name = item.product.name if item.product else "未知商品"
            logger.info(f"  - 订单项: product_id={item.product_id}, product_name={product_name}, quantity={item.quantity}")
    else:
        logger.warning(f"💾 [DB_GET_ORDER_BY_NUMBER] 查询订单: order_number={order_number}, 未找到订单")
    
    return order


def get_user_orders(
    db: Session,
    user_id: str,
    status: Optional[str] = None,
    limit: int = 20,
) -> List[Order]:
    """获取用户订单列表

    Args:
        db: 数据库会话
        user_id: 用户ID（session_id）
        status: 订单状态筛选
        limit: 返回数量限制

    Returns:
        订单列表
    """
    import logging
    logger = logging.getLogger(__name__)
    
    query = select(Order).options(
        joinedload(Order.order_items).joinedload(OrderItem.product),
    ).where(Order.user_id == user_id)

    if status:
        query = query.where(Order.status == status)

    query = query.order_by(Order.created_at.desc()).limit(limit)

    # 使用 unique() 去重，因为 joinedload 会产生重复行
    result = db.execute(query).unique().scalars().all()
    orders = list(result)
    
    # 【关键日志】记录从数据库查询到的订单状态
    logger.info(f"💾 [DB_QUERY] 查询用户订单: user_id={user_id}, status_filter={status}, 找到{len(orders)}个订单")
    for order in orders:
        logger.info(f"  - 订单ID: {order.id}, 订单号: {order.order_id}, 状态: {order.status}")
    
    return orders


def create_order(
    db: Session,
    user_id: str,
    items: List[dict],  # [{"product_id": 1, "quantity": 2}]
    notes: Optional[str] = None,
) -> Order:
    """创建订单

    Args:
        db: 数据库会话
        user_id: 用户ID（session_id）
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

    import logging
    logger = logging.getLogger(__name__)
    
    for item in items:
        product_id = item["product_id"]
        logger.info(f"💾 [DB_CREATE_ORDER] 处理订单项: product_id={product_id}, quantity={item['quantity']}")
        
        product = get_product_by_id(db, product_id)
        if not product:
            logger.error(f"💾 [DB_CREATE_ORDER] 产品不存在: product_id={product_id}")
            raise ValueError(f"Product {product_id} not found")

        logger.info(f"💾 [DB_CREATE_ORDER] 找到产品: product_id={product.id}, name={product.name}, price={product.price}")
        
        if product.stock and product.stock < item["quantity"]:
            logger.warning(f"💾 [DB_CREATE_ORDER] 库存不足: product_id={product.id}, name={product.name}, stock={product.stock}, need={item['quantity']}")
            raise ValueError(f"Product {product.name} insufficient stock")

        price = product.price or Decimal("0")
        subtotal = price * item["quantity"]
        total_amount += subtotal

        order_item = OrderItem(
            product_id=product.id,  # 使用 product.id 确保使用正确的产品ID
            quantity=item["quantity"],
            price=price,
        )
        logger.info(f"💾 [DB_CREATE_ORDER] 创建订单项: product_id={order_item.product_id}, quantity={order_item.quantity}, price={order_item.price}")
        order_items.append(order_item)

    # 创建订单
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    order = Order(
        order_id=order_number,
        user_id=user_id,  # 使用 user_id (session_id) 作为用户标识
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
