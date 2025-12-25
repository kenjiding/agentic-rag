"""数据库层 - SQLAlchemy + PostgreSQL

提供电商系统的数据库访问能力：
- models: SQLAlchemy ORM 模型
- engine: 连接管理和会话工厂
- crud: CRUD 操作封装
"""

from .engine import init_db, get_session, engine
from .models import (
    Base,
    Brand,
    MainCategory,
    SubCategory,
    Product,
    Order,
    OrderItem,
)

__all__ = [
    "init_db",
    "get_session",
    "engine",
    "Base",
    "Brand",
    "MainCategory",
    "SubCategory",
    "Product",
    "Order",
    "OrderItem",
]
