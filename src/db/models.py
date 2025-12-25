"""SQLAlchemy ORM 模型定义 - 适配现有数据库

电商数据库模型，匹配现有表结构：
- brands: 品牌表
- main_categories: 主分类表
- sub_categories: 子分类表
- products: 商品表
- orders: 订单表
- order_items: 订单明细表
"""

from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    String,
    Integer,
    Numeric,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """所有模型的基类"""
    pass


class Brand(Base):
    """品牌表"""
    __tablename__ = "brands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    image: Mapped[str] = mapped_column(String(255), nullable=False)

    # 关系
    products: Mapped[list["Product"]] = relationship(
        "Product", back_populates="brand"
    )
    main_categories: Mapped[list["MainCategory"]] = relationship(
        "MainCategory", back_populates="brand"
    )


class MainCategory(Base):
    """主分类表"""
    __tablename__ = "main_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    brand_id: Mapped[int] = mapped_column(ForeignKey("brands.id"), nullable=False)

    # 关系
    brand: Mapped["Brand"] = relationship("Brand", back_populates="main_categories")
    sub_categories: Mapped[list["SubCategory"]] = relationship(
        "SubCategory", back_populates="main_category"
    )
    products: Mapped[list["Product"]] = relationship(
        "Product", back_populates="main_category"
    )


class SubCategory(Base):
    """子分类表"""
    __tablename__ = "sub_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    main_category_id: Mapped[int] = mapped_column(
        ForeignKey("main_categories.id"), nullable=False
    )

    # 关系
    main_category: Mapped["MainCategory"] = relationship(
        "MainCategory", back_populates="sub_categories"
    )
    products: Mapped[list["Product"]] = relationship(
        "Product", back_populates="sub_category"
    )


class Product(Base):
    """商品表"""
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    sub_category_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("sub_categories.id"), nullable=True
    )
    main_category_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("main_categories.id"), nullable=True
    )
    brand_id: Mapped[int] = mapped_column(ForeignKey("brands.id"), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    features: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(precision=10, scale=2), nullable=True)
    stock: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    images: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    special: Mapped[bool] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # 关系
    brand: Mapped["Brand"] = relationship("Brand", back_populates="products")
    main_category: Mapped[Optional["MainCategory"]] = relationship(
        "MainCategory", back_populates="products"
    )
    sub_category: Mapped[Optional["SubCategory"]] = relationship(
        "SubCategory", back_populates="products"
    )

    # 评分字段（本地计算，不存储在数据库）
    @property
    def rating(self) -> float:
        """从 features 中解析评分"""
        if self.features:
            import re
            match = re.search(r'评分[:\s]*(\d+\.?\d*)', self.features)
            if match:
                return float(match.group(1))
        return 0.0

    @property
    def review_count(self) -> int:
        """从 features 中解析评论数"""
        if self.features:
            import re
            match = re.search(r'评论[:\s]*(\d+)', self.features)
            if match:
                return int(match.group(1))
        return 0

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "model_number": self.model_number,
            "brand": self.brand.name if self.brand else None,
            "main_category": self.main_category.name if self.main_category else None,
            "sub_category": self.sub_category.name if self.sub_category else None,
            "description": self.description,
            "features": self.features,
            "price": float(self.price) if self.price else None,
            "stock": self.stock or 0,
            "images": self.images,
            "special": self.special if self.special is not None else False,
            "rating": self.rating,
            "review_count": self.review_count,
        }


class Order(Base):
    """订单表"""
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[str] = mapped_column(String(50), nullable=False)  # 订单号
    user_id: Mapped[str] = mapped_column(String(50), nullable=False)  # 用户ID（字符串）
    total_amount: Mapped[Decimal] = mapped_column(Numeric(precision=10, scale=2), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # 关系
    order_items: Mapped[list["OrderItem"]] = relationship(
        "OrderItem", back_populates="order"
    )

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "order_number": self.order_id,
            "user_id": self.user_id,
            "status": self.status,
            "total_amount": float(self.total_amount),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "items": [item.to_dict() for item in self.order_items] if hasattr(self, 'order_items') and self.order_items else [],
        }


class OrderItem(Base):
    """订单明细表"""
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(precision=10, scale=2), nullable=False)

    # 关系
    order: Mapped["Order"] = relationship("Order", back_populates="order_items")
    product: Mapped["Product"] = relationship()

    def to_dict(self) -> dict:
        """转换为字典格式"""
        # 获取商品名称（通过 product 关系）
        product_name = None
        if self.product:
            product_name = self.product.name

        return {
            "id": self.id,
            "order_id": self.order_id,
            "product_id": self.product_id,
            "product_name": product_name,
            "quantity": self.quantity,
            "price": float(self.price),
            "subtotal": float(self.price * self.quantity),
        }
