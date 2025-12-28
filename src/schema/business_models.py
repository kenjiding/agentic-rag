"""业务领域模型定义

使用 Pydantic v2 定义业务工具的输入输出 Schema，
确保类型安全和自动校验。
"""

from decimal import Decimal
from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field


# ============== 商品相关 Schema ==============

class ProductSearchArgs(BaseModel):
    """商品搜索参数 - 支持多条件组合筛选

    LLM 会解析用户自然语言，自动填充这些参数。
    所有参数都是可选的，工具内部会动态构建查询条件。
    """

    # 核心搜索参数
    name: Optional[str] = Field(
        default=None,
        description="商品名称或型号的模糊搜索关键词，如'iPhone'、'Mate60'"
    )

    # 分类筛选
    category: Optional[str] = Field(
        default=None,
        description="主分类名称，如'手机'、'电脑'、'家电'"
    )
    sub_category: Optional[str] = Field(
        default=None,
        description="子分类名称，如'智能手机'、'5G手机'、'笔记本电脑'"
    )
    brand: Optional[str] = Field(
        default=None,
        description="品牌名称，如'Apple'、'华为'、'小米'"
    )

    # 价格范围
    price_min: Optional[Decimal] = Field(
        default=None,
        description="最低价格，单位：元"
    )
    price_max: Optional[Decimal] = Field(
        default=None,
        description="最高价格，单位：元"
    )

    # 其他条件
    min_rating: Optional[float] = Field(
        default=None,
        description="最低评分，范围 1-5"
    )
    in_stock_only: bool = Field(
        default=True,
        description="是否仅显示有货商品"
    )
    special_only: bool = Field(
        default=False,
        description="是否仅显示特价商品"
    )

    limit: int = Field(
        default=10,
        description="返回结果数量限制，默认10条"
    )


class ProductDisplay(BaseModel):
    """商品展示信息"""

    id: int
    name: str
    model_number: Optional[str] = None
    brand: Optional[str] = None
    main_category: Optional[str] = None
    sub_category: Optional[str] = None
    price: Optional[float] = None
    stock: int = 0
    rating: float = 0.0
    special: bool = False
    description: Optional[str] = None
    images: Optional[List[str]] = None

    @classmethod
    def from_db(cls, product: Any) -> "ProductDisplay":
        """从数据库模型转换"""
        # 处理images字段：如果是list就直接用，如果是dict则提取值
        images_list = []
        if product.images:
            if isinstance(product.images, list):
                images_list = product.images
            elif isinstance(product.images, dict):
                # 如果是dict，尝试提取所有值
                images_list = [v for v in product.images.values() if isinstance(v, str)]
        
        return cls(
            id=product.id,
            name=product.name,
            model_number=product.model_number,
            brand=product.brand.name if product.brand else None,
            main_category=product.main_category.name if product.main_category else None,
            sub_category=product.sub_category.name if product.sub_category else None,
            price=float(product.price) if product.price else None,
            stock=product.stock,
            rating=product.rating,
            special=product.special,
            description=product.description,
            images=images_list if images_list else None,
        )

    def format_text(self) -> str:
        """格式化为易读文本"""
        special_mark = " [特价]" if self.special else ""
        stock_info = f"库存: {self.stock}件" if self.stock > 0 else "[缺货]"
        price_info = f"¥{self.price:.2f}" if self.price else "价格面议"

        return (
            f"📦 {self.name}{special_mark}\n"
            f"   品牌: {self.brand or '未知'} | 分类: {self.main_category or '未知'}/{self.sub_category or '未知'}\n"
            f"   价格: {price_info} | 评分: {'⭐' * int(self.rating)}{self.rating:.1f}\n"
            f"   {stock_info}"
        )


class ProductListResult(BaseModel):
    """商品搜索结果"""

    products: List[ProductDisplay]
    total: int
    query_summary: str  # 搜索条件摘要


# ============== 订单相关 Schema ==============

class OrderQueryArgs(BaseModel):
    """订单查询参数"""

    user_phone: Optional[str] = Field(
        default=None,
        description="用户手机号"
    )
    order_id: Optional[int] = Field(
        default=None,
        description="订单ID"
    )
    order_number: Optional[str] = Field(
        default=None,
        description="订单号，如 ORD123456"
    )
    status: Optional[str] = Field(
        default=None,
        description="订单状态筛选: pending/paid/shipped/delivered/cancelled"
    )
    limit: int = Field(
        default=20,
        description="返回结果数量限制"
    )


class OrderDisplay(BaseModel):
    """订单展示信息"""

    id: int
    order_number: str
    status: str
    total_amount: float
    created_at: str
    items: List[dict] = []

    @classmethod
    def from_db(cls, order: Any) -> "OrderDisplay":
        """从数据库模型转换"""
        return cls(
            id=order.id,
            order_number=order.order_id,  # 修复: Order模型字段是order_id不是order_number
            status=order.status,
            total_amount=float(order.total_amount),
            created_at=order.created_at.strftime("%Y-%m-%d %H:%M:%S") if order.created_at else "",
            items=[item.to_dict() for item in order.order_items],
        )

    def format_text(self) -> str:
        """格式化为易读文本"""
        status_emoji = {
            "pending": "⏳ 待支付",
            "paid": "💰 已支付",
            "shipped": "🚚 已发货",
            "delivered": "✅ 已收货",
            "cancelled": "❌ 已取消",
        }.get(self.status, self.status)

        items_text = "\n".join([
            f"   - {item.get('product_name', 'Unknown')} x {item['quantity']} = ¥{item['subtotal']:.2f}"
            for item in self.items
        ])

        return (
            f"📋 订单: {self.order_number} (ID: {self.id})\n"
            f"   状态: {status_emoji}\n"
            f"   商品:\n{items_text}\n"
            f"   总金额: ¥{self.total_amount:.2f}\n"
            f"   创建时间: {self.created_at}"
        )


class OrderCancelArgs(BaseModel):
    """订单取消参数"""

    order_id: int = Field(
        description="要取消的订单ID"
    )
    reason: Optional[str] = Field(
        default=None,
        description="取消原因"
    )
    user_phone: str = Field(
        description="用户手机号，用于验证权限"
    )


class OrderCreateItem(BaseModel):
    """订单商品项"""

    product_id: int = Field(description="商品ID")
    quantity: int = Field(description="购买数量", ge=1)


class OrderCreateArgs(BaseModel):
    """订单创建参数"""

    user_phone: str = Field(description="用户手机号")
    items: List[OrderCreateItem] = Field(description="商品列表")
    notes: Optional[str] = Field(default=None, description="订单备注")


# ============== 确认机制 Schema ==============

class ConfirmationRequest(BaseModel):
    """确认请求

    当 Agent 需要用户确认操作时返回此结构。
    """

    action_type: Literal["cancel_order", "create_order"] = Field(
        description="操作类型"
    )
    data: dict = Field(
        description="操作相关数据"
    )
    message: str = Field(
        description="向用户展示的确认消息"
    )


class ConfirmationResponse(BaseModel):
    """用户确认响应"""

    confirmed: bool = Field(description="用户是否确认")
    action_type: str = Field(description="操作类型")
    data: dict = Field(default_factory=dict, description="操作相关数据")


# ============== 工具执行结果 Schema ==============

class ToolResult(BaseModel):
    """工具执行结果"""

    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None
    requires_confirmation: Optional[ConfirmationRequest] = None


# ============== 意图识别扩展 ==============

class BusinessIntent(BaseModel):
    """业务意图识别结果"""

    intent_type: Literal["product_search", "order_query", "order_cancel", "order_create", "general"] = Field(
        description="意图类型"
    )
    confidence: float = Field(description="置信度 0-1")
    entities: dict = Field(default_factory=dict, description="提取的实体参数")


# ============== 状态扩展 ==============

class BusinessContext(BaseModel):
    """业务上下文 - 存储在 MultiAgentState 中"""

    # 商品搜索上下文
    last_search_results: Optional[List[ProductDisplay]] = None

    # 订单操作上下文
    pending_order: Optional[dict] = None
    pending_cancel: Optional[dict] = None

    # 确认机制
    awaiting_confirmation: Optional[ConfirmationRequest] = None


# ============== 结构化输出 Schema ==============

class SupervisorDecision(BaseModel):
    """Supervisor 路由决策结构化输出"""

    next_agent: Literal["rag_agent", "product_agent", "order_agent", "chat_agent", "finish"] = Field(
        description="下一个要执行的 Agent"
    )
    reasoning: str = Field(description="路由决策的原因")
    business_intent: Optional[BusinessIntent] = Field(
        default=None,
        description="识别到的业务意图"
    )
