"""Agent响应模型标准化定义

设计原则（2025-2026终极重构）：
1. 所有Agent返回完整的、前端可直接使用的ResponseModel
2. BaseResponseModel包含所有通用字段（content、role、response_type等）
3. 具体ResponseModel继承基类并添加特定字段
4. format_state_update只做SSE封装，零业务逻辑
5. 单一数据源：ResponseModel是前端数据的唯一来源

企业级规范：
- 使用Pydantic提供类型安全和自动验证
- Agent构建ResponseModel后，通过to_full_response()获取完整数据
- 包含AI消息content、role、response_type、response_data等所有字段
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field


class BaseResponseModel(BaseModel):
    """Agent响应基类 - 包含完整的前端数据

    所有Agent的响应数据都必须继承此类。
    提供统一的接口和类型安全保证。

    设计原则：
    - 单一数据源：ResponseModel是前端数据的唯一来源
    - 完整性：包含前端需要的所有字段（content、role、response_type等）
    - 格式化透传：format_state_update只做SSE封装，零业务逻辑
    - 类型安全：Pydantic提供运行时验证

    通用字段：
    - response_type: 响应类型标识
    - content: AI消息内容（可选，用于文本响应）
    - role: 角色，固定为"assistant"
    - response_data: 结构化数据字典（自动从特定字段生成）
    """

    response_type: str = Field(description="响应类型标识（如text、product_list、order_list等）")
    content: Optional[str] = Field(default=None, description="AI消息内容")
    role: str = Field(default="assistant", description="角色，固定为assistant")
    response_data: Dict[str, Any] = Field(default_factory=dict, description="结构化响应数据")

    def to_full_response(self) -> Dict[str, Any]:
        """生成完整的前端响应数据

        将ResponseModel转换为前端可直接使用的完整数据格式。
        包含所有必要字段，可直接展开到Agent返回值中。

        设计原则（2025-2026终极重构）：
        - response_data 自动从特定字段生成（products/orders等）
        - 前端只需要访问 response_data，不需要知道具体的字段名
        - 通用字段（response_type, content, role）在根级别

        Returns:
            完整的前端响应数据字典

        Example:
            >>> response = ProductListResponse(
            ...     products=[...],
            ...     total=10,
            ...     content="找到10个产品"
            ... )
            >>> full_response = response.to_full_response()
            >>> # full_response = {
            >>> #     "response_type": "product_list",
            >>> #     "response_data": {"products": [...], "total": 10},
            >>> #     "content": "找到10个产品",
            >>> #     "role": "assistant"
            >>> # }
        """
        # 获取特定字段名（子类可以覆盖）
        response_data_fields = self._get_response_data_fields()

        # 构建response_data
        response_data = {}
        for field in response_data_fields:
            value = getattr(self, field, None)
            if value is not None and value != [] and value != "":
                response_data[field] = value

        # 构建完整响应（排除已经放入response_data的字段）
        # 注意：只使用 exclude_none=True，不使用 exclude_unset=True
        # 因为有默认值的字段（如 response_type）必须包含在响应中
        all_fields = self.model_dump(exclude_none=True)
        exclude_fields = set(response_data_fields)

        clean_response = {
            k: v for k, v in all_fields.items()
            if k not in exclude_fields
        }

        # 设置response_data
        clean_response["response_data"] = response_data

        return clean_response

    def _get_response_data_fields(self) -> List[str]:
        """获取应该放入response_data的字段名

        子类可以覆盖此方法来指定哪些字段应该放入response_data。
        默认返回空列表（基类TextResponse没有结构化数据）。

        Returns:
            字段名列表
        """
        return []

    @classmethod
    def get_response_type(cls) -> str:
        """获取此模型的响应类型

        子类必须实现此方法以提供响应类型标识。
        """
        raise NotImplementedError("Subclasses must implement get_response_type")


class TextResponse(BaseResponseModel):
    """纯文本响应

    用于ChatAgent、RAGAgent等只返回文本的Agent。

    Fields:
        content: AI生成的文本内容（必需）

    Example:
        >>> response = TextResponse(content="你好，有什么可以帮助你的？")
        >>> full_response = response.to_full_response()
    """

    response_type: Literal["text"] = "text"
    content: str = Field(description="AI生成的文本内容")

    @classmethod
    def get_response_type(cls) -> str:
        return "text"


class ProductListResponse(BaseResponseModel):
    """商品列表响应

    用于ProductAgent返回商品搜索结果。

    Fields:
        products: 商品列表，每个商品包含id、name、price等信息
        total: 商品总数量
        query_summary: 搜索条件摘要（可选）
        content: AI消息内容（可选，如"找到10个商品"）

    Example:
        >>> response = ProductListResponse(
        ...     products=[{"id": 1, "name": "iPhone", "price": 7999}],
        ...     total=1,
        ...     query_summary="搜索结果",
        ...     content="找到1个商品"
        ... )
        >>> full_response = response.to_full_response()
        >>> # full_response = {
        >>> #     "response_type": "product_list",
        >>> #     "response_data": {"products": [...], "total": 1, "query_summary": "..."},
        >>> #     "content": "找到1个商品",
        >>> #     "role": "assistant"
        >>> # }
    """

    response_type: Literal["product_list"] = "product_list"
    products: List[Dict[str, Any]] = Field(description="商品列表")
    total: int = Field(description="总数量", default=0)
    query_summary: str = Field(description="搜索条件摘要", default="")

    def _get_response_data_fields(self) -> List[str]:
        """指定哪些字段应该放入response_data"""
        return ["products", "total", "query_summary"]

    @classmethod
    def get_response_type(cls) -> str:
        return "product_list"


class OrderListResponse(BaseResponseModel):
    """订单列表响应

    用于OrderAgent返回订单查询结果。

    Fields:
        orders: 订单列表，每个订单包含id、order_number、status等信息
        total: 订单总数量
        content: AI消息内容（可选，如"找到2个订单"）

    Example:
        >>> response = OrderListResponse(
        ...     orders=[{"id": 1, "order_number": "ORD001", "status": "pending"}],
        ...     total=1,
        ...     content="找到1个订单"
        ... )
        >>> full_response = response.to_full_response()
        >>> # full_response = {
        >>> #     "response_type": "order_list",
        >>> #     "response_data": {"orders": [...], "total": 1},
        >>> #     "content": "找到1个订单",
        >>> #     "role": "assistant"
        >>> # }
    """

    response_type: Literal["order_list"] = "order_list"
    orders: List[Dict[str, Any]] = Field(description="订单列表")
    total: int = Field(description="总数量", default=0)

    def _get_response_data_fields(self) -> List[str]:
        """指定哪些字段应该放入response_data"""
        return ["orders", "total"]

    @classmethod
    def get_response_type(cls) -> str:
        return "order_list"


class ConfirmationResponse(BaseResponseModel):
    """确认响应

    用于OrderAgent的prepare_create_order、prepare_cancel_order等操作。
    当Agent需要用户确认操作时使用此响应类型。

    Fields:
        confirmation_id: 确认ID，用于后续确认操作
        action_type: 操作类型（如create_order、cancel_order）
        display_message: 展示给用户的消息
        display_data: 展示数据（如订单详情）
        content: AI消息内容（可选，通常与display_message相同）

    Example:
        >>> response = ConfirmationResponse(
        ...     confirmation_id="uuid-123",
        ...     action_type="create_order",
        ...     display_message="请确认订单信息",
        ...     display_data={"items": [...], "total": 7999},
        ...     content="请确认订单信息"
        ... )
        >>> full_response = response.to_full_response()
        >>> # full_response = {
        >>> #     "response_type": "confirmation",
        >>> #     "response_data": {
        >>> #         "confirmation_id": "uuid-123",
        >>> #         "action_type": "create_order",
        >>> #         "display_message": "请确认订单信息",
        >>> #         "display_data": {...}
        >>> #     },
        >>> #     "content": "请确认订单信息",
        >>> #     "role": "assistant"
        >>> # }
    """

    response_type: Literal["confirmation"] = "confirmation"
    confirmation_id: str = Field(description="确认ID")
    action_type: str = Field(description="操作类型")
    display_message: str = Field(description="展示给用户的消息")
    display_data: Dict[str, Any] = Field(default_factory=dict, description="展示数据")

    def _get_response_data_fields(self) -> List[str]:
        """指定哪些字段应该放入response_data"""
        return ["confirmation_id", "action_type", "display_message", "display_data"]

    @classmethod
    def get_response_type(cls) -> str:
        return "confirmation"


class ErrorResponse(BaseResponseModel):
    """错误响应

    用于所有Agent返回错误信息。

    Fields:
        error_message: 错误信息描述
        error_code: 错误代码（可选）
        content: AI消息内容（可选，通常与error_message相同）

    Example:
        >>> response = ErrorResponse(
        ...     error_message="订单创建失败",
        ...     error_code="ORDER_CREATE_FAILED",
        ...     content="订单创建失败，请重试"
        ... )
        >>> full_response = response.to_full_response()
        >>> # full_response = {
        >>> #     "response_type": "error",
        >>> #     "response_data": {"error_message": "...", "error_code": "..."},
        >>> #     "content": "订单创建失败，请重试",
        >>> #     "role": "assistant"
        >>> # }
    """

    response_type: Literal["error"] = "error"
    error_message: str = Field(description="错误信息")
    error_code: Optional[str] = Field(default=None, description="错误代码")

    def _get_response_data_fields(self) -> List[str]:
        """指定哪些字段应该放入response_data"""
        fields = ["error_message"]
        if self.error_code is not None:
            fields.append("error_code")
        return fields

    @classmethod
    def get_response_type(cls) -> str:
        return "error"
