"""业务工具模块 - LangChain Tool 集合

提供电商场��的工具调用能力：
- product_tools: 商品搜索、详情查询
- order_tools: 订单查询、取消、创建
"""

from .product_tools import get_product_tools
from .order_tools import get_order_tools

__all__ = ["get_product_tools", "get_order_tools"]


def load_all_tools() -> list:
    """加载所有业务工具"""
    tools = []
    tools.extend(get_product_tools())
    tools.extend(get_order_tools())
    return tools
