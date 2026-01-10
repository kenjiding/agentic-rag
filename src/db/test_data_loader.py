"""测试数据加载器

当USE_TEST_DATA=True时，从JSON文件加载测试数据，而不是从数据库查询。
这样可以在没有数据库的情况下测试功能。
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from decimal import Decimal
from dotenv import load_dotenv

# 确保环境变量已加载（可能在模块被导入时还未加载）
load_dotenv()

logger = logging.getLogger(__name__)

# 测试数据缓存
_test_data_cache: Optional[Dict[str, Any]] = None

TEST_DATA_PATH = Path(__file__).parent.parent.parent / "tmp" / "test_products.json"


def load_test_data() -> Dict[str, Any]:
    """加载测试数据JSON文件"""
    global _test_data_cache
    
    if _test_data_cache is not None:
        return _test_data_cache
    
    try:
        if not TEST_DATA_PATH.exists():
            logger.warning(f"测试数据文件不存在: {TEST_DATA_PATH}")
            return {"products": []}
        
        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            _test_data_cache = json.load(f)
            logger.info(f"已加载测试数据: {len(_test_data_cache.get('products', []))} 个产品")
            return _test_data_cache
    except Exception as e:
        logger.error(f"加载测试数据失败: {e}", exc_info=True)
        return {"products": []}


def clear_test_data_cache():
    """清除测试数据缓存（用于重新加载）"""
    global _test_data_cache
    _test_data_cache = None


def get_test_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
    """从测试数据中根据ID获取产品"""
    data = load_test_data()
    products = data.get("products", [])
    
    for product in products:
        if product.get("id") == product_id:
            return product
    
    return None


def search_test_products(
    name: Optional[str] = None,
    category: Optional[str] = None,
    sub_category: Optional[str] = None,
    brand: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    min_rating: Optional[float] = None,
    in_stock_only: bool = False,
    special_only: bool = False,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """从测试数据中搜索产品"""
    data = load_test_data()
    products = data.get("products", [])
    
    # 应用筛选条件
    filtered_products = []
    
    for product in products:
        # 名称筛选
        if name:
            if name.lower() not in product.get("name", "").lower():
                continue
        
        # 分类筛选
        if category:
            if product.get("main_category") != category:
                continue
        
        if sub_category:
            if product.get("sub_category") != sub_category:
                continue
        
        # 品牌筛选
        if brand:
            if product.get("brand") != brand:
                continue
        
        # 价格筛选
        price = product.get("price", 0)
        if price_min is not None and price < price_min:
            continue
        if price_max is not None and price > price_max:
            continue
        
        # 评分筛选
        rating = product.get("rating", 0)
        if min_rating is not None and rating < min_rating:
            continue
        
        # 库存筛选
        if in_stock_only:
            if product.get("stock", 0) <= 0:
                continue
        
        # 特价筛选
        if special_only:
            if not product.get("special", False):
                continue
        
        filtered_products.append(product)
    
    # 限制数量
    return filtered_products[:limit]


def is_use_test_data() -> bool:
    """检查是否使用测试数据
    
    注意：此函数在运行时动态检查环境变量，确保环境变量已正确加载。
    """
    return os.getenv("USE_TEST_DATA", "False").lower() == "true"
