"""测试数据加载器

当USE_TEST_DATA=True时，从JSON文件加载测试数据，而不是从数据库查询。
这样可以在没有数据库的情况下测试功能。
"""

import json
import os
import re
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


def _normalize_string_for_search(text: str) -> str:
    """标准化字符串用于搜索比较（企业级最佳实践）
    
    从源头解决问题：在搜索时对关键词和数据库字段都进行标准化处理，
    而不是生成多个变体。这样可以处理空格、大小写等格式差异。
    
    标准化规则：
    1. 去除所有空格（处理"华为Mate 60 Pro" vs "华为 Mate 60 Pro"）
    2. 转换为小写（处理大小写差异）
    3. 去除首尾空白
    
    Args:
        text: 原始文本
        
    Returns:
        标准化后的文本
    """
    if not text:
        return ""
    # 去除所有空格并转换为小写
    return text.replace(' ', '').replace('\t', '').replace('\n', '').lower().strip()


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
        # 名称筛选（企业级最佳实践：标准化比较，从源头解决问题）
        if name:
            # 标准化搜索关键词和产品字段，去除空格后比较
            normalized_name = _normalize_string_for_search(name)
            product_name_normalized = _normalize_string_for_search(product.get("name", ""))
            product_model_normalized = _normalize_string_for_search(product.get("model_number", "") or "")
            product_description_normalized = _normalize_string_for_search(product.get("description", "") or "")
            
            # 检查标准化后的关键词是否在标准化后的产品字段中
            if (normalized_name not in product_name_normalized and
                (not product_model_normalized or normalized_name not in product_model_normalized) and
                (not product_description_normalized or normalized_name not in product_description_normalized)):
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
