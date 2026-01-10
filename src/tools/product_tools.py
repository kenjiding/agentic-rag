"""商品搜索工具

提供商品查询、搜索功能，支持多条件组合筛选。
返回 JSON 格式，包含人类可读文本和结构化产品数据。

当USE_TEST_DATA=True时，从JSON文件读取测试数据。
"""

import json
import os
from typing import Annotated, Optional

from langchain_core.tools import tool
from pydantic import Field
from sqlalchemy.orm import Session

from src.db.engine import get_db_session
from src.db.crud import search_products, get_product_by_id
from src.db.test_data_loader import is_use_test_data
from src.schema.business_models import ProductSearchArgs, ProductDisplay, ProductListResult


@tool
def search_products_tool(
    name: Annotated[
        Optional[str],
        Field(
            default=None,
            description="商品名称或型号的模糊搜索关键词",
            examples=["iPhone", "Mate60", "西门子", "Siemens"]
        )
    ] = None,
    category: Annotated[
        Optional[str],
        Field(
            default=None,
            description="主分类名称",
            examples=["手机", "电脑", "家电"]
        )
    ] = None,
    sub_category: Annotated[
        Optional[str],
        Field(
            default=None,
            description="子分类名称",
            examples=["智能手机", "5G手机", "笔记本电脑"]
        )
    ] = None,
    # brand: Annotated[
    #     Optional[str],
    #     Field(
    #         default=None,
    #         description="品牌名称",
    #         examples=["品牌: Apple", "品牌: 华为", "品牌: 小米"]
    #     )
    # ] = None,
    price_min: Annotated[
        Optional[float],
        Field(
            default=None,
            description="最低价格（元）",
            examples=[1000.0, 2000.0]
        )
    ] = None,
    price_max: Annotated[
        Optional[float],
        Field(
            default=None,
            description="最高价格（元）",
            examples=[5000.0, 10000.0]
        )
    ] = None,
    min_rating: Annotated[
        Optional[float],
        Field(
            default=None,
            description="最低评分（1-5分）",
            examples=[4.0, 4.5]
        )
    ] = None,
    in_stock_only: Annotated[
        bool,
        Field(
            default=False,
            description="是否仅显示有货商品（默认False，显示所有商品包括无库存的）。**重要规则**：只有当用户明确表达购买意图（如'购买'、'下单'、'订购'）时才设置为True；如果用户只是搜索、查看、了解商品（如'帮我搜一些产品'、'找找看'），必须设置为False（保持默认值），以便显示所有相关商品让用户了解完整信息。",
            examples=[False, True]
        )
    ] = False,
    special_only: Annotated[
        bool,
        Field(
            default=False,
            description="是否仅显示特价商品",
            examples=[True, False]
        )
    ] = False,
    limit: Annotated[
        int,
        Field(
            default=10,
            description="返回结果数量限制, 这不是商品购买数量, 而是商品搜索结果数量限制",
            examples=[10, 20, 50]
        )
    ] = 10,
) -> str:
    """商品搜索工具 - 支持多条件组合筛选

    返回 JSON 格式，包含人类可读文本和结构化产品数据。
    """
    try:
        # 构建搜索参数（注意：name参数用于商品名搜索，brand是品牌）
        args = ProductSearchArgs(
            name=name,      # 商品名称关键词
            # brand=brand,  # 品牌名称
            category=category,
            sub_category=sub_category,
            price_min=price_min,
            price_max=price_max,
            min_rating=min_rating,
            in_stock_only=in_stock_only,
            special_only=special_only,
            limit=limit,
        )

        # 运行时动态检查是否使用测试数据
        if is_use_test_data():
            db = None  # 测试数据模式不需要数据库会话
            products = search_products(
                db,
                name=args.name,
                category=args.category,
                sub_category=args.sub_category,
                brand=args.brand,
                price_min=args.price_min,
                price_max=args.price_max,
                min_rating=args.min_rating,
                in_stock_only=args.in_stock_only,
                special_only=args.special_only,
                limit=args.limit,
            )
        else:
            # 真实数据模式：使用数据库会话
            with get_db_session() as db:
                products = search_products(
                    db,
                    name=args.name,
                    category=args.category,
                    sub_category=args.sub_category,
                    brand=args.brand,
                    price_min=args.price_min,
                    price_max=args.price_max,
                    min_rating=args.min_rating,
                    in_stock_only=args.in_stock_only,
                    special_only=args.special_only,
                    limit=args.limit,
                )

        # 构建结构化产品数据（兼容测试数据和真实数据）
        products_data = []
        for product in products:
            # 安全访问属性（兼容SimpleNamespace和Product对象）
            brand_name = getattr(getattr(product, 'brand', None), 'name', None) if hasattr(product, 'brand') and product.brand else None
            main_cat_name = getattr(getattr(product, 'main_category', None), 'name', None) if hasattr(product, 'main_category') and product.main_category else None
            sub_cat_name = getattr(getattr(product, 'sub_category', None), 'name', None) if hasattr(product, 'sub_category') and product.sub_category else None
            
            products_data.append({
                "id": getattr(product, 'id', 0),
                "name": getattr(product, 'name', '未知产品'),
                "model_number": getattr(product, 'model_number', None),
                "brand": brand_name,
                "main_category": main_cat_name,
                "sub_category": sub_cat_name,
                "price": float(getattr(product, 'price', 0)) if getattr(product, 'price', None) else None,
                "stock": getattr(product, 'stock', 0),
                "rating": float(getattr(product, 'rating', 0)) if hasattr(product, 'rating') else 0.0,
                "special": getattr(product, 'special', False),
                "description": getattr(product, 'description', None),
                "images": getattr(product, 'images', []) if getattr(product, 'images', None) else [],
            })

        # 生成人类可读的文本（兼容测试数据和真实数据）
        if not products:
            conditions = []
            if name:
                conditions.append(f"关键词'{name}'")
            if category:
                conditions.append(f"分类'{category}'")
            if price_max:
                conditions.append(f"价格≤{price_max}元")

            cond_str = "、".join(conditions) if conditions else "指定条件"
            text = f"未找到符合{cond_str}的商品。建议尝试放宽筛选条件或更换关键词。"
        else:
            result_lines = [f"找到 {len(products)} 件商品：\n"]
            for i, product in enumerate(products, 1):
                try:
                    display = ProductDisplay.from_db(product)
                    result_lines.append(f"{i}. {display.format_text()}")
                except Exception:
                    # 如果from_db失败，手动构建
                    product_name = getattr(product, 'name', '未知产品')
                    product_price = float(getattr(product, 'price', 0)) if getattr(product, 'price', None) else None
                    price_str = f"¥{product_price:.2f}" if product_price else "价格面议"
                    result_lines.append(f"{i}. {product_name} - {price_str}")
            text = "\n".join(result_lines)

        # 返回 JSON 格式：包含文本和结构化数据
        result = {
            "text": text,
            "products": products_data
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        error_result = {
            "text": f"搜索商品时出错: {str(e)}",
            "products": []
        }
        return json.dumps(error_result, ensure_ascii=False)


@tool
def get_product_detail(
    product_id: Annotated[
        int,
        Field(
            description="商品ID",
            examples=[1, 2, 100]
        )
    ]
) -> str:
    """获取商品详细信息

    参数说明:
    - product_id: 商品ID

    Returns:
        商品详细信息（JSON格式，包含text和product）
    """
    try:
        # 运行时动态检查是否使用测试数据
        if is_use_test_data():
            db = None
            product = get_product_by_id(db, product_id)
        else:
            with get_db_session() as db:
                product = get_product_by_id(db, product_id)

        if not product:
            return json.dumps({
                "text": f"未找到ID为 {product_id} 的商品",
                "product": None
            }, ensure_ascii=False)

        # 构建产品数据（兼容测试数据和真实数据）
        brand_name = getattr(getattr(product, 'brand', None), 'name', None) if hasattr(product, 'brand') and product.brand else None
        main_cat_name = getattr(getattr(product, 'main_category', None), 'name', None) if hasattr(product, 'main_category') and product.main_category else None
        sub_cat_name = getattr(getattr(product, 'sub_category', None), 'name', None) if hasattr(product, 'sub_category') and product.sub_category else None
        
        product_data = {
            "id": getattr(product, 'id', product_id),
            "name": getattr(product, 'name', '未知产品'),
            "model_number": getattr(product, 'model_number', None),
            "brand": brand_name,
            "main_category": main_cat_name,
            "sub_category": sub_cat_name,
            "price": float(getattr(product, 'price', 0)) if getattr(product, 'price', None) else None,
            "stock": getattr(product, 'stock', 0),
            "rating": float(getattr(product, 'rating', 0)) if hasattr(product, 'rating') else 0.0,
            "special": getattr(product, 'special', False),
            "description": getattr(product, 'description', None),
            "images": getattr(product, 'images', []) if getattr(product, 'images', None) else [],
        }

        # 生成人类可读文本（使用ProductDisplay.from_db需要兼容SimpleNamespace）
        try:
            display = ProductDisplay.from_db(product)
        except Exception:
            # 如果from_db失败，手动构建
            from src.schema.business_models import ProductDisplay
            display = ProductDisplay(
                id=product_data["id"],
                name=product_data["name"],
                model_number=product_data["model_number"],
                brand=product_data["brand"],
                main_category=product_data["main_category"],
                sub_category=product_data["sub_category"],
                price=product_data["price"],
                stock=product_data["stock"],
                rating=product_data["rating"],
                special=product_data["special"],
                description=product_data["description"],
                images=product_data["images"]
            )
        
        special_mark = " [特价商品]" if display.special else ""
        stock_info = "现货" if display.stock > 0 else "缺货"

        text_parts = [
            f"📦 {display.name}{special_mark}",
            f"品牌: {display.brand or '未知'}",
            f"分类: {display.main_category or '未知'} / {display.sub_category or '未知'}",
            f"型号: {display.model_number or '未提供'}",
            f"价格: ¥{display.price:.2f}" if display.price else "价格: 面议",
            f"评分: {display.rating:.1f}分",
            f"库存: {display.stock}件 ({stock_info})",
        ]
        if display.description:
            text_parts.append(f"描述: {display.description}")
        text = "\n".join(text_parts)

        return json.dumps({
            "text": text,
            "product": product_data
        }, ensure_ascii=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({
            "text": f"获取商品详情时出错: {str(e)}",
            "product": None
        }, ensure_ascii=False)


@tool
def get_brands() -> str:
    """获取所有可用品牌列表

    Returns:
        品牌列表（JSON格式）
    """
    try:
        from src.db.models import Brand
        from sqlalchemy import select

        with get_db_session() as db:
            brands = db.execute(
                select(Brand).order_by(Brand.name)
            ).scalars().all()

            if not brands:
                return json.dumps({
                    "text": "暂无品牌数据",
                    "brands": []
                }, ensure_ascii=False)

            brand_list = [{"name": brand.name} for brand in brands]
            text = f"可选品牌列表 (共{len(brands)}个):\n" + "\n".join([f"  • {b.name}" for b in brands])

            return json.dumps({
                "text": text,
                "brands": brand_list
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "text": f"获取品牌列表时出错: {str(e)}",
            "brands": []
        }, ensure_ascii=False)


@tool
def get_categories() -> str:
    """获取所有可用分类列表

    Returns:
        分类列表（JSON格式）
    """
    try:
        from src.db.models import MainCategory, Brand
        from sqlalchemy import select

        with get_db_session() as db:
            categories = db.execute(
                select(MainCategory)
                .join(Brand)
                .order_by(Brand.name, MainCategory.name)
            ).scalars().all()

            if not categories:
                return json.dumps({
                    "text": "暂无分类数据",
                    "categories": []
                }, ensure_ascii=False)

            # 按品牌分组
            from collections import defaultdict
            brand_cats = defaultdict(list)
            for cat in categories:
                brand_cats[cat.brand.name if cat.brand else "未知"].append(cat.name)

            categories_list = [
                {"brand": brand, "categories": cats}
                for brand, cats in sorted(brand_cats.items())
            ]

            text_lines = ["可选分类列表:"]
            for brand, cats in sorted(brand_cats.items()):
                text_lines.append(f"{brand}:")
                text_lines.extend([f"  • {cat}" for cat in cats])

            return json.dumps({
                "text": "\n".join(text_lines),
                "categories": categories_list
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "text": f"获取分类列表时出错: {str(e)}",
            "categories": []
        }, ensure_ascii=False)


def get_product_tools() -> list:
    """获取所有商品工具"""
    return [
        search_products_tool,
        get_product_detail,
        get_brands,
        get_categories,
    ]
