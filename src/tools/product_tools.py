"""商品搜索工具

提供商品查询、搜索功能，支持多条件组合筛选。
"""

from typing import Annotated, Optional

from langchain_core.tools import tool
from pydantic import Field
from sqlalchemy.orm import Session

from src.db.engine import get_db_session
from src.db.crud import search_products, get_product_by_id
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
    brand: Annotated[
        Optional[str],
        Field(
            default=None,
            description="品牌名称",
            examples=["Apple", "华为", "小米"]
        )
    ] = None,
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
            description="是否仅显示有货商品（默认False，显示所有商品包括无库存的）",
            examples=[True, False]
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
            description="返回结果数量限制",
            examples=[10, 20, 50]
        )
    ] = 10,
) -> str:
    """商品搜索工具 - 支持多条件组合筛选

    用户可以用自然语言描述多个条件，LLM 会自动解析并填充参数。

    参数说明:
    - name: 商品名称或型号的模糊搜索关���词，如'iPhone'、'Mate60'
    - category: 主分类名称，如'手机'、'电脑'、'家电'
    - sub_category: 子分类名称，如'智能手机'、'5G手机'、'笔记本电脑'
    - brand: 品牌名称，如'Apple'、'华为'、'小米'
    - price_min: 最低价格（元）
    - price_max: 最高价格（元）
    - min_rating: 最低评分（1-5分）
    - in_stock_only: 是否仅显示有货商品，默认是
    - special_only: 是否仅显示特价商品，默认否
    - limit: 返回结果数量限制，默认10条

    使用场景示例:
    - "找2000元以下的手机" → price_max=2000, category='手机'
    - "华为的笔记本电脑" → brand='华为', category='电脑'
    - "评分4.5以上的有货商品" → min_rating=4.5, in_stock_only=True
    - "特价手机有哪些" → category='手机', special_only=True

    Returns:
        商品列表的格式化文本
    """
    try:
        with get_db_session() as db:
            # 构建搜索参数
            args = ProductSearchArgs(
                name=brand,
                category=category,
                sub_category=sub_category,
                brand="",
                price_min=price_min,
                price_max=price_max,
                min_rating=min_rating,
                in_stock_only=in_stock_only,
                special_only=special_only,
                limit=limit,
            )

            # 执行搜索
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

            if not products:
                # 生成未找到结果的提示
                conditions = []
                if name:
                    conditions.append(f"关键词'{name}'")
                if category:
                    conditions.append(f"分类'{category}'")
                if brand:
                    conditions.append(f"品牌'{brand}'")
                if price_max:
                    conditions.append(f"价格≤{price_max}元")

                cond_str = "、".join(conditions) if conditions else "指定条件"
                return f"🔍 未找到符合{cond_str}的商品。\n\n💡 建议：\n   - 尝试放宽筛选条件\n   - 更换关键词搜索"

            # 格式化结果
            result_lines = [f"🔍 找到 {len(products)} 件商品：\n"]

            for i, product in enumerate(products, 1):
                display = ProductDisplay.from_db(product)
                result_lines.append(f"{i}. {display.format_text()}")

            # 生成搜索条件摘要
            summary_parts = []
            if name:
                summary_parts.append(f"关键词:{name}")
            if category:
                summary_parts.append(f"分类:{category}")
            if brand:
                summary_parts.append(f"品牌:{brand}")
            if price_max:
                summary_parts.append(f"价格≤{price_max}元")

            summary = " | ".join(summary_parts) if summary_parts else "全部商品"
            result_lines.append(f"\n📊 搜索条件: {summary}")

            return "\n".join(result_lines)

    except Exception as e:
        return f"❌ 搜索商品时出错: {str(e)}"


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
        商品详细信息
    """
    try:
        with get_db_session() as db:
            product = get_product_by_id(db, product_id)

            if not product:
                return f"❌ 未找到ID为 {product_id} 的商品"

            display = ProductDisplay.from_db(product)

            # 详细信息格式化
            special_mark = " [特价商品]" if display.special else ""
            stock_info = "现货" if display.stock > 0 else "缺货"

            result = [
                f"📦 {display.name}{special_mark}",
                f"━━━━━━━━━━━━━━━━━━━━",
                f"🏷️ 品牌: {display.brand or '未知'}",
                f"📂 分类: {display.main_category or '未知'} / {display.sub_category or '未知'}",
                f"🔖 型号: {display.model_number or '未提供'}",
                f"💰 价格: ¥{display.price:.2f}" if display.price else "💰 价格: 面议",
                f"⭐ 评分: {'⭐' * int(display.rating)}{display.rating:.1f}分",
                f"📦 库存: {display.stock}件 ({stock_info})",
            ]

            if display.description:
                result.append(f"\n📝 商品描述:\n{display.description}")

            result.append(f"\n━━━━━━━━━━━━━━━━━━━━")
            result.append(f"💡 商品ID: {display.id} (用于下单和查询)")

            return "\n".join(result)

    except Exception as e:
        return f"❌ 获取商品详情时出错: {str(e)}"


@tool
def get_brands() -> str:
    """获取所有可用品牌列表

    Returns:
        品牌列表
    """
    try:
        from src.db.models import Brand
        from sqlalchemy import select

        with get_db_session() as db:
            brands = db.execute(
                select(Brand).order_by(Brand.name)
            ).scalars().all()

            if not brands:
                return "❌ 暂无品牌数据"

            brand_names = [f"   • {brand.name}" for brand in brands]
            return f"🏭 可选品牌列表 (共{len(brands)}个):\n" + "\n".join(brand_names)

    except Exception as e:
        return f"❌ 获取品牌列表时出错: {str(e)}"


@tool
def get_categories() -> str:
    """获取所有可用分类列表

    Returns:
        分类列表
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
                return "❌ 暂无分类数据"

            # 按品牌分组
            from collections import defaultdict
            brand_cats = defaultdict(list)
            for cat in categories:
                brand_cats[cat.brand.name if cat.brand else "未知"].append(cat.name)

            result = ["📂 可选分类列表:\n"]
            for brand, cats in sorted(brand_cats.items()):
                result.append(f"🏭 {brand}:")
                for cat in cats:
                    result.append(f"      • {cat}")

            return "\n".join(result)

    except Exception as e:
        return f"❌ 获取分类列表时出错: {str(e)}"


def get_product_tools() -> list:
    """获取所有商品工具"""
    return [
        search_products_tool,
        get_product_detail,
        get_brands,
        get_categories,
    ]
