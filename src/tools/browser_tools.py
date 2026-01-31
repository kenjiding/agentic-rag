"""Browser Tools - 基于 browser-use 的 AI 浏览器代理工具

browser-use 是一个 AI Agent 框架，通过自然语言任务描述让 LLM 自动操作浏览器。

核心设计理念：
- 不手动编写浏览器操作代码，而是让 AI 自动决定如何操作
- 使用自然语言描述任务，Agent 自动完成
- 支持真实网站交互、数据提取、价格比较等

企业级最佳实践：
- 工具设计遵循单一职责原则
- 返回结构化 JSON 数据
- 完善的错误处理
- 支持异步执行
"""

import json
import logging
from typing import Annotated, List
from langchain_core.tools import tool
from pydantic import Field

# browser-use imports
try:
    from browser_use import Agent as BrowserUseAgent
    BROWSER_USE_AVAILABLE = True
    logging.info("✅ browser-use 导入成功")
except ImportError as e:
    BROWSER_USE_AVAILABLE = False
    logging.error(f"❌ browser-use 导入失败: {e}")
except Exception as e:
    BROWSER_USE_AVAILABLE = False
    logging.error(f"❌ browser-use 导入时发生异常: {e}", exc_info=True)

logger = logging.getLogger(__name__)

# 获取 LLM 配置
def _get_llm_for_browser_agent():
    """获取用于 browser-use Agent 的 LLM
    
    关键发现：browser-use 提供了自己的 LLM 包装类（from browser_use import ChatOpenAI）
    这些包装类有 provider 属性，而 langchain 的原生 LLM 没有。
    
    参考：https://github.com/browser-use/browser-use/blob/main/examples/models/gpt-4.1.py
    """
    try:
        import os
        # 使用 browser-use 提供的 ChatOpenAI 包装类（不是 langchain 的）
        from browser_use import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未设置 OPENAI_API_KEY 环境变量")
        
        # 获取系统配置的模型名称
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        llm = ChatOpenAI(model=model_name)
        logger.info(f"使用 browser-use 的 ChatOpenAI 包装类: {model_name}")
        return llm
            
    except Exception as e:
        logger.error(f"无法创建 browser-use LLM: {e}", exc_info=True)
        return None


# Google 搜索时的 site: 限定域名（用于限定搜索范围）
GOOGLE_SITE_DOMAINS = {
    "jd": "jd.com",
    "taobao": "taobao.com",
    "tmall": "tmall.com",
    "xianyu": "goofish.com OR xianyu.com OR 2.taobao.com",  # 闲鱼有多个域名
}

# 平台显示名称
PLATFORM_DISPLAY_NAMES = {
    "jd": "京东",
    "taobao": "淘宝",
    "tmall": "天猫",
    "xianyu": "闲鱼",
}


async def _google_search_product_impl(
    query: str,
    site: str = "all",
    max_results: int = 10
) -> str:
    """使用 Google 搜索商品（推荐方式）
    
    优势：
    - 不需要登录电商网站
    - 避开反爬虫和验证码
    - Google 已索引商品信息，结果稳定可靠
    - 使用 site: 操作符限定搜索范围
    
    Args:
        query: 搜索关键词
        site: 目标网站代码（"all" 表示搜索所有电商平台）
        max_results: 返回结果数量
    
    Returns:
        JSON 格式的搜索结果字符串
    """
    if not BROWSER_USE_AVAILABLE:
        return json.dumps({
            "text": "浏览器工具不可用：browser-use 未安装",
            "products": [],
            "error": "browser-use not installed"
        }, ensure_ascii=False)

    try:
        # 构建 Google 搜索 query
        if site == "all":
            # 搜索所有主流电商平台
            site_filter = "(site:jd.com OR site:taobao.com OR site:tmall.com OR site:goofish.com)"
            site_name = "多平台（京东/淘宝/天猫/闲鱼）"
        elif site in GOOGLE_SITE_DOMAINS:
            site_filter = f"site:{GOOGLE_SITE_DOMAINS[site]}"
            site_name = PLATFORM_DISPLAY_NAMES.get(site, site)
        else:
            # 不限定平台，通用 Google 搜索
            site_filter = ""
            site_name = "全网"

        # 构建完整搜索词
        google_query = f"{query} {site_filter} 价格".strip()
        google_url = f"https://www.google.com/search?q={google_query.replace(' ', '+')}"
        
        logger.info(f"Google 搜索商品: {google_query}")

        # 获取 LLM
        llm = _get_llm_for_browser_agent()
        if llm is None:
            return json.dumps({
                "text": "无法初始化 LLM",
                "products": [],
                "error": "llm initialization failed"
            }, ensure_ascii=False)

        # 创建自然语言任务描述
        task = f"""
你是一个商品搜索助手，使用 Google 搜索来查找电商平台上的商品信息。

**任务：**
1. 打开 Google 搜索页面: {google_url}
2. 等待搜索结果加载完成
3. 从搜索结果中提取商品信息（最多 {max_results} 个）

**提取规则：**
- 只提取来自电商网站（京东、淘宝、天猫、闲鱼等）的搜索结果
- 忽略广告和非商品页面
- 从搜索结果标题和描述中提取：
  - name: 商品名称
  - price: 价格（如果显示，只保留数字）
  - url: 商品链接
  - platform: 来源平台（京东/淘宝/天猫/闲鱼）
  - description: 商品描述摘要

**输出格式（JSON）：**
{{
    "products": [
        {{
            "name": "商品名称",
            "price": 价格数字或null,
            "url": "商品链接",
            "platform": "京东/淘宝/天猫/闲鱼",
            "description": "简短描述"
        }}
    ]
}}

注意：
- 如果价格不在搜索结果中显示，设为 null
- platform 根据 URL 域名判断：jd.com=京东, taobao.com=淘宝, tmall.com=天猫, goofish.com/xianyu=闲鱼
- 如果没有找到商品结果，返回空列表
"""

        # 创建 browser-use Agent
        agent = BrowserUseAgent(
            task=task,
            llm=llm,
            use_vision=True,
            max_actions_per_step=4,
            max_failures=3,
        )

        # 执行任务
        logger.info(f"Google 搜索 Agent 开始执行...")
        result = await agent.run(max_steps=15)
        
        # 提取结果
        final_result = None
        if hasattr(result, 'final_result'):
            final_result = result.final_result()
        elif hasattr(result, '__iter__'):
            result_list = list(result)
            if result_list:
                last_item = result_list[-1]
                if hasattr(last_item, 'result'):
                    final_result = last_item.result
                elif hasattr(last_item, 'model_output'):
                    final_result = last_item.model_output
                else:
                    final_result = str(last_item)
        else:
            final_result = str(result)

        logger.info(f"Google 搜索完成，结果类型: {type(final_result)}")

        # 解析 JSON 结果
        products = []
        try:
            if isinstance(final_result, str):
                import re
                json_match = re.search(r'\{.*"products".*\}', final_result, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    products = parsed.get("products", [])
                else:
                    parsed = json.loads(final_result)
                    products = parsed.get("products", [])
            elif isinstance(final_result, dict):
                products = final_result.get("products", [])
                
            logger.info(f"Google 搜索找到 {len(products)} 个商品")
            
        except json.JSONDecodeError as e:
            logger.warning(f"Google 搜索结果 JSON 解析失败: {e}")
            return json.dumps({
                "text": f"通过 Google 搜索 \"{query}\" 的结果：\n{str(final_result)[:500]}",
                "products": [],
                "total": 0,
                "query_summary": f"Google搜索'{query}'",
                "platform": site_name,
                "raw_result": str(final_result)
            }, ensure_ascii=False)

        # 构建响应
        text_lines = [f"通过 Google 在 {site_name} 搜索 \"{query}\" 找到 {len(products)} 个商品：\n"]
        for i, product in enumerate(products[:max_results], 1):
            name = product.get("name", "未知商品")
            price = product.get("price")
            platform = product.get("platform", "")
            price_str = f"¥{price}" if price else "价格待查"
            text_lines.append(f"{i}. [{platform}] {name} - {price_str}")

        response = {
            "text": "\n".join(text_lines),
            "products": products[:max_results],
            "total": len(products),
            "query_summary": f"Google搜索'{query}'（{site_name}）",
            "platform": site_name,
            "search_method": "google"
        }

        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Google 搜索失败: {e}", exc_info=True)
        return json.dumps({
            "text": f"Google 搜索商品时出错: {str(e)}",
            "products": [],
            "error": str(e)
        }, ensure_ascii=False)


@tool
async def browser_search_product(
    query: Annotated[
        str,
        Field(
            description="商品搜索关键词（支持品牌、型号、产品名称）",
            examples=["iPhone 15 Pro", "小米电视", "戴森吸尘器"]
        )
    ],
    platform: Annotated[
        str,
        Field(
            default="all",
            description="目标电商平台：all=搜索所有平台, jd=京东, taobao=淘宝, tmall=天猫, xianyu=闲鱼",
            examples=["all", "jd", "taobao", "tmall", "xianyu"]
        )
    ] = "all",
    max_results: Annotated[
        int,
        Field(
            default=10,
            description="返回结果数量限制",
            ge=1,
            le=20
        )
    ] = 10
) -> str:
    """通过 Google 搜索电商平台商品

    使用 Google 搜索 + site: 操作符来搜索电商平台商品，优势：
    - 不需要登录电商网站账号
    - 避免验证码和反爬虫机制
    - 可同时搜索多个平台
    - 结果稳定可靠

    Args:
        query: 搜索关键词
        platform: 目标平台（"all" 搜索京东/淘宝/天猫/闲鱼）
        max_results: 返回结果数量

    Returns:
        JSON 格式的搜索结果
    """
    return await _google_search_product_impl(query, platform, max_results)


@tool
async def browser_compare_prices(
    query: Annotated[
        str,
        Field(
            description="商品搜索关键词",
            examples=["iPhone 15 Pro 256GB", "小米电视 65英寸"]
        )
    ],
    max_results: Annotated[
        int,
        Field(
            default=10,
            description="返回的总结果数量",
            ge=1,
            le=20
        )
    ] = 10
) -> str:
    """跨多个电商平台比价（通过 Google 搜索一次性聚合）

    优化策略：使用一次 Google 搜索同时查询多个电商平台，比逐个平台搜索更高效。
    
    搜索范围：京东、淘宝、天猫、闲鱼

    Args:
        query: 搜索关键词
        max_results: 返回结果数量

    Returns:
        JSON 格式的比价结果：
        {
            "text": "人类可读的比价摘要",
            "comparison": {
                "query": "搜索关键词",
                "all_products": [...],
                "by_platform": {
                    "京东": [...],
                    "淘宝": [...],
                    ...
                },
                "best_deal": {...},
                "price_range": {...}
            }
        }
    """
    if not BROWSER_USE_AVAILABLE:
        return json.dumps({
            "text": "浏览器工具不可用：browser-use 未安装",
            "comparison": {},
            "error": "browser-use not installed"
        }, ensure_ascii=False)

    try:
        logger.info(f"开始跨平台比价（Google 聚合搜索）: {query}")

        # 使用 Google 搜索所有平台（一次搜索，效率更高）
        result = await _google_search_product_impl(
            query=query,
            site="all",
            max_results=max_results
        )
        
        # 解析结果
        result_data = json.loads(result)
        products = result_data.get("products", [])
        
        # 按平台分组
        by_platform = {}
        all_prices = []
        
        for product in products:
            platform = product.get("platform", "未知")
            if platform not in by_platform:
                by_platform[platform] = []
            by_platform[platform].append(product)
            
            price = product.get("price")
            if isinstance(price, (int, float)):
                all_prices.append(price)

        # 构建比价结果
        comparison = {
            "query": query,
            "all_products": products,
            "by_platform": by_platform,
            "price_range": {"min": 0, "max": 0, "avg": 0}
        }

        # 计算价格统计
        if all_prices:
            comparison["price_range"] = {
                "min": min(all_prices),
                "max": max(all_prices),
                "avg": sum(all_prices) / len(all_prices)
            }

            # 找到最佳价格
            products_with_price = [p for p in products if isinstance(p.get("price"), (int, float))]
            if products_with_price:
                best_product = min(products_with_price, key=lambda p: p.get("price", float('inf')))
                comparison["best_deal"] = {
                    "product": best_product,
                    "platform": best_product.get("platform", "未知"),
                    "price": best_product.get("price", 0)
                }

        # 生成人类可读摘要
        text_lines = [f"'{query}' 跨平台比价结果（通过 Google 搜索）：\n"]
        
        for platform, platform_products in by_platform.items():
            prices = [p.get("price") for p in platform_products if isinstance(p.get("price"), (int, float))]
            if prices:
                text_lines.append(f"✅ {platform}: 找到 {len(platform_products)} 个商品，最低价 ¥{min(prices):.2f}")
            else:
                text_lines.append(f"✅ {platform}: 找到 {len(platform_products)} 个商品")

        if "best_deal" in comparison:
            best = comparison["best_deal"]
            text_lines.append(f"\n🏆 最佳价格: ¥{best['price']:.2f} ({best['platform']})")

        if all_prices:
            price_range = comparison["price_range"]
            text_lines.append(f"📊 价格区间: ¥{price_range['min']:.2f} - ¥{price_range['max']:.2f}")
            text_lines.append(f"📈 平均价格: ¥{price_range['avg']:.2f}")

        response = {
            "text": "\n".join(text_lines),
            "comparison": comparison
        }

        logger.info(f"比价完成: {len(products)} 个商品，来自 {len(by_platform)} 个平台")
        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        logger.error(f"比价失败: {e}", exc_info=True)
        return json.dumps({
            "text": f"商品比价时出错: {str(e)}",
            "comparison": {},
            "error": str(e)
        }, ensure_ascii=False)


async def _browser_extract_product_info_impl(url: str) -> str:
    """商品信息提取的内部实现函数
    
    使用 browser-use Agent 通过自然语言任务提取商品详情。
    
    Args:
        url: 商品详情页URL
    
    Returns:
        JSON 格式的商品详细信息字符串
    """
    if not BROWSER_USE_AVAILABLE:
        return json.dumps({
            "text": "浏览器工具不可用：browser-use 未安装",
            "product": {},
            "error": "browser-use not installed"
        }, ensure_ascii=False)

    try:
        logger.info(f"开始提取商品信息: {url}")

        # 获取 LLM
        llm = _get_llm_for_browser_agent()
        if llm is None:
            return json.dumps({
                "text": "无法初始化 LLM",
                "product": {},
                "error": "llm initialization failed"
            }, ensure_ascii=False)

        # 创建自然语言任务描述
        task = f"""
访问商品详情页 {url} 并提取完整的商品信息。

请按以下步骤操作：
1. 打开URL: {url}
2. 等待页面完全加载完成（包括动态价格、库存等信息）
3. 提取以下所有可见信息：
   - 商品标题 (name)
   - 当前价格 (price) - 只保留数字
   - 原价 (original_price) - 如果有促销
   - 库存状态 (stock_status)
   - 商品评分 (rating)
   - 评价数量 (review_count)
   - 商品规格参数 (specs) - 如颜色、容量、尺寸等
   - 商家名称 (shop)
   - 促销活动 (promotion) - 如果有
   - 商品图片列表 (images)
   - 商品描述 (description)

请以 JSON 格式返回结果，格式如下：
{{
    "name": "商品标题",
    "price": 价格数字,
    "original_price": 原价数字,
    "stock_status": "库存状态",
    "rating": 评分数字,
    "review_count": 评价数量,
    "specs": {{"规格名": "规格值"}},
    "shop": "店铺名称",
    "promotion": "促销信息",
    "images": ["图片URL1", "图片URL2"],
    "description": "商品描述"
}}

如果某些字段不存在，可以省略或设为 null。
"""

        # 创建 browser-use Agent
        agent = BrowserUseAgent(
            task=task,
            llm=llm,
            use_vision=True,  # 启用视觉能力
            max_actions_per_step=4,
            max_failures=3,
        )

        # 执行任务
        logger.info(f"Agent 开始提取商品信息...")
        result = await agent.run(max_steps=15)
        
        # 获取最终结果
        final_result = result.final_result() if hasattr(result, 'final_result') else str(result)
        logger.info(f"Agent 执行完成")

        # 解析结果
        product_info = {}
        try:
            if isinstance(final_result, str):
                product_info = json.loads(final_result)
            elif isinstance(final_result, dict):
                product_info = final_result
        except json.JSONDecodeError:
            logger.warning(f"无法解析为 JSON: {final_result}")
            product_info = {"raw_result": str(final_result)}

        # 生成人类可读摘要
        name = product_info.get("name", "未知商品")
        price = product_info.get("price", "价格未知")
        stock = product_info.get("stock_status", "未知")

        text = f"📦 {name}\n"
        text += f"💰 价格: ¥{price}\n"
        text += f"📦 库存: {stock}\n"

        if "rating" in product_info:
            text += f"⭐ 评分: {product_info['rating']}\n"
        if "review_count" in product_info:
            text += f"💬 评价数: {product_info['review_count']}\n"
        if "shop" in product_info:
            text += f"🏪 店铺: {product_info['shop']}\n"

        response = {
            "text": text,
            "product": product_info
        }

        logger.info(f"信息提取完成: {name}")
        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        logger.error(f"提取商品信息失败: {e}", exc_info=True)
        return json.dumps({
            "text": f"提取商品信息时出错: {str(e)}",
            "product": {},
            "error": str(e)
        }, ensure_ascii=False)


@tool
async def browser_extract_product_info(
    url: Annotated[
        str,
        Field(
            description="商品详情页URL",
            examples=["https://item.jd.com/100012345678.html"]
        )
    ]
) -> str:
    """从商品详情页提取完整信息（使用浏览器自动化）

    此工具打开商品详情页，提取所有可见信息：
    - 基本信息（名称、价格、库存）
    - 规格参数
    - 用户评价统计
    - 商家信息
    - 促销活动

    Args:
        url: 商品详情页URL

    Returns:
        JSON 格式的商品详细信息
    """
    return await _browser_extract_product_info_impl(url)


def get_browser_tools() -> List:
    """获取所有浏览器工具"""
    logger.info(f"[get_browser_tools] BROWSER_USE_AVAILABLE = {BROWSER_USE_AVAILABLE}")
    
    if not BROWSER_USE_AVAILABLE:
        logger.error("❌ browser-use 不可用，返回空工具列表")
        return []
    
    tools = [
        browser_search_product,
        browser_compare_prices,
        browser_extract_product_info,
    ]
    logger.info(f"✅ 返回 {len(tools)} 个浏览器工具: {[t.name for t in tools]}")
    return tools


def is_browser_available() -> bool:
    """检查浏览器工具是否可用"""
    return BROWSER_USE_AVAILABLE
