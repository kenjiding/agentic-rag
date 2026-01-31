"""Browser Agent 示例 - 浏览器自动化搜索与比价

演示如何使用 BrowserAgent 在真实电商网站进行商品搜索和比价。

使用场景：
1. 在真实网站搜索商品（处理 JavaScript 渲染）
2. 跨平台比价（同时在京东、淘宝、天猫搜索）
3. 获取实时价格和库存信息
4. 提取商品详细信息

企业级最佳实践：
- 使用 MultiAgentGraph 统一接口
- BrowserAgent 自动路由到合适的工具
- 结构化数据返回（JSON）
- 完善的错误处理
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.multi_agent.graph import MultiAgentGraph
from src.tools.browser_tools import is_browser_available

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_browser_search():
    """演示：在单个网站搜索商品"""
    print("\n" + "="*80)
    print("演示 1: 在京东搜索 iPhone 15 Pro")
    print("="*80 + "\n")

    # 初始化多 Agent 系统
    graph = MultiAgentGraph(
        enable_business_agents=True,
        init_web_search=False  # 不需要 web search
    )

    # 执行搜索
    query = "帮我在京东搜索 iPhone 15 Pro，要256GB的"
    result = await graph.ainvoke(query)

    # 显示结果
    print("\n--- 最终回复 ---")
    if result.messages:
        print(result.messages[-1].content)

    # 显示结构化数据
    if result.response_data:
        print("\n--- 结构化数据 ---")
        products = result.response_data.get("products", [])
        print(f"找到 {len(products)} 个商品")
        for i, product in enumerate(products[:3], 1):
            print(f"{i}. {product.get('name', '未知')} - ¥{product.get('price', 'N/A')}")


async def demo_cross_platform_comparison():
    """演示：跨平台比价"""
    print("\n" + "="*80)
    print("演示 2: 在京东和淘宝比价 iPhone 15 Pro")
    print("="*80 + "\n")

    graph = MultiAgentGraph(
        enable_business_agents=True,
        init_web_search=False
    )

    # 执行比价
    query = "帮我在京东和淘宝比价 iPhone 15 Pro 256GB，看看哪里便宜"
    result = await graph.ainvoke(query)

    # 显示结果
    print("\n--- 最终回复 ---")
    if result.messages:
        print(result.messages[-1].content)

    # 显示比价结果
    if result.response_data and "comparison" in result.response_data:
        print("\n--- 比价结果 ---")
        comparison = result.response_data["comparison"]
        
        # 显示各平台统计
        for site_name, site_data in comparison.get("sites", {}).items():
            if "error" in site_data:
                print(f"❌ {site_name}: 搜索失败")
            else:
                min_price = site_data.get("min_price", 0)
                count = site_data.get("product_count", 0)
                print(f"✅ {site_name}: {count} 个商品，最低价 ¥{min_price:.2f}")
        
        # 显示最佳价格
        if "best_deal" in comparison:
            best = comparison["best_deal"]
            print(f"\n🏆 最佳价格: ¥{best['price']:.2f} ({best['platform']})")
            print(f"商品: {best['product'].get('name', '未知')}")


async def demo_product_detail_extraction():
    """演示：提取商品详情"""
    print("\n" + "="*80)
    print("演示 3: 提取商品详情页信息")
    print("="*80 + "\n")

    graph = MultiAgentGraph(
        enable_business_agents=True,
        init_web_search=False
    )

    # 假设我们已经有了商品链接（实际使用中从搜索结果获取）
    query = "帮我提取这个商品的详细信息：https://item.jd.com/100012345678.html"
    result = await graph.ainvoke(query)

    # 显示结果
    print("\n--- 最终回复 ---")
    if result.messages:
        print(result.messages[-1].content)

    # 显示详细信息
    if result.response_data and "product" in result.response_data:
        print("\n--- 商品详情 ---")
        product = result.response_data["product"]
        print(f"名称: {product.get('name', '未知')}")
        print(f"价格: ¥{product.get('price', 'N/A')}")
        print(f"库存: {product.get('stock_status', '未知')}")
        print(f"评分: {product.get('rating', 'N/A')}")
        print(f"评价数: {product.get('review_count', 'N/A')}")


async def demo_intelligent_routing():
    """演示：智能路由（Supervisor 自动选择 BrowserAgent）"""
    print("\n" + "="*80)
    print("演示 4: 智能路由 - Supervisor 自动选择 BrowserAgent")
    print("="*80 + "\n")

    graph = MultiAgentGraph(
        enable_business_agents=True,
        init_web_search=False
    )

    # 使用自然语言查询，让 Supervisor 自动路由
    queries = [
        "帮我看看现在京东上iPhone 15 Pro卖多少钱",
        "我想买个电视，帮我在京东和淘宝比比价",
        "帮我找找小米电视，要65英寸的，看看哪里便宜"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n--- 查询 {i}: {query} ---")
        result = await graph.ainvoke(query)
        
        print(f"路由到的 Agent: {result.current_agent}")
        if result.messages:
            print(f"回复: {result.messages[-1].content[:200]}...")  # 只显示前200字符


async def main():
    """主函数"""
    print("\n" + "="*80)
    print("Browser Agent 演示")
    print("="*80)
    
    # 检查 browser-use 是否可用
    if not is_browser_available():
        print("\n⚠️  警告: browser-use 未安装或不可用")
        print("请安装: pip install browser-use")
        print("\n系统将使用模拟模式演示（不会真正执行浏览器操作）\n")
    else:
        print("\n✅ browser-use 已就绪\n")

    # 运行演示
    try:
        # 演示 1: 单网站搜索
        await demo_browser_search()
        
        # 演示 2: 跨平台比价
        await demo_cross_platform_comparison()
        
        # 演示 3: 提取详情
        # await demo_product_detail_extraction()
        
        # 演示 4: 智能路由
        await demo_intelligent_routing()
        
    except Exception as e:
        logger.error(f"演示执行失败: {e}", exc_info=True)
        print(f"\n❌ 演示执行失败: {e}")

    print("\n" + "="*80)
    print("演示完成")
    print("="*80 + "\n")


if __name__ == "__main__":
    # 设置环境变量（如果需要）
    # os.environ["USE_TEST_DATA"] = "true"  # 使用测试数据
    
    # 运行演示
    asyncio.run(main())
