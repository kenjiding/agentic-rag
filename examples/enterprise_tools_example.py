"""企业级工具管理使用示例 - 2025-2026 最佳实践

本示例展示如何使用企业级工具管理系统：
1. 创建企业级工具注册表
2. 注册工具并设置权限
3. 为不同Agent分配不同工具
4. 工具使用监控和审计
"""
import sys
from pathlib import Path
import logging
from colorama import Fore, Style, init

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from src.multi_agent import (
    MultiAgentGraph,
    ToolRegistry,
    ToolCategory,
    ToolPermission,
    ChatAgent,
    RAGAgent
)
from dotenv import load_dotenv

# 初始化colorama
init(autoreset=True)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def print_section(title: str):
    """打印分节标题"""
    print(f"\n{Style.BRIGHT}{Fore.CYAN}{'='*60}")
    print(f"{title}")
    print(f"{'='*60}{Style.RESET_ALL}\n")


# 1. 定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入一个数学表达式字符串，返回计算结果。
    
    Args:
        expression: 数学表达式，例如 "2 + 2" 或 "10 * 5"
    
    Returns:
        计算结果字符串
    """
    try:
        result = eval(expression)  # 注意：生产环境应使用更安全的方法
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。
    
    Args:
        city: 城市名称
    
    Returns:
        天气信息字符串
    """
    # 这里可以调用真实的天气API
    return f"{city}的天气：晴天，25°C"


@tool
def web_search(query: str) -> str:
    """在互联网上搜索信息。
    
    Args:
        query: 搜索查询字符串
    
    Returns:
        搜索结果字符串
    """
    # 这里可以调用真实的搜索API
    return f"搜索结果: {query}"


@tool
def get_current_time() -> str:
    """获取当前时间。
    
    Returns:
        当前时间字符串
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函数"""
    print_section("企业级工具管理示例")
    print("本示例展示企业级工具管理系统的使用方式。\n")
    
    # 2. 创建工具注册表
    print_section("步骤1: 创建工具注册表")
    tool_registry = ToolRegistry()
    print(f"{Fore.GREEN}✓ 工具注册表已创建{Style.RESET_ALL}\n")
    
    # 3. 注册工具（带完整元数据）
    print_section("步骤2: 注册工具")
    
    # 注册计算器工具（所有Agent可用）
    tool_registry.register_tool(
        name="calculator",
        tool=calculator,
        category=ToolCategory.CALCULATION,
        permission=ToolPermission.PUBLIC,  # 所有Agent可用
        tags=["math", "calculation"],
        description="执行数学计算",
        rate_limit=100,  # 每分钟100次
        cost_per_call=0.001
    )
    print(f"{Fore.GREEN}✓ 计算器工具已注册（PUBLIC权限）{Style.RESET_ALL}")
    
    # 注册天气工具（仅chat_agent可用）
    tool_registry.register_tool(
        name="get_weather",
        tool=get_weather,
        category=ToolCategory.INFORMATION,
        permission=ToolPermission.RESTRICTED,  # 需要授权
        allowed_agents=["chat_agent"],  # 只允许chat_agent使用
        tags=["weather", "information"],
        description="获取天气信息",
        rate_limit=60,
        cost_per_call=0.01
    )
    print(f"{Fore.GREEN}✓ 天气工具已注册（RESTRICTED权限，仅chat_agent可用）{Style.RESET_ALL}")
    
    # 注册网络搜索工具（仅rag_agent可用）
    tool_registry.register_tool(
        name="web_search",
        tool=web_search,
        category=ToolCategory.SEARCH,
        permission=ToolPermission.PRIVATE,  # 私有工具
        allowed_agents=["rag_agent"],  # 只允许rag_agent使用
        tags=["search", "web"],
        description="网络搜索工具",
        rate_limit=30,
        cost_per_call=0.05
    )
    print(f"{Fore.GREEN}✓ 网络搜索工具已注册（PRIVATE权限，仅rag_agent可用）{Style.RESET_ALL}")
    
    # 注册时间工具（所有Agent可用）
    tool_registry.register_tool(
        name="get_current_time",
        tool=get_current_time,
        category=ToolCategory.UTILITY,
        permission=ToolPermission.PUBLIC,
        tags=["time", "utility"],
        description="获取当前时间",
        rate_limit=200,
        cost_per_call=0.0001
    )
    print(f"{Fore.GREEN}✓ 时间工具已注册（PUBLIC权限）{Style.RESET_ALL}\n")
    
    # 4. 查看工具分配情况
    print_section("步骤3: 查看工具分配情况")
    
    print(f"{Fore.BLUE}ChatAgent可用工具:{Style.RESET_ALL}")
    chat_tools = tool_registry.get_tools_for_agent("chat_agent")
    for tool in chat_tools:
        metadata = tool_registry.get_tool_metadata(tool.name)
        print(f"  - {tool.name}: {metadata.description if metadata else 'N/A'}")
    print()
    
    print(f"{Fore.BLUE}RAGAgent可用工具:{Style.RESET_ALL}")
    rag_tools = tool_registry.get_tools_for_agent("rag_agent")
    for tool in rag_tools:
        metadata = tool_registry.get_tool_metadata(tool.name)
        print(f"  - {tool.name}: {metadata.description if metadata else 'N/A'}")
    print()
    
    # 5. 创建Agent并分配工具
    print_section("步骤4: 创建Agent")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    chat_agent = ChatAgent(
        llm=llm,
        tool_registry=tool_registry  # ChatAgent会自动获取它有权限的工具
    )
    print(f"{Fore.GREEN}✓ ChatAgent已创建{Style.RESET_ALL}")
    
    rag_agent = RAGAgent(
        llm=llm,
        persist_directory="./tmp/chroma_db/agentic_rag"
    )
    print(f"{Fore.GREEN}✓ RAGAgent已创建{Style.RESET_ALL}\n")
    
    # 6. 初始化MultiAgentGraph
    print_section("步骤5: 初始化MultiAgentGraph")
    graph = MultiAgentGraph(
        llm=llm,
        agents=[rag_agent, chat_agent],
        tool_registry=tool_registry
    )
    print(f"{Fore.GREEN}✓ MultiAgentGraph已初始化{Style.RESET_ALL}\n")
    
    # 7. 查看工具摘要
    print_section("步骤6: 工具摘要信息")
    summary = tool_registry.get_tools_summary()
    for name, info in summary.items():
        print(f"{Fore.YELLOW}{name}:{Style.RESET_ALL}")
        print(f"  描述: {info['description']}")
        print(f"  类别: {info['category']}")
        print(f"  权限: {info['permission']}")
        print(f"  启用: {info['is_enabled']}")
        print(f"  标签: {', '.join(info['tags'])}")
        print(f"  成本: ${info['cost_per_call']:.4f}/次")
        print()
    
    # 8. 演示动态权限管理
    print_section("步骤7: 动态权限管理演示")
    
    # 授予rag_agent使用天气工具的权限
    print("授予rag_agent使用天气工具的权限...")
    tool_registry.grant_permission("get_weather", "rag_agent")
    
    # 查看更新后的工具分配
    print(f"\n{Fore.BLUE}更新后RAGAgent可用工具:{Style.RESET_ALL}")
    rag_tools_updated = tool_registry.get_tools_for_agent("rag_agent")
    for tool in rag_tools_updated:
        print(f"  - {tool.name}")
    print()
    
    # 9. 查看使用统计（模拟）
    print_section("步骤8: 使用统计")
    stats = tool_registry.get_usage_stats()
    print(f"{Fore.MAGENTA}工具使用统计:{Style.RESET_ALL}")
    for tool_name, stat in stats.items():
        print(f"  {tool_name}:")
        print(f"    总调用: {stat['total_calls']}")
        print(f"    成功: {stat['successful_calls']}")
        print(f"    失败: {stat['failed_calls']}")
        print(f"    总成本: ${stat['total_cost']:.4f}")
    print()
    
    # 10. 测试查询
    print_section("步骤9: 测试查询")
    test_question = "帮我计算 123 * 456"
    print(f"{Fore.CYAN}问题: {Style.RESET_ALL}{test_question}\n")
    
    result = graph.invoke(test_question)
    
    # 打印结果
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            print(f"{Fore.GREEN}答案: {Style.RESET_ALL}")
            print(f"{last_message.content}\n")
    
    # 打印路由信息
    routing_reason = result.get("routing_reason")
    if routing_reason:
        print(f"{Fore.YELLOW}路由决策: {Style.RESET_ALL}{routing_reason}\n")
    
    print_section("示例完成")
    print("💡 企业级工具管理特性:")
    print("1. ✅ Agent级别的工具权限控制")
    print("2. ✅ 工具分类和标签管理")
    print("3. ✅ 使用审计和监控")
    print("4. ✅ 动态工具注册和权限管理")
    print("5. ✅ 成本追踪和速率限制")
    print("6. ✅ 工具健康检查支持")


if __name__ == "__main__":
    main()

