"""多Agent系统使用示例 - 2025-2026 企业级最佳实践

本示例展示如何使用多Agent系统框架。

功能演示：
1. 初始化多Agent系统
2. 执行查询（自动路由到合适的Agent）
3. 查看执行结果和统计信息
"""
import sys
from pathlib import Path
import logging
from colorama import Fore, Style, init

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.multi_agent import MultiAgentGraph
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


def print_section(title: str):
    """打印分节标题"""
    print(f"\n{Style.BRIGHT}{Fore.CYAN}{'='*60}")
    print(f"{title}")
    print(f"{'='*60}{Style.RESET_ALL}\n")


def print_result(result, question: str):
    """打印执行结果"""
    print_section("执行结果")

    # 打印意图识别结果
    query_intent = result.get("query_intent")
    if query_intent:
        print(f"{Fore.MAGENTA}🎯 意图识别结果: {Style.RESET_ALL}")
        print(f"  意图类型: {query_intent.get('intent_type', 'N/A')}")
        print(f"  复杂度: {query_intent.get('complexity', 'N/A')}")
        if query_intent.get('needs_decomposition'):
            print(f"  需要分解: 是 ({query_intent.get('decomposition_type', 'N/A')})")
            sub_queries = query_intent.get('sub_queries', [])
            if sub_queries:
                print(f"  子查询数: {len(sub_queries)}")
        print()

    # 打印最终答案
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

    # 打印Agent执行历史
    agent_history = result.get("agent_history", [])
    if agent_history:
        print(f"{Fore.BLUE}Agent执行历史: {Style.RESET_ALL}")
        for i, record in enumerate(agent_history, 1):
            agent_name = record.get("agent", "unknown")
            metadata = record.get("metadata", {})
            print(f"  {i}. {agent_name}")
            if "answer_quality" in metadata:
                print(f"     答案质量: {metadata['answer_quality']:.2f}")
            if "retrieval_quality" in metadata:
                print(f"     检索质量: {metadata['retrieval_quality']:.2f}")
        print()

    # 打印统计信息
    print(f"{Fore.CYAN}统计信息: {Style.RESET_ALL}")
    print(f"  迭代次数: {result.get('iteration_count', 0)}")
    print(f"  使用的Agent: {result.get('current_agent', 'N/A')}")
    if result.get("error_message"):
        print(f"  错误: {Fore.RED}{result['error_message']}{Style.RESET_ALL}")
    print()


def main():
    """主函数"""
    print_section("多Agent系统示例")
    print("本示例展示多Agent系统的使用方式。")
    print("系统会自动分析用户意图，并路由到合适的Agent。\n")
    
    # 初始化多Agent系统
    print_section("初始化多Agent系统")
    print("正在初始化...")
    
    graph = MultiAgentGraph(
        rag_persist_directory="./tmp/chroma_db/agentic_rag",
        max_iterations=10
    )
    
    print(f"{Fore.GREEN}✓ 多Agent系统初始化完成{Style.RESET_ALL}\n")
    print(f"可用Agent:")
    available_agents = graph.supervisor.get_available_agents()
    for agent in available_agents:
        print(f"  - {agent['name']}: {agent['description']}")
    print()
    
    # 测试查询
    test_questions = [
      # "广东有哪些知名粤菜?",
      # "中国有哪些著名的旅游景点最受欢迎?",
      "黑悟空游戏怎样?",
        # "2019-2021年福布斯富豪榜杰夫·贝索斯财富是多少?",
        # "为什么我的快递还没到?",  # 应该路由到chat_agent
        # "你好，介绍一下你自己",  # 应该路由到chat_agent
    ]
    
    for i, question in enumerate(test_questions, 1):
        print_section(f"测试 {i}/{len(test_questions)}")
        print(f"{Fore.CYAN}问题: {Style.RESET_ALL}{question}\n")
        
        # 执行查询
        result = graph.invoke(question)
        
        # 打印结果
        print_result(result, question)
        
        print("\n" + "-"*60 + "\n")
    
    print_section("示例完成")
    print("💡 提示:")
    print("1. 系统会先进行意图识别，分析用户查询的类型和复杂度")
    print("2. 然后Supervisor根据意图识别结果智能路由到合适的Agent")
    print("3. 需要知识检索的问题会路由到RAG Agent")
    print("4. 一般对话会路由到Chat Agent")
    print("5. 可以通过添加新的Agent来扩展系统功能")


if __name__ == "__main__":
    main()

