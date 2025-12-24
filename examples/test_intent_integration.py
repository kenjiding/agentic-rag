"""意图识别集成测试

测试意图识别模块集成到 multi_agent 的功能。
不依赖 RAG Agent，避免环境依赖问题。
"""
import sys
from pathlib import Path
import logging
from colorama import Fore, Style, init

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.multi_agent.state import MultiAgentState
from src.intent import IntentClassifier
from src.multi_agent.supervisor import SupervisorAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
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


def test_intent_classifier():
    """测试意图分类器"""
    print_section("测试 1: 意图分类器")

    classifier = IntentClassifier()

    test_queries = [
        "为什么我的快递还没到?",  # 简单因果查询
        "你好，介绍一下你自己",  # 简单对话
        "2019年和2020年苹果营收对比",  # 对比查询（需要分解）
        "分析特斯拉成功的原因",  # 分析查询（可能需要分解）
        "北京的人口是多少?",  # 简单事实查询
    ]

    for query in test_queries:
        print(f"{Fore.YELLOW}查询: {Style.RESET_ALL}{query}")
        intent = classifier.classify(query)

        print(f"{Fore.GREEN}  意图类型: {Style.RESET_ALL}{intent.intent_type}")
        print(f"{Fore.GREEN}  复杂度: {Style.RESET_ALL}{intent.complexity}")
        print(f"{Fore.GREEN}  需要分解: {Style.RESET_ALL}{intent.needs_decomposition}")
        if intent.needs_decomposition:
            print(f"{Fore.GREEN}  分解类型: {Style.RESET_ALL}{intent.decomposition_type}")
            print(f"{Fore.GREEN}  子查询数: {Style.RESET_ALL}{len(intent.sub_queries)}")
            for i, sq in enumerate(intent.sub_queries[:2], 1):
                print(f"    {i}. {sq.query[:50]}...")
        print(f"{Fore.GREEN}  置信度: {Style.RESET_ALL}{intent.confidence:.2f}")
        print()


def test_intent_in_state():
    """测试意图识别结果在状态中的存储和使用"""
    print_section("测试 2: 意图识别与状态集成")

    # 初始化
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    classifier = IntentClassifier(llm=llm)
    supervisor = SupervisorAgent(llm=llm)

    # 创建测试状态
    test_queries = [
        "2019年和2020年苹果营收对比是多少？",
        "你好",
    ]

    for question in test_queries:
        print(f"{Fore.YELLOW}查询: {Style.RESET_ALL}{question}")

        # 创建初始状态
        state: MultiAgentState = {
            "messages": [HumanMessage(content=question)],
            "current_agent": None,
            "agent_results": {},
            "agent_history": [],
            "tools_used": [],
            "metadata": {},
            "error_message": None,
            "iteration_count": 0,
            "max_iterations": 10,
            "next_action": None,
            "routing_reason": None,
            "query_intent": None,
            "original_question": None,
        }

        # 步骤1: 意图识别
        print(f"{Fore.MAGENTA}步骤1: 意图识别{Style.RESET_ALL}")
        intent = classifier.classify(question)
        intent_dict = intent.model_dump()

        print(f"  意图类型: {intent.intent_type}")
        print(f"  复杂度: {intent.complexity}")
        print(f"  需要分解: {intent.needs_decomposition}")
        if intent.needs_decomposition:
            print(f"  分解类型: {intent.decomposition_type}")
            print(f"  子查询数: {len(intent.sub_queries)}")

        # 更新状态
        state["query_intent"] = intent_dict
        state["original_question"] = question
        if intent.needs_decomposition and intent.sub_queries:
            state["metadata"]["sub_queries"] = [sq.model_dump() for sq in intent.sub_queries]
            state["metadata"]["decomposition_type"] = intent.decomposition_type

        print()

        # 步骤2: Supervisor 路由决策（使用意图信息）
        print(f"{Fore.MAGENTA}步骤2: Supervisor 路由决策{Style.RESET_ALL}")
        import asyncio
        routing_decision = asyncio.run(supervisor.route(state))

        print(f"  路由决策: {routing_decision['next_action']}")
        print(f"  选中的Agent: {routing_decision.get('selected_agent', 'N/A')}")
        print(f"  路由原因: {routing_decision['routing_reason']}")
        print(f"  置信度: {routing_decision['confidence']:.2f}")
        print()


def main():
    """主函数"""
    print_section("意图识别集成测试")
    print("本示例测试意图识别模块与 multi_agent 的集成。\n")

    try:
        # 测试1: 意图分类器
        test_intent_classifier()

        # 测试2: 状态集成
        test_intent_in_state()

        print_section("测试完成")
        print(f"{Fore.GREEN}✓ 所有测试通过！{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}测试失败: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
