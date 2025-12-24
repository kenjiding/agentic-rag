"""意图识别模块独立测试

直接测试 intent 模块，不依赖 multi_agent。
"""
import sys
from pathlib import Path
import logging
from colorama import Fore, Style, init

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.intent import IntentClassifier, QueryIntent, SubQuery, IntentConfig
from langchain_openai import ChatOpenAI
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
    print_section("测试 1: 意图分类器基础功能")

    classifier = IntentClassifier()

    test_queries = [
        ("简单事实查询", "北京的人口是多少?"),
        ("因果查询", "为什么我的快递还没到?"),
        ("对比查询", "2019年和2020年苹果营收对比"),
        ("分析查询", "分析特斯拉成功的原因"),
        ("对话", "你好，介绍一下你自己"),
        ("复杂查询", "2019到2021年福布斯富豪榜杰夫·贝索斯财富变化趋势是什么？"),
    ]

    for desc, query in test_queries:
        print(f"{Fore.YELLOW}[{desc}]{Style.RESET_ALL} {query}")
        intent = classifier.classify(query)

        print(f"  {Fore.GREEN}类型:{Style.RESET_ALL} {intent.intent_type}")
        print(f"  {Fore.GREEN}复杂度:{Style.RESET_ALL} {intent.complexity}")
        print(f"  {Fore.GREEN}需分解:{Style.RESET_ALL} {intent.needs_decomposition}")
        if intent.needs_decomposition:
            print(f"  {Fore.GREEN}分解类型:{Style.RESET_ALL} {intent.decomposition_type}")
            print(f"  {Fore.GREEN}子查询数:{Style.RESET_ALL} {len(intent.sub_queries)}")
            for i, sq in enumerate(intent.sub_queries[:2], 1):
                print(f"    {i}. {sq.query[:50]}...")
        print(f"  {Fore.GREEN}置信度:{Style.RESET_ALL} {intent.confidence:.2f}")
        print()


def test_intent_with_config():
    """测试使用自定义配置的意图分类器"""
    print_section("测试 2: 自定义配置")

    config = IntentConfig(
        llm_temperature=0.0,
        llm_model="gpt-4o-mini",
        enable_intent_classification=True,
        min_confidence=0.7
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    classifier = IntentClassifier(llm=llm, config=config)

    query = "对比分析iPhone和Android手机的优缺点"
    print(f"{Fore.YELLOW}查询:{Style.RESET_ALL} {query}")

    intent = classifier.classify(query)

    print(f"  {Fore.GREEN}类型:{Style.RESET_ALL} {intent.intent_type}")
    print(f"  {Fore.GREEN}复杂度:{Style.RESET_ALL} {intent.complexity}")
    print(f"  {Fore.GREEN}需分解:{Style.RESET_ALL} {intent.needs_decomposition}")
    if intent.needs_decomposition:
        print(f"  {Fore.GREEN}分解类型:{Style.RESET_ALL} {intent.decomposition_type}")
        print(f"  {Fore.GREEN}子查询:{Style.RESET_ALL}")
        for i, sq in enumerate(intent.sub_queries, 1):
            print(f"    {i}. [{sq.purpose}] {sq.query}")
            print(f"       策略: {sq.recommended_strategy}, k={sq.recommended_k}")
    print()


def test_query_intent_model():
    """测试 QueryIntent 数据模型"""
    print_section("测试 3: 数据模型")

    from src.intent.models import QueryIntent, SubQuery

    # 创建一个测���意图 - 使用字符串值（Literal类型不能直接作为值使用）
    intent = QueryIntent(
        intent_type="comparison",
        complexity="moderate",
        needs_decomposition=True,
        decomposition_type="comparison",
        decomposition_reason="需要对比两个对象",
        sub_queries=[
            SubQuery(
                query="iPhone的优缺点是什么？",
                purpose="获取iPhone的信息",
                recommended_strategy=["semantic"],
                recommended_k=5,
                order=0
            ),
            SubQuery(
                query="Android手机的优缺点是什么？",
                purpose="获取Android的信息",
                recommended_strategy=["semantic"],
                recommended_k=5,
                order=0
            ),
        ],
        entities=["iPhone", "Android"],
        recommended_retrieval_strategy=["hybrid"],
        recommended_k=10,
        needs_multi_round_retrieval=False,
        confidence=0.95,
        reasoning="这是一个典型���对比查询，需要分别获取两个对象的信息"
    )

    print(f"{Fore.GREEN}✓ QueryIntent 模型创建成功{Style.RESET_ALL}")
    print(f"  类型: {intent.intent_type}")
    print(f"  复杂度: {intent.complexity}")
    print(f"  子查询数: {len(intent.sub_queries)}")

    # 测试序列化
    intent_dict = intent.model_dump()
    print(f"{Fore.GREEN}✓ 序列化成功，字段数: {len(intent_dict)}{Style.RESET_ALL}")

    # 测试反序列化
    intent_restored = QueryIntent(**intent_dict)
    print(f"{Fore.GREEN}✓ 反序列化成功{Style.RESET_ALL}")
    print()


def main():
    """主函数"""
    print_section("意图识别模块独立测试")
    print("本示例测试通用意图识别模块的核心功能。\n")

    try:
        # 测试1: 基础意图分类
        test_intent_classifier()

        # 测试2: 自定义配置
        test_intent_with_config()

        # 测试3: 数据模型
        test_query_intent_model()

        print_section("测试完成")
        print(f"{Fore.GREEN}✓ 所有测试通过！{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}集成要点:{Style.RESET_ALL}")
        print("1. 意图识别模块已迁移到 src/intent/")
        print("2. 可以独立于 agentic_rag 使用")
        print("3. 支持多种意图类型和查询分解")
        print("4. 与 multi_agent 集成流程: 用户问题 → 意图识别 → Supervisor 路由")

    except Exception as e:
        print(f"{Fore.RED}测试失败: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
