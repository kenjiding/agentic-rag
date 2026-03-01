"""评测系统使用示例

本文件提供评测系统的各种使用示例，可以直接复制使用。

运行方式:
    python -m src.evaluation.examples
"""
import asyncio
from pathlib import Path


def example_1_quick_start():
    """示例1: 快速开始 - 最简单的评测"""
    print("\n" + "=" * 50)
    print("示例1: 快速开始")
    print("=" * 50)
    
    code = '''
from src.evaluation.models import EvaluationCase, EvaluationResult
from src.evaluation.evaluators.task_completion import TaskCompletionEvaluator
from src.evaluation.config import EvaluationConfig

# 创建评测用例
case = EvaluationCase(
    case_id="test_001",
    input_messages=["帮我搜索西门子冰箱"],
    expected_intent="PRODUCT_SEARCH",
    expected_agent="product_agent",
    expected_tool_calls=["search_products"],
)

# 创建评测器
config = EvaluationConfig.default()
evaluator = TaskCompletionEvaluator(config)

# 模拟执行轨迹（实际使用时从LangGraph获取）
execution_trace = [
    {"node": "intent_router", "event": "intent_classified", "intent_type": "PRODUCT_SEARCH"},
    {"node": "product_agent", "event": "tool_called", "tool_name": "search_products"},
]

# 模拟最终状态
from src.multi_agent.state import MultiAgentState
final_state = MultiAgentState(content="找到3款西门子冰箱...")

# 执行评测
import asyncio
result = asyncio.run(evaluator.evaluate(case, execution_trace, final_state))

print(f"评测结果: {'通过' if result.success else '失败'}")
print(f"任务完成度: {result.task_completion_score:.2f}")
'''
    print(code)


def example_2_batch_evaluation():
    """示例2: 批量评测"""
    print("\n" + "=" * 50)
    print("示例2: 批量评测")
    print("=" * 50)
    
    code = '''
from src.evaluation.config import EvaluationConfig
from src.evaluation.datasets.loader import DatasetLoader
from src.evaluation.runners.single_run import SingleRunner
from src.evaluation.runners.batch_runner import BatchRunner

# 加载配置
config = EvaluationConfig.from_yaml("config/evaluation.yaml")

# 加载数据集
loader = DatasetLoader(config=config)
cases = loader.load("data/eval/basic_cases.json")

# 创建运行器（需要提供LangGraph图）
from src.multi_agent.graph import MultiAgentGraph
graph = MultiAgentGraph().build()

runner = SingleRunner(graph=graph, config=config)
batch_runner = BatchRunner(single_runner=runner, config=config)

# 设置进度回调
def on_progress(completed, total, result):
    print(f"[{completed}/{total}] {result.case_id}: {'✓' if result.success else '✗'}")

batch_runner.set_progress_callback(on_progress)

# 运行批量评测
import asyncio
summary = asyncio.run(batch_runner.run(cases, name="批量评测"))

print(f"\\n通过率: {summary.overall_success_rate:.1%}")
'''
    print(code)


def example_3_consistency_test():
    """示例3: 一致性测试（pass@k）"""
    print("\n" + "=" * 50)
    print("示例3: 一致性测试")
    print("=" * 50)
    
    code = '''
from src.evaluation.runners.batch_runner import BatchRunner

# 运行一致性测试
# 每个用例执行5次，计算pass@5和pass^5
summary = await batch_runner.run_consistency_test(
    cases,
    k=5,  # 执行次数
    name="一致性测试"
)

# 查看一致性指标
metrics = summary.consistency_metrics
print(f"pass@{metrics.k}: {metrics.pass_at_k}")   # 至少成功1次
print(f"pass^{metrics.k}: {metrics.pass_power_k}") # 全部成功
print(f"成功率: {metrics.success_rate:.1%}")
'''
    print(code)


def example_4_safety_evaluation():
    """示例4: 安全合规评测"""
    print("\n" + "=" * 50)
    print("示例4: 安全合规评测")
    print("=" * 50)
    
    code = '''
from src.evaluation.evaluators.safety import SafetyEvaluator, PolicyRule, COMMON_POLICIES
from src.evaluation.config import EvaluationConfig

# 使用预定义策略
evaluator = SafetyEvaluator(
    policies=[
        COMMON_POLICIES["no_unauthorized_orders"],
        COMMON_POLICIES["pii_protection"],
    ],
    config=EvaluationConfig.default()
)

# 或者定义自定义策略
custom_policy = PolicyRule(
    rule_id="max_order_amount",
    name="订单金额限制",
    description="单笔订单不能超过10000元",
    severity="high",
    detection_type="event_pattern",
    detection_config={
        "pattern": {"event": "order_amount_exceeded"}
    }
)
evaluator.add_policy(custom_policy)

# 执行安全评测
result = await evaluator.evaluate(case, execution_trace, final_state)

print(f"合规分数: {result.policy_compliance_score:.2f}")
print(f"违规数量: {len(result.policy_violations)}")
for v in result.policy_violations:
    print(f"  - {v.rule_name} ({v.severity})")
'''
    print(code)


def example_5_generate_reports():
    """示例5: 生成评测报告"""
    print("\n" + "=" * 50)
    print("示例5: 生成评测报告")
    print("=" * 50)
    
    code = '''
from src.evaluation.reporters.json_reporter import JSONReporter
from src.evaluation.reporters.html_reporter import HTMLReporter

# 生成JSON报告
json_reporter = JSONReporter(output_dir="reports/evaluation")
json_path = json_reporter.generate_summary_report(
    summary,
    filename="my_evaluation.json",
    include_traces=True  # 包含执行轨迹
)
print(f"JSON报告: {json_path}")

# 生成HTML可视化报告
html_reporter = HTMLReporter(output_dir="reports/evaluation")
html_path = html_reporter.generate_summary_report(
    summary,
    filename="my_evaluation.html"
)
print(f"HTML报告: {html_path}")

# 在浏览器中打开HTML报告
import webbrowser
webbrowser.open(f"file://{html_path}")
'''
    print(code)


def example_6_langsmith_integration():
    """示例6: LangSmith集成"""
    print("\n" + "=" * 50)
    print("示例6: LangSmith集成")
    print("=" * 50)
    
    code = '''
# 需要设置环境变量:
# export LANGCHAIN_API_KEY=your_api_key
# export LANGCHAIN_PROJECT=my-project

from src.evaluation.integrations.langsmith import LangSmithIntegration

# 初始化LangSmith集成
langsmith = LangSmithIntegration(project_name="multi-agent-eval")

# 创建数据集
dataset_id = langsmith.create_dataset(
    name="customer_service_v1",
    cases=cases,
    description="客服系统评测数据集"
)

# 运行在线评测
summary = await langsmith.run_evaluation(
    graph=graph,
    dataset_name="customer_service_v1",
    experiment_prefix="eval_v1"
)

# 查看数据集列表
datasets = langsmith.list_datasets()
for d in datasets:
    print(f"- {d['name']}: {d['id']}")
'''
    print(code)


def example_7_custom_evaluator():
    """示例7: 自定义评测器"""
    print("\n" + "=" * 50)
    print("示例7: 自定义评测器")
    print("=" * 50)
    
    code = '''
from src.evaluation.evaluators.base import BaseAgentEvaluator
from src.evaluation.models import EvaluationCase, EvaluationResult, EvaluationStatus
from datetime import datetime

class ResponseQualityEvaluator(BaseAgentEvaluator):
    """自定义响应质量评测器"""
    
    def __init__(self, config=None):
        super().__init__(config, name="ResponseQualityEvaluator")
    
    async def evaluate(self, case, execution_trace, final_state):
        result = self._create_base_result(case, execution_trace, final_state)
        
        # 自定义评测逻辑
        response = final_state.content or ""
        
        # 检查响应长度
        length_score = min(len(response) / 100, 1.0)
        
        # 检查是否包含关键信息
        keywords = ["抱歉", "谢谢", "请", "为您"]
        keyword_score = sum(1 for k in keywords if k in response) / len(keywords)
        
        # 综合评分
        result.task_completion_score = (length_score + keyword_score) / 2
        result.success = result.task_completion_score >= 0.5
        result.status = EvaluationStatus.SUCCESS if result.success else EvaluationStatus.FAILED
        result.completed_at = datetime.now()
        
        return result

# 使用自定义评测器
evaluator = ResponseQualityEvaluator()
result = await evaluator.evaluate(case, trace, state)
'''
    print(code)


def example_8_generate_dataset():
    """示例8: 生成合成数据集"""
    print("\n" + "=" * 50)
    print("示例8: 生成合成数据集")
    print("=" * 50)
    
    code = '''
from src.evaluation.datasets.generator import DatasetGenerator

# 创建生成器
generator = DatasetGenerator()

# 查看可用场景
print("可用场景:", list(generator._templates.keys()))
# ['greeting', 'product_search', 'order_query', 'order_create', 
#  'knowledge_qa', 'product_comparison']

# 生成特定场景的用例
product_cases = generator.generate("product_search", count=10)
order_cases = generator.generate("order_create", count=5)

# 生成所有场景的用例
all_cases = generator.generate_all(count_per_scenario=5)

# 导出到文件
generator.export_to_json(
    all_cases,
    path="data/eval/generated_cases.json",
    metadata={"version": "1.0", "generated": True}
)

# 添加自定义模板
generator.add_template("custom_scenario", {
    "input_patterns": [
        "自定义问题: {param1}",
    ],
    "params": {"param1": ["值1", "值2"]},
    "expected_intent": "CUSTOM",
    "tags": ["custom"],
})
'''
    print(code)


def main():
    """显示所有示例"""
    print("\n" + "=" * 60)
    print("🔬 多Agent智能客服评测系统 - 使用示例")
    print("=" * 60)
    
    examples = [
        example_1_quick_start,
        example_2_batch_evaluation,
        example_3_consistency_test,
        example_4_safety_evaluation,
        example_5_generate_reports,
        example_6_langsmith_integration,
        example_7_custom_evaluator,
        example_8_generate_dataset,
    ]
    
    for example in examples:
        example()
    
    print("\n" + "=" * 60)
    print("📚 更多信息请查看:")
    print("   - config/evaluation.yaml  # 评测配置")
    print("   - data/eval/              # 评测数据集")
    print("   - scripts/run_evaluation.py  # 运行脚本")
    print("=" * 60)


if __name__ == "__main__":
    main()
