#!/usr/bin/env python3
"""评测系统 - 真实系统评测

使用方法:
    1. 基础评测（使用真实LangGraph系统）:
       python scripts/run_evaluation.py
    
    2. 指定数据集:
       python scripts/run_evaluation.py --dataset basic
       python scripts/run_evaluation.py --dataset order
       python scripts/run_evaluation.py --dataset all
    
    3. 一致性测试:
       python scripts/run_evaluation.py --consistency --k 3
    
    4. 演示模式（不调用LLM，使用模拟数据）:
       python scripts/run_evaluation.py --demo

示例输出:
    - JSON报告: reports/evaluation/summary_xxx.json
    - HTML报告: reports/evaluation/report_xxx.html
"""
import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.config import EvaluationConfig
from src.evaluation.datasets.loader import DatasetLoader
from src.evaluation.runners.single_run import SingleRunner
from src.evaluation.runners.batch_runner import BatchRunner
from src.evaluation.reporters.json_reporter import JSONReporter
from src.evaluation.reporters.html_reporter import HTMLReporter
from src.evaluation.models import EvaluationCase, EvaluationSummary

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_summary(summary: EvaluationSummary) -> None:
    """打印评测摘要"""
    print("\n" + "=" * 60)
    print(f"📊 评测报告: {summary.name}")
    print("=" * 60)
    
    print(f"\n📈 总体统计:")
    print(f"   总用例数: {summary.total_cases}")
    print(f"   通过用例: {summary.passed_cases}")
    print(f"   失败用例: {summary.failed_cases}")
    print(f"   错误用例: {summary.error_cases}")
    print(f"   通过率: {summary.overall_success_rate:.1%}")
    
    print(f"\n📊 平均分数:")
    print(f"   任务完成度: {summary.avg_task_completion:.2f}")
    print(f"   轨迹质量: {summary.avg_trajectory_quality:.2f}")
    print(f"   工具准确率: {summary.avg_tool_accuracy:.2f}")
    print(f"   策略合规: {summary.avg_policy_compliance:.2f}")
    
    print(f"\n⏱️  性能指标:")
    print(f"   平均延迟: {summary.avg_latency_ms:.0f}ms")
    print(f"   Token消耗: {summary.total_token_usage}")
    
    if summary.total_violations > 0:
        print(f"\n⚠️  安全违规: {summary.total_violations}")
        for rule_id, count in summary.violations_by_rule.items():
            print(f"   - {rule_id}: {count}次")
    
    if summary.consistency_metrics:
        m = summary.consistency_metrics
        print(f"\n🔄 一致性指标 (k={m.k}):")
        print(f"   pass@{m.k}: {'✓' if m.pass_at_k else '✗'}")
        print(f"   pass^{m.k}: {'✓' if m.pass_power_k else '✗'}")
        print(f"   成功率: {m.success_rate:.1%}")
    
    # 打印每个用例的详细结果
    print(f"\n📋 详细结果:")
    for r in summary.results:
        status = "✓" if r.success else "✗"
        print(f"   {status} {r.case_id}: 任务={r.task_completion_score:.2f}, 轨迹={r.trajectory_quality_score:.2f}, 延迟={r.latency_ms}ms")
        if r.error_message:
            print(f"      错误: {r.error_message[:50]}...")
    
    print("\n" + "=" * 60)


def create_real_graph():
    """创建真实的LangGraph系统"""
    from src.multi_agent.graph import MultiAgentGraph
    
    print("🔧 初始化多Agent系统...")
    graph_builder = MultiAgentGraph(
        init_web_search=False,  # 评测时关闭web搜索加速
        enable_business_agents=True,
    )
    # MultiAgentGraph在初始化时已经构建好graph，直接访问
    graph = graph_builder.graph
    print("✓ 多Agent系统初始化完成")
    
    return graph


async def run_real_evaluation(
    dataset_name: str = "basic",
    max_cases: Optional[int] = None,
    run_consistency: bool = False,
    consistency_k: int = 3
):
    """运行真实系统评测
    
    Args:
        dataset_name: 数据集名称 (basic, order, product, edge, all)
        max_cases: 最大测试用例数（用于快速测试）
        run_consistency: 是否运行一致性测试
        consistency_k: 一致性测试的k值
    """
    print("\n🚀 启动真实系统评测...\n")
    
    # 1. 加载配置
    config_path = project_root / "config" / "evaluation.yaml"
    if config_path.exists():
        config = EvaluationConfig.from_yaml(str(config_path))
        print(f"✓ 加载配置文件: {config_path}")
    else:
        config = EvaluationConfig.default()
        print("✓ 使用默认配置")
    
    # 2. 加载数据集
    loader = DatasetLoader(base_path=project_root, config=config)
    
    dataset_map = {
        "basic": "data/eval/basic_cases.json",
        "order": "data/eval/order_flow_cases.json",
        "product": "data/eval/product_search_cases.json",
        "edge": "data/eval/edge_cases.json",
    }
    
    if dataset_name == "all":
        cases = []
        for name, path in dataset_map.items():
            cases.extend(loader.load(path))
        print(f"✓ 加载所有数据集: 共 {len(cases)} 个用例")
    else:
        dataset_path = dataset_map.get(dataset_name, dataset_map["basic"])
        cases = loader.load(dataset_path)
        print(f"✓ 加载数据集 [{dataset_name}]: {len(cases)} 个用例")
    
    # 限制用例数量（用于快速测试）
    if max_cases and len(cases) > max_cases:
        cases = cases[:max_cases]
        print(f"   (限制为前 {max_cases} 个用例)")
    
    # 3. 创建真实的LangGraph
    graph = create_real_graph()
    
    # 4. 创建运行器
    runner = SingleRunner(graph=graph, config=config)
    batch_runner = BatchRunner(single_runner=runner, config=config)
    print("✓ 创建评测运行器")
    
    # 5. 设置进度回调
    def progress_callback(completed: int, total: int, result):
        status = "✓" if result.success else "✗"
        latency = f"{result.latency_ms}ms" if result.latency_ms else "N/A"
        print(f"   [{completed}/{total}] {result.case_id}: {status} ({latency})")
    
    batch_runner.set_progress_callback(progress_callback)
    
    # 6. 运行评测
    if run_consistency:
        print(f"\n📋 开始一致性评测 (k={consistency_k})...")
        summary = await batch_runner.run_consistency_test(
            cases,
            k=consistency_k,
            name=f"一致性评测 (k={consistency_k})",
            description=f"数据集: {dataset_name}, 一致性测试"
        )
    else:
        print("\n📋 开始评测...")
        summary = await batch_runner.run(
            cases,
            name=f"系统评测 - {dataset_name}",
            description=f"数据集: {dataset_name}"
        )
    
    # 7. 打印结果
    print_summary(summary)
    
    # 8. 生成报告
    output_dir = project_root / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_reporter = JSONReporter(str(output_dir), config)
    json_path = json_reporter.generate_summary_report(summary)
    print(f"\n📄 JSON报告: {json_path}")
    
    html_reporter = HTMLReporter(str(output_dir), config)
    html_path = html_reporter.generate_summary_report(summary)
    print(f"📄 HTML报告: {html_path}")
    
    return summary


async def run_demo_evaluation():
    """运行演示评测（使用模拟数据，不调用LLM）"""
    from src.evaluation.runners.single_run import MockGraphRunner
    
    print("\n🚀 启动演示评测（模拟模式）...\n")
    print("⚠️  注意: 这是演示模式，使用模拟数据，不调用真实LLM\n")
    
    # 加载配置
    config_path = project_root / "config" / "evaluation.yaml"
    config = EvaluationConfig.from_yaml(str(config_path)) if config_path.exists() else EvaluationConfig.default()
    print(f"✓ 加载配置")
    
    # 加载数据集
    loader = DatasetLoader(base_path=project_root, config=config)
    cases = loader.load("data/eval/basic_cases.json")
    print(f"✓ 加载评测用例: {len(cases)} 个")
    
    # 创建模拟运行器
    mock_runner = MockGraphRunner(config=config)
    
    # 添加模拟响应
    mock_runner.add_mock_response(
        case_id="basic_greeting_01",
        execution_trace=[
            {"node": "intent_router", "event": "intent_classified", "intent_type": "GREETING"},
            {"node": "chat_agent", "event": "agent_executed", "agent_name": "chat_agent"},
        ],
        final_state_dict={"content": "你好！", "messages": [], "current_agent": "chat_agent"}
    )
    mock_runner.add_mock_response(
        case_id="basic_product_01",
        execution_trace=[
            {"node": "intent_router", "event": "intent_classified", "intent_type": "PRODUCT_SEARCH"},
            {"node": "product_agent", "event": "tool_called", "tool_name": "search_products"},
        ],
        final_state_dict={"content": "找到西门子冰箱...", "messages": [], "current_agent": "product_agent"}
    )
    
    batch_runner = BatchRunner(config=config, single_runner=mock_runner)
    
    # 运行评测
    print("\n📋 开始演示评测...")
    
    def progress_callback(completed: int, total: int, result):
        status = "✓" if result.success else "✗"
        print(f"   [{completed}/{total}] {result.case_id}: {status}")
    
    batch_runner.set_progress_callback(progress_callback)
    
    summary = await batch_runner.run(cases[:5], name="演示评测", description="模拟数据演示")
    
    print_summary(summary)
    
    # 生成报告
    output_dir = project_root / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_reporter = JSONReporter(str(output_dir), config)
    json_path = json_reporter.generate_summary_report(summary)
    print(f"\n📄 JSON报告: {json_path}")
    
    html_reporter = HTMLReporter(str(output_dir), config)
    html_path = html_reporter.generate_summary_report(summary)
    print(f"📄 HTML报告: {html_path}")


async def main():
    parser = argparse.ArgumentParser(description="多Agent系统评测工具")
    parser.add_argument(
        "--dataset", 
        choices=["basic", "order", "product", "edge", "all"],
        default="basic",
        help="选择数据集 (默认: basic)"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="限制最大测试用例数（用于快速测试）"
    )
    parser.add_argument(
        "--consistency",
        action="store_true",
        help="运行一致性测试 (pass@k)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="一致性测试的k值 (默认: 3)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="演示模式（使用模拟数据，不调用LLM）"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        await run_demo_evaluation()
    else:
        await run_real_evaluation(
            dataset_name=args.dataset,
            max_cases=args.max_cases,
            run_consistency=args.consistency,
            consistency_k=args.k
        )


if __name__ == "__main__":
    asyncio.run(main())
