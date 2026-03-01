"""HTML报告生成器

生成可视化的HTML评测报告。

设计原则：
- 美观的可视化展示
- 交互式图表
- 支持详细信息展开
"""
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import logging

from src.evaluation.models import (
    EvaluationResult,
    EvaluationSummary,
)
from src.evaluation.config import EvaluationConfig

logger = logging.getLogger(__name__)


class HTMLReporter:
    """HTML报告生成器
    
    生成可视化的HTML评测报告，支持：
    - 综合仪表盘
    - 图表展示
    - 详细结果表格
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """初始化HTML报告生成器
        
        Args:
            output_dir: 输出目录
            config: 评测配置
        """
        self.config = config or EvaluationConfig.default()
        self.output_dir = Path(output_dir or self.config.reporter.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_summary_report(
        self,
        summary: EvaluationSummary,
        filename: Optional[str] = None
    ) -> Path:
        """生成评测汇总HTML报告
        
        Args:
            summary: 评测汇总
            filename: 文件名（可选）
            
        Returns:
            报告文件路径
        """
        # 生成文件名
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{summary.summary_id}_{timestamp}.html"
        
        # 生成HTML内容
        html_content = self._generate_html(summary)
        
        # 写入文件
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"生成HTML报告: {output_path}")
        return output_path
    
    def _generate_html(self, summary: EvaluationSummary) -> str:
        """生成HTML内容"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>评测报告 - {summary.name}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            padding: 24px;
            margin-bottom: 24px;
        }}
        .score-ring {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
        }}
        .status-badge {{
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 14px;
            font-weight: 500;
        }}
        .status-success {{
            background: #dcfce7;
            color: #166534;
        }}
        .status-failed {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .status-error {{
            background: #fef3c7;
            color: #92400e;
        }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="max-w-7xl mx-auto px-4 py-8">
        <!-- Header -->
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-gray-900">{summary.name}</h1>
            <p class="text-gray-600 mt-2">{summary.description}</p>
            <p class="text-sm text-gray-500 mt-1">
                生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </p>
        </header>
        
        <!-- Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {self._generate_summary_cards(summary)}
        </div>
        
        <!-- Score Overview -->
        <div class="card">
            <h2 class="text-xl font-semibold mb-6">评分概览</h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
                {self._generate_score_rings(summary)}
            </div>
        </div>
        
        <!-- Charts -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="card">
                <h2 class="text-xl font-semibold mb-4">用例状态分布</h2>
                <canvas id="statusChart"></canvas>
            </div>
            <div class="card">
                <h2 class="text-xl font-semibold mb-4">评分分布</h2>
                <canvas id="scoreChart"></canvas>
            </div>
        </div>
        
        <!-- Consistency Metrics (if available) -->
        {self._generate_consistency_section(summary)}
        
        <!-- Safety Violations -->
        {self._generate_safety_section(summary)}
        
        <!-- Results Table -->
        <div class="card">
            <h2 class="text-xl font-semibold mb-4">详细结果</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">用例ID</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">状态</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">任务完成</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">轨迹质量</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">工具准确率</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">合规性</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">延迟</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        {self._generate_results_rows(summary.results)}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // Status Chart
        new Chart(document.getElementById('statusChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['通过', '失败', '错误'],
                datasets: [{{
                    data: [{summary.passed_cases}, {summary.failed_cases}, {summary.error_cases}],
                    backgroundColor: ['#22c55e', '#ef4444', '#f59e0b'],
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ position: 'bottom' }}
                }}
            }}
        }});
        
        // Score Chart
        new Chart(document.getElementById('scoreChart'), {{
            type: 'radar',
            data: {{
                labels: ['任务完成', '轨迹质量', '工具准确率', '合规性'],
                datasets: [{{
                    label: '平均分数',
                    data: [
                        {summary.avg_task_completion:.2f},
                        {summary.avg_trajectory_quality:.2f},
                        {summary.avg_tool_accuracy:.2f},
                        {summary.avg_policy_compliance:.2f}
                    ],
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: 'rgb(59, 130, 246)',
                    pointBackgroundColor: 'rgb(59, 130, 246)',
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 1
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    def _generate_summary_cards(self, summary: EvaluationSummary) -> str:
        """生成汇总卡片"""
        success_rate_color = "text-green-600" if summary.overall_success_rate >= 0.7 else "text-red-600"
        
        return f"""
            <div class="card">
                <p class="text-sm text-gray-500">总用例数</p>
                <p class="text-3xl font-bold text-gray-900">{summary.total_cases}</p>
            </div>
            <div class="card">
                <p class="text-sm text-gray-500">通过率</p>
                <p class="text-3xl font-bold {success_rate_color}">{summary.overall_success_rate:.1%}</p>
            </div>
            <div class="card">
                <p class="text-sm text-gray-500">平均延迟</p>
                <p class="text-3xl font-bold text-gray-900">{summary.avg_latency_ms:.0f}ms</p>
            </div>
            <div class="card">
                <p class="text-sm text-gray-500">违规次数</p>
                <p class="text-3xl font-bold {'text-red-600' if summary.total_violations > 0 else 'text-green-600'}">{summary.total_violations}</p>
            </div>
        """
    
    def _generate_score_rings(self, summary: EvaluationSummary) -> str:
        """生成评分环"""
        def get_color(score: float) -> str:
            if score >= 0.8:
                return "bg-green-100 text-green-600"
            elif score >= 0.6:
                return "bg-yellow-100 text-yellow-600"
            else:
                return "bg-red-100 text-red-600"
        
        scores = [
            ("任务完成", summary.avg_task_completion),
            ("轨迹质量", summary.avg_trajectory_quality),
            ("工具准确率", summary.avg_tool_accuracy),
            ("合规性", summary.avg_policy_compliance),
        ]
        
        html = ""
        for name, score in scores:
            color = get_color(score)
            html += f"""
                <div class="flex flex-col items-center">
                    <div class="score-ring {color}">{score:.0%}</div>
                    <p class="mt-2 text-sm text-gray-600">{name}</p>
                </div>
            """
        return html
    
    def _generate_consistency_section(self, summary: EvaluationSummary) -> str:
        """生成一致性指标部分"""
        if not summary.consistency_metrics:
            return ""
        
        metrics = summary.consistency_metrics
        return f"""
        <div class="card">
            <h2 class="text-xl font-semibold mb-4">一致性指标 (k={metrics.k})</h2>
            <div class="grid grid-cols-3 gap-6">
                <div class="text-center">
                    <p class="text-3xl font-bold {'text-green-600' if metrics.pass_at_k else 'text-red-600'}">
                        {'✓' if metrics.pass_at_k else '✗'}
                    </p>
                    <p class="text-sm text-gray-600">pass@{metrics.k}</p>
                </div>
                <div class="text-center">
                    <p class="text-3xl font-bold {'text-green-600' if metrics.pass_power_k else 'text-red-600'}">
                        {'✓' if metrics.pass_power_k else '✗'}
                    </p>
                    <p class="text-sm text-gray-600">pass^{metrics.k}</p>
                </div>
                <div class="text-center">
                    <p class="text-3xl font-bold text-blue-600">{metrics.success_rate:.1%}</p>
                    <p class="text-sm text-gray-600">成功率</p>
                </div>
            </div>
        </div>
        """
    
    def _generate_safety_section(self, summary: EvaluationSummary) -> str:
        """生成安全违规部分"""
        if summary.total_violations == 0:
            return ""
        
        violations_html = ""
        for rule_id, count in summary.violations_by_rule.items():
            violations_html += f"""
                <tr>
                    <td class="px-4 py-2">{rule_id}</td>
                    <td class="px-4 py-2 text-red-600 font-medium">{count}</td>
                </tr>
            """
        
        return f"""
        <div class="card">
            <h2 class="text-xl font-semibold mb-4 text-red-600">安全违规 ({summary.total_violations})</h2>
            <table class="min-w-full">
                <thead class="bg-red-50">
                    <tr>
                        <th class="px-4 py-2 text-left text-sm font-medium text-red-800">规则ID</th>
                        <th class="px-4 py-2 text-left text-sm font-medium text-red-800">违规次数</th>
                    </tr>
                </thead>
                <tbody>
                    {violations_html}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_results_rows(self, results: List[EvaluationResult]) -> str:
        """生成结果表格行"""
        rows = ""
        for r in results:
            status_class = {
                "success": "status-success",
                "failed": "status-failed",
                "error": "status-error",
            }.get(r.status.value, "status-failed")
            
            status_text = {
                "success": "通过",
                "failed": "失败",
                "error": "错误",
            }.get(r.status.value, r.status.value)
            
            rows += f"""
                <tr>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{r.case_id}</td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <span class="status-badge {status_class}">{status_text}</span>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{r.task_completion_score:.2f}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{r.trajectory_quality_score:.2f}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{r.tool_accuracy_score:.2f}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{r.policy_compliance_score:.2f}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{r.latency_ms}ms</td>
                </tr>
            """
        return rows
