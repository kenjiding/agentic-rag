"""JSON报告生成器

生成JSON格式的评测报告。

设计原则：
- 结构化数据输出
- 支持增量写入
- 可配置的详细程度
"""
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import json
import logging

from src.evaluation.models import (
    EvaluationResult,
    EvaluationSummary,
)
from src.evaluation.config import EvaluationConfig

logger = logging.getLogger(__name__)


class JSONReporter:
    """JSON报告生成器
    
    生成结构化的JSON评测报告，支持：
    - 单个结果报告
    - 汇总报告
    - 可配置的详细程度
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """初始化JSON报告生成器
        
        Args:
            output_dir: 输出目录
            config: 评测配置
        """
        self.config = config or EvaluationConfig.default()
        self.output_dir = Path(output_dir or self.config.reporter.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_result_report(
        self,
        result: EvaluationResult,
        filename: Optional[str] = None,
        include_trace: Optional[bool] = None
    ) -> Path:
        """生成单个评测结果报告
        
        Args:
            result: 评测结果
            filename: 文件名（可选）
            include_trace: 是否包含执行轨迹
            
        Returns:
            报告文件路径
        """
        include_trace = include_trace if include_trace is not None else self.config.reporter.include_traces
        
        # 准备报告数据
        report_data = self._prepare_result_data(result, include_trace)
        
        # 生成文件名
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_{result.case_id}_{timestamp}.json"
        
        # 写入文件
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"生成JSON报告: {output_path}")
        return output_path
    
    def generate_summary_report(
        self,
        summary: EvaluationSummary,
        filename: Optional[str] = None,
        include_individual_results: bool = True,
        include_traces: Optional[bool] = None
    ) -> Path:
        """生成评测汇总报告
        
        Args:
            summary: 评测汇总
            filename: 文件名（可选）
            include_individual_results: 是否包含各用例详细结果
            include_traces: 是否包含执行轨迹
            
        Returns:
            报告文件路径
        """
        include_traces = include_traces if include_traces is not None else self.config.reporter.include_traces
        
        # 准备报告数据
        report_data = self._prepare_summary_data(
            summary,
            include_individual_results,
            include_traces
        )
        
        # 生成文件名
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{summary.summary_id}_{timestamp}.json"
        
        # 写入文件
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"生成JSON汇总报告: {output_path}")
        return output_path
    
    def _prepare_result_data(
        self,
        result: EvaluationResult,
        include_trace: bool
    ) -> Dict[str, Any]:
        """准备单个结果的报告数据"""
        data = {
            "metadata": {
                "report_type": "evaluation_result",
                "generated_at": datetime.now().isoformat(),
                "case_id": result.case_id,
            },
            "summary": {
                "status": result.status.value,
                "success": result.success,
                "overall_score": result.overall_score,
            },
            "scores": {
                "task_completion": result.task_completion_score,
                "trajectory_quality": result.trajectory_quality_score,
                "tool_accuracy": result.tool_accuracy_score,
                "policy_compliance": result.policy_compliance_score,
            },
            "performance": {
                "latency_ms": result.latency_ms,
                "token_usage": result.token_usage,
                "step_count": result.step_count,
            },
            "actual_outputs": {
                "intent": result.actual_intent,
                "agent": result.actual_agent,
                "tool_calls": result.actual_tool_calls,
                "response": result.final_response,
            },
            "milestones": [m.model_dump() for m in result.milestone_results],
            "policy_violations": [v.model_dump() for v in result.policy_violations],
            "details": result.details,
        }
        
        if include_trace:
            data["execution_trace"] = result.execution_trace
        
        if result.error_message:
            data["error"] = result.error_message
        
        return data
    
    def _prepare_summary_data(
        self,
        summary: EvaluationSummary,
        include_results: bool,
        include_traces: bool
    ) -> Dict[str, Any]:
        """准备汇总报告数据"""
        data = {
            "metadata": {
                "report_type": "evaluation_summary",
                "generated_at": datetime.now().isoformat(),
                "summary_id": summary.summary_id,
                "name": summary.name,
                "description": summary.description,
            },
            "statistics": {
                "total_cases": summary.total_cases,
                "passed_cases": summary.passed_cases,
                "failed_cases": summary.failed_cases,
                "error_cases": summary.error_cases,
                "success_rate": summary.overall_success_rate,
            },
            "average_scores": {
                "task_completion": summary.avg_task_completion,
                "trajectory_quality": summary.avg_trajectory_quality,
                "tool_accuracy": summary.avg_tool_accuracy,
                "policy_compliance": summary.avg_policy_compliance,
            },
            "performance": {
                "avg_latency_ms": summary.avg_latency_ms,
                "total_token_usage": summary.total_token_usage,
            },
            "safety": {
                "total_violations": summary.total_violations,
                "violations_by_rule": summary.violations_by_rule,
            },
            "results_by_tag": summary.results_by_tag,
        }
        
        # 添加一致性指标
        if summary.consistency_metrics:
            data["consistency"] = summary.consistency_metrics.model_dump()
        
        # 添加时间信息
        if summary.started_at:
            data["metadata"]["started_at"] = summary.started_at.isoformat()
        if summary.completed_at:
            data["metadata"]["completed_at"] = summary.completed_at.isoformat()
            if summary.started_at:
                duration = (summary.completed_at - summary.started_at).total_seconds()
                data["metadata"]["duration_seconds"] = duration
        
        # 添加个别结果
        if include_results:
            data["results"] = [
                self._prepare_result_data(r, include_traces)
                for r in summary.results
            ]
        
        return data
    
    def to_json_string(
        self,
        summary: EvaluationSummary,
        include_results: bool = True,
        include_traces: bool = False
    ) -> str:
        """转换为JSON字符串
        
        Args:
            summary: 评测汇总
            include_results: 是否包含各用例详细结果
            include_traces: 是否包含执行轨迹
            
        Returns:
            JSON字符串
        """
        data = self._prepare_summary_data(summary, include_results, include_traces)
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
