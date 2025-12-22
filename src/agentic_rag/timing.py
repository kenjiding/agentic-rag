"""统一耗时追踪模块

2025 最佳实践：
1. 集中管理所有阶段的耗时计算
2. 使用上下文管理器确保代码整洁
3. 支持嵌套计时和统计汇总
4. 与 LangGraph State 无缝集成
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from colorama import Fore, Style


# ==================== 数据结构 ====================

@dataclass
class TimingRecord:
    """单次计时记录"""
    stage: str  # 阶段名称
    duration_ms: float  # 耗时（毫秒）
    timestamp: float  # 开始时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于State存储）"""
        return {
            "stage": self.stage,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class TimingStats:
    """阶段统计信息"""
    stage: str
    total_ms: float
    count: int
    avg_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "total_ms": round(self.total_ms, 2),
            "count": self.count,
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2)
        }


# ==================== 阶段定义 ====================

class Stage:
    """预定义的阶段常量，避免魔法字符串"""
    INTENT = "intent_classification"  # 意图识别
    RETRIEVE = "retrieval"  # 检索
    RETRIEVE_QUALITY = "retrieval_quality_eval"  # 检索质量评估
    GENERATE = "generation"  # 生成
    GENERATE_QUALITY = "generation_quality_eval"  # 生成质量评估
    DECISION = "decision"  # 决策
    WEB_SEARCH = "web_search"  # Web搜索
    RERANK = "rerank"  # 重排序
    TOTAL = "total"  # 总耗时


# ==================== 核心追踪器 ====================

class TimingTracker:
    """统一的耗时追踪器

    使用方式：
    1. 上下文管理器：with tracker.track(Stage.INTENT): ...
    2. 装饰器：@tracker.timed(Stage.RETRIEVE)
    3. 从State恢复：tracker = TimingTracker.from_state(state)
    """

    def __init__(self, enabled: bool = True, verbose: bool = True):
        """
        Args:
            enabled: 是否启用计时（生产环境可关闭）
            verbose: 是否打印耗时信息
        """
        self.enabled = enabled
        self.verbose = verbose
        self.records: List[TimingRecord] = []
        self._start_time: Optional[float] = None

    def start_session(self):
        """开始一个新的追踪会话（记录总耗时的起点）"""
        self._start_time = time.perf_counter()

    def end_session(self) -> Optional[float]:
        """结束追踪会话，返回总耗时（毫秒）"""
        if self._start_time is None:
            return None
        total_ms = (time.perf_counter() - self._start_time) * 1000
        self.records.append(TimingRecord(
            stage=Stage.TOTAL,
            duration_ms=total_ms,
            timestamp=self._start_time,
            metadata={"type": "session"}
        ))
        return total_ms

    @contextmanager
    def track(self, stage: str, **metadata):
        """上下文管理器方式计时

        Args:
            stage: 阶段名称（建议使用 Stage 常量）
            **metadata: 额外元数据（如 iteration, query 等）

        Example:
            with tracker.track(Stage.INTENT, query="用户问题"):
                result = classifier.classify(query)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration_ms = (end - start) * 1000

            record = TimingRecord(
                stage=stage,
                duration_ms=duration_ms,
                timestamp=start,
                metadata=metadata
            )
            self.records.append(record)

            if self.verbose:
                self._print_timing(stage, duration_ms, metadata)

    def timed(self, stage: str, **default_metadata):
        """装饰器方式计时

        Args:
            stage: 阶段名称
            **default_metadata: 默认元数据

        Example:
            @tracker.timed(Stage.RETRIEVE)
            def retrieve(query):
                ...
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.track(stage, **default_metadata):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def record(self, stage: str, duration_ms: float, **metadata):
        """手动记录一条计时（用于外部计时结果）"""
        if not self.enabled:
            return

        record = TimingRecord(
            stage=stage,
            duration_ms=duration_ms,
            timestamp=time.time(),
            metadata=metadata
        )
        self.records.append(record)

        if self.verbose:
            self._print_timing(stage, duration_ms, metadata)

    def get_stage_records(self, stage: str) -> List[TimingRecord]:
        """获取某个阶段的所有记录"""
        return [r for r in self.records if r.stage == stage]

    def get_stats(self, stage: str) -> Optional[TimingStats]:
        """获取某个阶段的统计信息"""
        durations = [r.duration_ms for r in self.records if r.stage == stage]
        if not durations:
            return None
        return TimingStats(
            stage=stage,
            total_ms=sum(durations),
            count=len(durations),
            avg_ms=sum(durations) / len(durations),
            min_ms=min(durations),
            max_ms=max(durations)
        )

    def get_all_stats(self) -> Dict[str, TimingStats]:
        """获取所有阶段的统计汇总"""
        stages = set(r.stage for r in self.records)
        return {stage: self.get_stats(stage) for stage in stages if self.get_stats(stage)}

    def get_total_time(self) -> float:
        """获取所有记录的总耗时（不含重叠）"""
        return sum(r.duration_ms for r in self.records if r.stage != Stage.TOTAL)

    def to_state_format(self) -> List[Dict[str, Any]]:
        """转换为可存储在 State 中的格式"""
        return [r.to_dict() for r in self.records]

    @classmethod
    def from_state(cls, timing_records: List[Dict[str, Any]],
                   enabled: bool = True, verbose: bool = True) -> "TimingTracker":
        """从 State 中恢复 TimingTracker"""
        tracker = cls(enabled=enabled, verbose=verbose)
        for record_dict in timing_records:
            tracker.records.append(TimingRecord(
                stage=record_dict.get("stage", "unknown"),
                duration_ms=record_dict.get("duration_ms", 0),
                timestamp=record_dict.get("timestamp", 0),
                metadata=record_dict.get("metadata", {})
            ))
        return tracker

    def format_summary(self) -> str:
        """格式化输出统计汇总"""
        if not self.records:
            return "No timing records."

        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("📊 耗时统计汇总")
        lines.append("=" * 60)

        # 按阶段分组统计
        stats = self.get_all_stats()

        # 排序：先按总耗时降序
        sorted_stats = sorted(
            stats.values(),
            key=lambda s: s.total_ms,
            reverse=True
        )

        # 计算总耗时
        total_ms = self.get_total_time()

        for stat in sorted_stats:
            if stat.stage == Stage.TOTAL:
                continue

            percentage = (stat.total_ms / total_ms * 100) if total_ms > 0 else 0
            stage_display = self._get_stage_display_name(stat.stage)

            lines.append(
                f"  {stage_display:<20} | "
                f"总计: {stat.total_ms:>8.2f}ms ({percentage:>5.1f}%) | "
                f"次数: {stat.count:>2} | "
                f"平均: {stat.avg_ms:>8.2f}ms"
            )

        lines.append("-" * 60)
        lines.append(f"  {'总计':<20} | {total_ms:>8.2f}ms")
        lines.append("=" * 60 + "\n")

        return "\n".join(lines)

    def print_summary(self):
        """打印统计汇总"""
        print(self.format_summary())

    def _print_timing(self, stage: str, duration_ms: float, metadata: Dict):
        """打印单次计时信息"""
        stage_display = self._get_stage_display_name(stage)
        color = self._get_stage_color(stage)

        meta_str = ""
        if metadata:
            meta_items = [f"{k}={v}" for k, v in metadata.items() if k != "type"]
            if meta_items:
                meta_str = f" ({', '.join(meta_items[:2])})"

        print(f"{color}⏱️  [{stage_display}] {duration_ms:.2f}ms{meta_str}{Style.RESET_ALL}")

    def _get_stage_display_name(self, stage: str) -> str:
        """获取阶段的显示名称"""
        display_names = {
            Stage.INTENT: "意图识别",
            Stage.RETRIEVE: "检索",
            Stage.RETRIEVE_QUALITY: "检索质量评估",
            Stage.GENERATE: "生成",
            Stage.GENERATE_QUALITY: "生成质量评估",
            Stage.DECISION: "决策",
            Stage.WEB_SEARCH: "Web搜索",
            Stage.RERANK: "重排序",
            Stage.TOTAL: "总计"
        }
        return display_names.get(stage, stage)

    def _get_stage_color(self, stage: str) -> str:
        """获取阶段的颜色"""
        colors = {
            Stage.INTENT: Fore.MAGENTA,
            Stage.RETRIEVE: Fore.BLUE,
            Stage.RETRIEVE_QUALITY: Fore.CYAN,
            Stage.GENERATE: Fore.GREEN,
            Stage.GENERATE_QUALITY: Fore.LIGHTGREEN_EX,
            Stage.DECISION: Fore.YELLOW,
            Stage.WEB_SEARCH: Fore.CYAN,
            Stage.RERANK: Fore.LIGHTBLUE_EX,
            Stage.TOTAL: Fore.WHITE
        }
        return colors.get(stage, Fore.WHITE)


# ==================== 便捷函数 ====================

# 全局默认追踪器（用于简单场景）
_default_tracker: Optional[TimingTracker] = None


def get_tracker() -> TimingTracker:
    """获取全局默认追踪器"""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = TimingTracker()
    return _default_tracker


def reset_tracker():
    """重置全局追踪器"""
    global _default_tracker
    _default_tracker = TimingTracker()


def create_tracker(enabled: bool = True, verbose: bool = True) -> TimingTracker:
    """创建新的追踪器实例"""
    return TimingTracker(enabled=enabled, verbose=verbose)


@contextmanager
def track(stage: str, tracker: Optional[TimingTracker] = None, **metadata):
    """便捷的计时上下文管理器

    Example:
        from agentic_rag.timing import track, Stage

        with track(Stage.INTENT, query="用户问题"):
            result = classifier.classify(query)
    """
    t = tracker or get_tracker()
    with t.track(stage, **metadata):
        yield
