"""评测运行器模块

提供评测执行能力：
- SingleRunner: 单次评测运行
- BatchRunner: 批量评测运行（支持并行）
"""
from src.evaluation.runners.single_run import SingleRunner
from src.evaluation.runners.batch_runner import BatchRunner

__all__ = [
    "SingleRunner",
    "BatchRunner",
]
