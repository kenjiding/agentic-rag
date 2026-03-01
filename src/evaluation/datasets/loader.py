"""数据集加载器

从文件加载评测数据集。

设计原则：
- 支持JSON格式
- 支持配置驱动的数据集管理
- 提供数据验证
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging

from src.evaluation.models import EvaluationCase, Milestone, ExpectedOutcome, OutcomeType
from src.evaluation.config import EvaluationConfig, DatasetConfig

logger = logging.getLogger(__name__)


class DatasetLoader:
    """数据集加载器
    
    从文件加载评测数据集，支持：
    - JSON格式
    - 批量加载
    - 数据验证
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """初始化数据集加载器
        
        Args:
            base_path: 数据集基础路径
            config: 评测配置
        """
        self.config = config or EvaluationConfig.default()
        self.base_path = Path(base_path) if base_path else Path(".")
    
    def load(self, path: str) -> List[EvaluationCase]:
        """加载单个数据集文件
        
        Args:
            path: 文件路径（相对于base_path或绝对路径）
            
        Returns:
            评测用例列表
        """
        file_path = self._resolve_path(path)
        
        if not file_path.exists():
            logger.warning(f"数据集文件不存在: {file_path}")
            return []
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        cases = self._parse_cases(data)
        logger.info(f"加载数据集: {file_path}，共 {len(cases)} 个用例")
        
        return cases
    
    def load_from_config(self) -> Dict[str, List[EvaluationCase]]:
        """从配置加载所有启用的数据集
        
        Returns:
            数据集字典 {name: cases}
        """
        datasets = {}
        
        for dataset_config in self.config.get_enabled_datasets():
            cases = self.load(dataset_config.path)
            if cases:
                datasets[dataset_config.name] = cases
        
        return datasets
    
    def load_all(self) -> List[EvaluationCase]:
        """加载所有启用的数据集并合并
        
        Returns:
            所有用例列表
        """
        all_cases = []
        datasets = self.load_from_config()
        
        for name, cases in datasets.items():
            all_cases.extend(cases)
        
        return all_cases
    
    def _resolve_path(self, path: str) -> Path:
        """解析文件路径"""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return self.base_path / path_obj
    
    def _parse_cases(self, data: Any) -> List[EvaluationCase]:
        """解析JSON数据为评测用例
        
        Args:
            data: JSON数据（列表或字典）
            
        Returns:
            评测用例列表
        """
        # 支持两种格式：
        # 1. 直接是用例列表
        # 2. {"cases": [...], "metadata": {...}}
        
        if isinstance(data, list):
            raw_cases = data
        elif isinstance(data, dict):
            raw_cases = data.get("cases", data.get("test_cases", []))
        else:
            logger.error(f"无效的数据集格式: {type(data)}")
            return []
        
        cases = []
        for i, raw in enumerate(raw_cases):
            try:
                case = self._parse_single_case(raw, index=i)
                if case:
                    cases.append(case)
            except Exception as e:
                logger.warning(f"解析用例 {i} 失败: {e}")
        
        return cases
    
    def _parse_single_case(self, raw: Dict[str, Any], index: int) -> Optional[EvaluationCase]:
        """解析单个用例
        
        Args:
            raw: 原始用例数据
            index: 用例索引
            
        Returns:
            评测用例
        """
        # 必需字段
        case_id = raw.get("case_id") or raw.get("id") or f"case_{index}"
        
        # 输入消息
        input_messages = raw.get("input_messages") or raw.get("messages") or []
        if isinstance(input_messages, str):
            input_messages = [input_messages]
        
        if not input_messages:
            input_msg = raw.get("input") or raw.get("query") or raw.get("message")
            if input_msg:
                input_messages = [input_msg]
        
        if not input_messages:
            logger.warning(f"用例 {case_id} 缺少输入消息")
            return None
        
        # 解析预期结果
        expected_outcomes = self._parse_expected_outcomes(raw)
        
        # 解析里程碑
        milestones = self._parse_milestones(raw)
        
        return EvaluationCase(
            case_id=case_id,
            name=raw.get("name", ""),
            description=raw.get("description", ""),
            input_messages=input_messages,
            initial_state=raw.get("initial_state"),
            expected_intent=raw.get("expected_intent"),
            expected_agent=raw.get("expected_agent"),
            expected_tool_calls=raw.get("expected_tool_calls", []),
            expected_outcomes=expected_outcomes,
            milestones=milestones,
            applicable_policies=raw.get("applicable_policies", []),
            tags=raw.get("tags", []),
            timeout_seconds=raw.get("timeout_seconds", 60),
            priority=raw.get("priority", 0),
        )
    
    def _parse_expected_outcomes(self, raw: Dict[str, Any]) -> List[ExpectedOutcome]:
        """解析预期结果"""
        outcomes = []
        
        # 从 expected_outcomes 字段解析
        for outcome_data in raw.get("expected_outcomes", []):
            try:
                outcome_type = OutcomeType(outcome_data.get("type", "state_match"))
                outcomes.append(ExpectedOutcome(
                    outcome_type=outcome_type,
                    expected_value=outcome_data.get("value"),
                    tolerance=outcome_data.get("tolerance", 0.0),
                    weight=outcome_data.get("weight", 1.0),
                ))
            except Exception as e:
                logger.warning(f"解析预期结果失败: {e}")
        
        # 从简化字段解析
        if "expected_response_contains" in raw:
            outcomes.append(ExpectedOutcome(
                outcome_type=OutcomeType.RESPONSE_CONTAINS,
                expected_value=raw["expected_response_contains"],
            ))
        
        if "expected_state" in raw:
            outcomes.append(ExpectedOutcome(
                outcome_type=OutcomeType.STATE_MATCH,
                expected_value=raw["expected_state"],
            ))
        
        return outcomes
    
    def _parse_milestones(self, raw: Dict[str, Any]) -> List[Milestone]:
        """解析里程碑"""
        milestones = []
        
        for ms_data in raw.get("milestones", []):
            try:
                milestones.append(Milestone(
                    milestone_id=ms_data.get("id", ""),
                    name=ms_data.get("name", ""),
                    description=ms_data.get("description", ""),
                    required=ms_data.get("required", True),
                    weight=ms_data.get("weight", 1.0),
                    condition_type=ms_data.get("condition_type", "event_exists"),
                    condition_value=ms_data.get("condition_value", {}),
                ))
            except Exception as e:
                logger.warning(f"解析里程碑失败: {e}")
        
        return milestones
