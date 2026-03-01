"""数据集生成器

生成合成评测数据集。

设计原则：
- 支持多种场景模板
- 支持参数化生成
- 支持批量生成
"""
from typing import List, Dict, Any, Optional
import random
import uuid
import json
from pathlib import Path

from src.evaluation.models import (
    EvaluationCase,
    Milestone,
    ExpectedOutcome,
    OutcomeType,
)


class DatasetGenerator:
    """数据集生成器
    
    生成合成评测数据集，支持：
    - 预定义场景模板
    - 参数化生成
    - 批量导出
    """
    
    def __init__(self):
        """初始化数据集生成器"""
        self._templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载场景模板"""
        return {
            # 基础对话场景
            "greeting": {
                "input_patterns": [
                    "你好",
                    "早上好",
                    "晚上好",
                    "Hi",
                    "Hello",
                ],
                "expected_intent": "GREETING",
                "expected_agent": "chat_agent",
                "tags": ["basic", "greeting"],
            },
            
            # 商品搜索场景
            "product_search": {
                "input_patterns": [
                    "帮我搜索{brand}的{category}",
                    "我想买{brand}的产品",
                    "有没有{category}推荐",
                    "搜索{brand}{category}",
                ],
                "params": {
                    "brand": ["西门子", "海尔", "美的", "格力", "松下"],
                    "category": ["冰箱", "洗衣机", "空调", "电视", "微波炉"],
                },
                "expected_intent": "PRODUCT_SEARCH",
                "expected_agent": "product_agent",
                "expected_tool_calls": ["search_products"],
                "tags": ["product", "search"],
            },
            
            # 订单查询场景
            "order_query": {
                "input_patterns": [
                    "查询我的订单",
                    "我的订单状态是什么",
                    "订单{order_id}的物流信息",
                    "最近的订单",
                ],
                "params": {
                    "order_id": ["ORD001", "ORD002", "ORD003"],
                },
                "expected_intent": "ORDER_QUERY",
                "expected_agent": "order_agent",
                "expected_tool_calls": ["query_orders"],
                "tags": ["order", "query"],
            },
            
            # 订单创建场景
            "order_create": {
                "input_patterns": [
                    "帮我购买{quantity}个{product}",
                    "下单{product}，数量{quantity}",
                    "我要买{product}",
                ],
                "params": {
                    "product": ["西门子冰箱", "海尔洗衣机", "美的空调"],
                    "quantity": ["1", "2", "3"],
                },
                "expected_intent": "ORDER_CREATE",
                "expected_agent": "order_agent",
                "expected_tool_calls": ["create_order"],
                "tags": ["order", "create", "high_risk"],
                "applicable_policies": ["no_unauthorized_orders", "confirmation_required"],
            },
            
            # 知识问答场景
            "knowledge_qa": {
                "input_patterns": [
                    "{product}的保修政策是什么",
                    "如何使用{product}",
                    "{product}的售后服务",
                    "{product}的安装方法",
                ],
                "params": {
                    "product": ["冰箱", "洗衣机", "空调", "电视"],
                },
                "expected_intent": "KNOWLEDGE_QA",
                "expected_agent": "rag_agent",
                "tags": ["knowledge", "qa"],
            },
            
            # 产品对比场景
            "product_comparison": {
                "input_patterns": [
                    "对比{product1}和{product2}",
                    "{product1}和{product2}哪个好",
                    "帮我比较这两个产品",
                ],
                "params": {
                    "product1": ["西门子冰箱", "海尔洗衣机"],
                    "product2": ["松下冰箱", "美的洗衣机"],
                },
                "expected_intent": "PRODUCT_COMPARISON",
                "expected_agent": "consultation_agent",
                "tags": ["comparison", "consultation"],
            },
        }
    
    def generate(
        self,
        scenario: str,
        count: int = 1,
        randomize: bool = True
    ) -> List[EvaluationCase]:
        """生成指定场景的评测用例
        
        Args:
            scenario: 场景名称
            count: 生成数量
            randomize: 是否随机化参数
            
        Returns:
            评测用例列表
        """
        if scenario not in self._templates:
            raise ValueError(f"未知场景: {scenario}")
        
        template = self._templates[scenario]
        cases = []
        
        for i in range(count):
            case = self._generate_from_template(template, i, randomize)
            cases.append(case)
        
        return cases
    
    def generate_all(self, count_per_scenario: int = 5) -> List[EvaluationCase]:
        """生成所有场景的评测用例
        
        Args:
            count_per_scenario: 每个场景的数量
            
        Returns:
            所有评测用例
        """
        all_cases = []
        
        for scenario in self._templates:
            cases = self.generate(scenario, count_per_scenario)
            all_cases.extend(cases)
        
        return all_cases
    
    def _generate_from_template(
        self,
        template: Dict[str, Any],
        index: int,
        randomize: bool
    ) -> EvaluationCase:
        """从模板生成用例"""
        # 选择输入模式
        patterns = template["input_patterns"]
        pattern = random.choice(patterns) if randomize else patterns[index % len(patterns)]
        
        # 填充参数
        params = template.get("params", {})
        filled_pattern = pattern
        
        for param_name, param_values in params.items():
            if f"{{{param_name}}}" in filled_pattern:
                value = random.choice(param_values) if randomize else param_values[0]
                filled_pattern = filled_pattern.replace(f"{{{param_name}}}", value)
        
        # 生成用例ID
        case_id = f"{template.get('expected_intent', 'case').lower()}_{uuid.uuid4().hex[:8]}"
        
        return EvaluationCase(
            case_id=case_id,
            name=f"{template.get('expected_intent', 'Test')} Case",
            description=f"自动生成的评测用例: {filled_pattern}",
            input_messages=[filled_pattern],
            expected_intent=template.get("expected_intent"),
            expected_agent=template.get("expected_agent"),
            expected_tool_calls=template.get("expected_tool_calls", []),
            applicable_policies=template.get("applicable_policies", []),
            tags=template.get("tags", []),
        )
    
    def export_to_json(
        self,
        cases: List[EvaluationCase],
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """导出用例到JSON文件
        
        Args:
            cases: 评测用例列表
            path: 输出文件路径
            metadata: 额外的元数据
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": metadata or {
                "generated": True,
                "count": len(cases),
            },
            "cases": [case.model_dump() for case in cases],
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def add_template(self, name: str, template: Dict[str, Any]) -> None:
        """添加自定义模板
        
        Args:
            name: 模板名称
            template: 模板配置
        """
        self._templates[name] = template
