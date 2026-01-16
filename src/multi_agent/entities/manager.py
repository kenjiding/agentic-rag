"""实体管理器 - 企业级实体合并和提取

职��：
1. 智能合并多个Agent的实体模型
2. 从统一的entities字典中提取特定Agent的实体
3. 提供实体转换和验证方法
"""

import logging
from typing import Dict, Any, List, Optional, Type, TypeVar, Union
from pydantic import ValidationError

from src.multi_agent.entities.base import (
    BaseAgentEntities,
    ProductAgentEntities,
    OrderAgentEntities,
    ConsultationAgentEntities,
    RAGAgentEntities,
    ChatAgentEntities,
    AgentType,
    get_entity_model_for_agent,
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseAgentEntities)


class EntityManager:
    """实体管理器 - 企业级实体管理

    提供实体合并、提取、转换等功能。
    """

    # ==================== 实体合并 ====================

    @staticmethod
    def merge_entities(*entity_models: BaseAgentEntities) -> Dict[str, Any]:
        """
        智能合并多个Agent的实体模型

        合并策略：
        1. 标量字段：后者覆盖前者（非None值）
        2. 列表字段：去重合并
        3. 布尔字段：逻辑或操作

        Args:
            *entity_models: 可变数量的实体模型

        Returns:
            合并后的实体字典
        """
        merged: Dict[str, Any] = {}

        for model in entity_models:
            model_dict = model.to_dict()
            for key, value in model_dict.items():
                if value is None:
                    continue

                if key not in merged:
                    # 第一次出现
                    merged[key] = value
                elif isinstance(value, list):
                    # 列表字段：去重合并
                    if isinstance(merged[key], list):
                        # 合并并去重
                        merged[key] = list(set(merged[key] + value))
                    else:
                        # 类型冲突，后者覆盖
                        merged[key] = value
                elif isinstance(value, bool):
                    # 布尔字段：逻辑或
                    if isinstance(merged[key], bool):
                        merged[key] = merged[key] or value
                    else:
                        merged[key] = value
                else:
                    # 标量字段：后者覆盖
                    merged[key] = value

        logger.debug(f"实体合并结果: {merged}")
        return merged

    @staticmethod
    def merge_into_state(
        state_entities: Dict[str, Any],
        *new_entities: BaseAgentEntities
    ) -> Dict[str, Any]:
        """
        将新的实体合并到state.entities中

        Args:
            state_entities: 当前的state.entities
            *new_entities: 要合并的新实体模型

        Returns:
            更新后的实体字典
        """
        # 从state.entities创建临时实体模型（用于复用merge逻辑）
        temp_models = []

        # 将state.entities转换为具体的实体模型（根据字段推断）
        if state_entities:
            inferred_model = EntityManager._infer_entity_model(state_entities)
            if inferred_model:
                temp_models.append(inferred_model)

        # 添加新的实体模型
        temp_models.extend(new_entities)

        # 合并所有实体
        merged = EntityManager.merge_entities(*temp_models)
        return merged

    @staticmethod
    def _infer_entity_model(entities: Dict[str, Any]) -> Optional[BaseAgentEntities]:
        """
        从字典推断最合适的实体模型类型

        根据字典中的字段推断应该使用哪个Agent的实体模型。

        Args:
            entities: 实体字典

        Returns:
            推断的实体模型实例
        """
        # 检查关键字段，推断模型类型
        if "order_id" in entities and entities["order_id"]:
            return OrderAgentEntities.model_validate(entities)

        if "product_ids" in entities and len(entities.get("product_ids", [])) >= 2:
            return ConsultationAgentEntities.model_validate(entities)

        if "product_id" in entities or "search_keyword" in entities:
            return ProductAgentEntities.model_validate(entities)

        # 默认使用产品模型（最常见的场景）
        try:
            return ProductAgentEntities.model_validate(entities)
        except ValidationError:
            # 如果验证失败，返回None
            return None

    # ==================== 实体提取 ====================

    @staticmethod
    def extract_entities_for_agent(
        all_entities: Dict[str, Any],
        agent_name: str
    ) -> BaseAgentEntities:
        """
        从统一的entities字典中提取特定Agent相关的实体

        Args:
            all_entities: 所有的实体字典（state.entities）
            agent_name: 目标Agent名称

        Returns:
            特定Agent的实体模型实例
        """
        # 获取目标Agent的实体模型类
        entity_model_class = get_entity_model_for_agent(agent_name)

        try:
            # 尝试创建实体模型实例
            return entity_model_class.model_validate(all_entities)
        except ValidationError as e:
            logger.warning(f"实体验证失败（Agent: {agent_name}）: {e}")
            # 返回空实例
            return entity_model_class()

    @staticmethod
    def extract_product_entities(all_entities: Dict[str, Any]) -> ProductAgentEntities:
        """提取产品Agent的实体"""
        return EntityManager.extract_entities_for_agent(all_entities, "product_agent")

    @staticmethod
    def extract_order_entities(all_entities: Dict[str, Any]) -> OrderAgentEntities:
        """提取订单Agent的实体"""
        return EntityManager.extract_entities_for_agent(all_entities, "order_agent")

    @staticmethod
    def extract_consultation_entities(all_entities: Dict[str, Any]) -> ConsultationAgentEntities:
        """提取咨询Agent的实体"""
        return EntityManager.extract_entities_for_agent(all_entities, "consultation_agent")

    # ==================== 实体验证 ====================

    @staticmethod
    def validate_entities_for_agent(
        all_entities: Dict[str, Any],
        agent_name: str
    ) -> bool:
        """
        验证实体是否满足特定Agent的要求

        Args:
            all_entities: 所有的实体字典
            agent_name: 目标Agent名称

        Returns:
            是否验证通过
        """
        entity_model = EntityManager.extract_entities_for_agent(
            all_entities,
            agent_name
        )

        # 检查关键实体是否存在
        if agent_name == "product_agent":
            # 产品Agent需要product_id或search_keyword
            return bool(
                entity_model.product_id or
                entity_model.search_keyword or
                entity_model.product_ids
            )

        elif agent_name == "order_agent":
            # 订单Agent需要order_id或product_id（创建订单）
            return bool(
                entity_model.order_id or
                entity_model.product_id
            )

        elif agent_name == "consultation_agent":
            # 咨询Agent需要至少2个product_ids
            return len(entity_model.product_ids) >= 2

        return True

    # ==================== 实体差异对比 ====================

    @staticmethod
    def compare_entities(
        old_entities: Dict[str, Any],
        new_entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        对比两个实体字典，返回差异

        Args:
            old_entities: 旧实体字典
            new_entities: 新实体字典

        Returns:
            差异字典，包含新增和修改的字段
        """
        differences: Dict[str, Any] = {}

        all_keys = set(old_entities.keys()) | set(new_entities.keys())

        for key in all_keys:
            old_val = old_entities.get(key)
            new_val = new_entities.get(key)

            if old_val != new_val:
                differences[key] = {
                    "old": old_val,
                    "new": new_val
                }

        return differences

    # ==================== 实体摘要 ====================

    @staticmethod
    def summarize_entities(all_entities: Dict[str, Any]) -> str:
        """
        生成实体摘要（用于日志和调试）

        Args:
            all_entities: 实体字典

        Returns:
            摘要字符串
        """
        if not all_entities:
            return "（无实体信息）"

        summary_parts = []

        for key, value in all_entities.items():
            if value is None:
                continue

            if isinstance(value, list):
                if value:
                    summary_parts.append(f"{key}: {value}")
            elif isinstance(value, bool):
                summary_parts.append(f"{key}: {value}")
            else:
                summary_parts.append(f"{key}: {value}")

        return ", ".join(summary_parts) if summary_parts else "（无实体信息）"
