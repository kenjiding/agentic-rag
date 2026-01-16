"""Context-Aware Entity Cleaner - 对话阶段感知的实体清理器

企业级实现：基于对话阶段智能清理过时实体，避免entities残留干扰路由决策。

问题场景：
1. 用户创建订单后，order_id残留，导致后续"谢谢"仍被误判为订单相关
2. 用户搜索产品后，search_keyword残留，导致后续感谢被误判为产品搜索

解决方案：
- 基于对话阶段转换规则，智能清理过时实体
- 保留关键实体，清理过时实体
- 对话阶段感知的生命周期管理
"""
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ContextAwareEntityCleaner:
    """对话感知的实体清理器

    核心理念：
    - entities在多轮对话中持续累积，可能包含过时信息
    - 不同对话阶段需要不同的实体集
    - 对话阶段转换时，应清理过时实体

    设计原则：
    - 明确定义阶段转换规则
    - 保留关键实体（如product_id）
    - 清理过时实体（如已完成的订单ID）
    """

    # 对话阶段转换规则
    # 格式: (current_phase, next_phase) -> {"clear_entities": [...], "preserve_entities": [...], "reason": "..."}
    PHASE_TRANSITION_RULES: Dict[Tuple[str, str], Dict[str, Any]] = {
        # 规则1: 订单完成 → 空闲：清理订单和产品相关实体
        ("order_completed", "idle"): {
            "clear_entities": ["order_id", "order_number", "product_id", "product_ids", "search_keyword"],
            "preserve_entities": [],
            "reason": "订单完成，进入空闲状态，清理所有订单和产品相关实体"
        },

        # 规则2: 产品选择 → 订单创建：清理搜索关键词，保���选定产品
        ("product_selecting", "order_creating"): {
            "clear_entities": ["search_keyword"],
            "preserve_entities": ["product_id", "product_ids"],
            "reason": "进入订单创建阶段，保留选定产品，清理搜索关键词"
        },

        # 规则3: 订单创建 → 订单完成：清理产品相关实体，保留订单信息
        ("order_creating", "order_completed"): {
            "clear_entities": ["product_ids", "search_keyword"],
            "preserve_entities": ["order_id", "order_number", "product_id"],
            "reason": "订单创建完成，清理产品搜索相关实体，保留订单和选定产品"
        },

        # 规则4: 空闲 → 产品选择：清理之前的订单和产品信息
        ("idle", "product_selecting"): {
            "clear_entities": ["order_id", "order_number", "product_id", "product_ids"],
            "preserve_entities": [],
            "reason": "开始新的产品选择流程，清理之前的订单和产品信息"
        },

        # 规则5: 订单完成 → 产品选择：清理订单信息，开始新的产品选择
        ("order_completed", "product_selecting"): {
            "clear_entities": ["order_id", "order_number", "product_id"],
            "preserve_entities": [],
            "reason": "订单完成后开始新的产品选择，清理订单信息"
        },

        # 规则6: 产品选择 → 空闲：用户放弃选择，清理所有产品相关实体
        ("product_selecting", "idle"): {
            "clear_entities": ["product_id", "product_ids", "search_keyword"],
            "preserve_entities": [],
            "reason": "用户放弃产品选择，清理所有产品相关实体"
        },
    }

    @classmethod
    def clean_entities_on_phase_change(
        cls,
        current_phase: str,
        next_phase: str,
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        基于对话阶段转换清理实体

        Args:
            current_phase: 当前对话阶段
            next_phase: 下一个对话阶段
            entities: 当前实体字典

        Returns:
            清理后的实体字典
        """
        transition_key = (current_phase, next_phase)
        rule = cls.PHASE_TRANSITION_RULES.get(transition_key)

        # 如果没有特定规则，保留所有实体
        if not rule:
            logger.debug(
                f"[EntityCleaner] 无特定清理规则: {current_phase} → {next_phase}，保留所有实体"
            )
            return entities.copy()

        cleaned = entities.copy()
        clear_entities = rule.get("clear_entities", [])
        preserve_entities = rule.get("preserve_entities", [])
        reason = rule.get("reason", "阶段转换清理")

        # 如果preserve_entities为空列表，则只清理clear_entities中的实体
        # 如果preserve_entities有值，则只保留这些实体，清理其他所有实体
        if preserve_entities:
            # 模式1: 白名单模式 - 只保留preserve_entities中的实体
            keys_to_clear = [k for k in cleaned.keys() if k not in preserve_entities]
            for key in keys_to_clear:
                if key in cleaned:
                    del cleaned[key]
                    logger.info(f"[EntityCleaner] 清理实体: {key}（原因: {reason}）")
        else:
            # 模式2: 黑名单模式 - 只清理clear_entities中的实体
            for key in clear_entities:
                if key in cleaned:
                    del cleaned[key]
                    logger.info(f"[EntityCleaner] 清理实体: {key}（原因: {reason}）")

        logger.info(
            f"[EntityCleaner] 阶段转换: {current_phase} → {next_phase} | "
            f"清理后实体: {list(cleaned.keys())}"
        )

        return cleaned

    @classmethod
    def should_clear_entity(cls, entity_key: str, current_phase: str, next_phase: str) -> bool:
        """
        判断特定实体是否应该在阶段转换时清理

        Args:
            entity_key: 实体键名
            current_phase: 当前对话阶段
            next_phase: 下一个对话阶段

        Returns:
            bool: 是否应该清理该实体
        """
        transition_key = (current_phase, next_phase)
        rule = cls.PHASE_TRANSITION_RULES.get(transition_key)

        if not rule:
            return False

        preserve_entities = rule.get("preserve_entities", [])

        # 白名单模式：如果preserve_entities有值，只有白名单中的实体保留
        if preserve_entities:
            return entity_key not in preserve_entities

        # 黑名单模式：如果preserve_entities为空，只清理clear_entities中的实体
        clear_entities = rule.get("clear_entities", [])
        return entity_key in clear_entities

    @classmethod
    def get_all_transition_rules(cls) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        获取所有阶段转换规则（用于调试和文档）

        Returns:
            所有阶段转换规则
        """
        return cls.PHASE_TRANSITION_RULES.copy()

    @classmethod
    def infer_next_phase(cls, current_phase: str, next_action: str) -> str:
        """
        基于当前对话阶段和next_action推断下一个对话阶段

        Args:
            current_phase: 当前对话阶段
            next_action: 下一步行动（如product_search, order_management等）

        Returns:
            推断的下一个对话阶段
        """
        # 规则1: 如果当前是空闲状态
        if current_phase == "idle":
            if next_action == "product_search":
                return "product_selecting"
            elif next_action == "order_management":
                return "order_creating"

        # 规则2: 如果当前是产品选择阶段
        elif current_phase == "product_selecting":
            if next_action == "order_management":
                return "order_creating"
            elif next_action == "finish":
                return "idle"

        # 规则3: 如果当前是订单创建阶段
        elif current_phase == "order_creating":
            if next_action == "finish":
                return "order_completed"

        # 规则4: 如果当前是订单完成阶段
        elif current_phase == "order_completed":
            if next_action == "product_search":
                return "product_selecting"
            elif next_action == "finish":
                return "idle"

        # 默认保持当前阶段
        logger.warning(
            f"[EntityCleaner] 无法推断阶段转换: {current_phase} + {next_action}，保持当前阶段"
        )
        return current_phase
