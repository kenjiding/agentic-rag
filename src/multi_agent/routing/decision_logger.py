"""Routing Decision Logger - 路由决策日志格式化工具

提供统一的路由决策日志格式，增强可观察性和可调试性。
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RoutingDecisionLogger:
    """路由决策日志格式化器

    职责：
    - 统一路由决策的日志格式
    - 提供结构化的决策信息输出
    - 支持路由决策的追踪和调试

    设计原则：
    - 日志信息清晰、简洁、完整
    - 包含关键决策因素（business_intent, entity_signals等）
    - 便于问题排查和性能分析
    """

    @staticmethod
    def format_decision(decision: Dict[str, Any]) -> str:
        """
        格式化路由决策用于日志输出

        Args:
            decision: 路由决策字典，包含：
                - next_action: 下一步行动
                - selected_agent: 选中的Agent
                - routing_reason: 路由原因
                - confidence: 置信度
                - business_intent: 业务意图��可选）
                - routing_method: 路由方法（可选）
                - entity_signals: 实体信号（可选）

        Returns:
            格式化的日志字符串
        """
        lines = [
            "═══════════════════════════════════════",
            f"【路由决策】{decision.get('next_action', 'unknown')} → {decision.get('selected_agent', 'none')}",
            f"  方法: {decision.get('routing_method', 'unknown')}",
            f"  置信度: {decision.get('confidence', 0):.2f}",
            f"  原因: {decision.get('routing_reason', 'no reason')}",
        ]

        if decision.get('business_intent'):
            lines.append(f"  业务意图: {decision['business_intent']}")

        if decision.get('entity_signals'):
            signals = decision['entity_signals']
            if isinstance(signals, list):
                lines.append(f"  实体信号: {', '.join(signals)}")
            else:
                lines.append(f"  实体信号: {signals}")

        lines.append("═══════════════════════════════════════")
        return "\n".join(lines)

    @staticmethod
    def log_decision(decision: Dict[str, Any], level: int = logging.INFO):
        """
        记录路由决策到日志

        Args:
            decision: 路由决策字典
            level: 日志级别（默认INFO）
        """
        formatted = RoutingDecisionLogger.format_decision(decision)
        logger.log(level, formatted)

    @staticmethod
    def format_decision_compact(decision: Dict[str, Any]) -> str:
        """
        紧凑格式化路由决策（用于单行日志）

        Args:
            decision: 路由决策字典

        Returns:
            紧凑格式的日志字符串
        """
        action = decision.get('next_action', 'unknown')
        agent = decision.get('selected_agent', 'none')
        method = decision.get('routing_method', 'unknown')
        confidence = decision.get('confidence', 0)
        reason = decision.get('routing_reason', '')

        return (
            f"Route: {action}→{agent} | "
            f"Method: {method} | "
            f"Conf: {confidence:.2f} | "
            f"Reason: {reason}"
        )

    @staticmethod
    def create_decision_record(
        next_action: str,
        selected_agent: Optional[str],
        routing_reason: str,
        confidence: float,
        routing_method: str,
        business_intent: Optional[str] = None,
        entity_signals: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建标准化的路由决策记录

        Args:
            next_action: 下一步行动
            selected_agent: 选中的Agent名称
            routing_reason: 路由原因
            confidence: 置信度（0-1）
            routing_method: 路由方法（如llm_reasoning_parsing, deterministic_rule等）
            business_intent: 业务意图类型（可选）
            entity_signals: 触发路由的实体信号（可选）
            **kwargs: 其他自定义字段

        Returns:
            标准化的路由决策字典
        """
        record = {
            "next_action": next_action,
            "selected_agent": selected_agent,
            "routing_reason": routing_reason,
            "confidence": confidence,
            "routing_method": routing_method,
            "timestamp": datetime.now().isoformat(),
        }

        if business_intent:
            record["business_intent"] = business_intent

        if entity_signals:
            record["entity_signals"] = entity_signals

        # 添加自定义字段
        record.update(kwargs)

        return record

    @staticmethod
    def compare_decisions(
        old_decision: Optional[Dict[str, Any]],
        new_decision: Dict[str, Any]
    ) -> Optional[str]:
        """
        比较两个路由决策，返回变更信息

        Args:
            old_decision: 旧的路由决策
            new_decision: 新的路由决策

        Returns:
            变更信息字符串，如果没有变更返回None
        """
        if not old_decision:
            return f"新路由决策: {new_decision.get('next_action')} → {new_decision.get('selected_agent')}"

        changes = []

        # 比较next_action
        if old_decision.get('next_action') != new_decision.get('next_action'):
            changes.append(
                f"action: {old_decision.get('next_action')} → {new_decision.get('next_action')}"
            )

        # 比较selected_agent
        if old_decision.get('selected_agent') != new_decision.get('selected_agent'):
            changes.append(
                f"agent: {old_decision.get('selected_agent')} → {new_decision.get('selected_agent')}"
            )

        # 比较confidence
        old_conf = old_decision.get('confidence', 0)
        new_conf = new_decision.get('confidence', 0)
        if abs(old_conf - new_conf) > 0.1:
            changes.append(f"confidence: {old_conf:.2f} → {new_conf:.2f}")

        # 比较routing_method
        if old_decision.get('routing_method') != new_decision.get('routing_method'):
            changes.append(
                f"method: {old_decision.get('routing_method')} → {new_decision.get('routing_method')}"
            )

        if not changes:
            return None

        return "路由决策变更: " + ", ".join(changes)
