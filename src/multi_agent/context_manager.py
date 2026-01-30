"""Context Manager - 企业级上下文管理模块

这个模块负责统一管理多Agent系统的上下文信息提取和压缩，
符合LangGraph 1.x最佳实践，使用state作为单一数据源。

核心职责：
1. 从messages中智能提取和压缩对话历史（最近N轮）
2. 压缩tool results，只保留关键信息
3. 提取关键实体状态（product_id, order_id等）
4. 生成结构化的context_summary

设计原则：
- 单一职责：只负责上下文提取和压缩
- 统一管理：所有上下文处理逻辑集中在一处
- 可扩展性：新增上下文类型只需修改此处
- 性能优化：使用规则提取，避免额外LLM调用
"""

import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field

# TYPE_CHECKING用于避免循环导入
if TYPE_CHECKING:
    from src.multi_agent.state import MultiAgentState

logger = logging.getLogger(__name__)


class ContextSummary(BaseModel):
    """上下文摘要 - 结构化的上下文信息

    这是ContextManager的输出，包含意图识别和路由决策所需的所有上下文信息。
    """
    # 当前查询信息
    current_query: str = Field(description="当前用户查询")

    # 对话历史摘要（最近N轮，已压缩tool results）
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="对话历史摘要，每轮包含用户问题、AI回复、工具调用"
    )

    # 关键实体状态（从entities中提取）
    key_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="关键实体状态（product_id, order_id, product_ids等）"
    )

    # 对话阶段
    conversation_phase: str = Field(description="当前对话阶段")

    # 最近工具调用摘要（压缩后的tool results）
    recent_tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="最近工具调用的摘要信息"
    )

    # 上下文元数据
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="上下文处理的元数据（如使用的策略、压缩比例等）"
    )


class ContextManager:
    """上下文管理器 - 企业级上下文处理

    职责：
    1. 从messages中智能提取和压缩上下文
    2. 支持增量更新（避免每次重新处理全部历史）
    3. 提供结构化的上下文摘要
    4. 支持多种压缩策略

    使用方式：
        context_manager = ContextManager(max_history_rounds=5)
        summary = await context_manager.build_context_summary(state, current_query)
    """

    def __init__(
        self,
        max_history_rounds: int = 10,
        max_tool_calls: int = 10
    ):
        """
        初始化上下文管理器

        Args:
            max_history_rounds: 保留最近N轮对话
            max_tool_calls: 保留最近N个工具调用
        """
        self.max_history_rounds = max_history_rounds
        self.max_tool_calls = max_tool_calls

    async def build_context_summary(
        self,
        state: 'MultiAgentState',
        current_query: str
    ) -> ContextSummary:
        """
        构建上下文摘要

        Args:
            state: 当前多Agent状态
            current_query: 当前用户查询

        Returns:
            ContextSummary: 结构化上下文摘要
        """
        try:
            # 1. 提取对话历史（按对话组分组）
            conversation_history = self._extract_conversation_history(
                state.messages,
                max_rounds=self.max_history_rounds
            )

            # 2. 提取关键实体（从state.entities）
            key_entities = self._extract_key_entities(state.entities)

            # 3. 提取最近工具调用摘要（压缩tool results）
            recent_tool_calls = self._extract_recent_tool_calls(
                state.messages,
                max_calls=self.max_tool_calls
            )

            # 4. 构建上下文摘要
            context_summary = ContextSummary(
                current_query=current_query,
                conversation_history=conversation_history,
                key_entities=key_entities,
                conversation_phase=state.conversation_phase,
                recent_tool_calls=recent_tool_calls,
                metadata={
                    "total_messages": len(state.messages),
                    "history_rounds": len(conversation_history),
                    "tool_calls_count": len(recent_tool_calls),
                    "extraction_strategy": "rule_based"
                }
            )

            logger.info(
                f"📊【上下文管理】构建摘要: "
                f"{len(conversation_history)}轮对话, "
                f"{len(recent_tool_calls)}个工具调用, "
                f"{len(key_entities)}个关键实体"
            )

            return context_summary

        except Exception as e:
            logger.error(f"构建上下文摘要失败: {e}", exc_info=True)
            # 返回最小化的上下文摘要
            return ContextSummary(
                current_query=current_query,
                conversation_history=[],
                key_entities={},
                conversation_phase=state.conversation_phase or "idle",
                recent_tool_calls=[],
                metadata={"error": str(e)}
            )

    def _extract_conversation_history(
        self,
        messages: List[BaseMessage],
        max_rounds: int = 10
    ) -> List[Dict[str, Any]]:
        """
        提取对话历史（按对话组分组）

        分组原则：
        - 尽量把连续的对话放在同一个组
        - 只有在「完成了一轮工具调用」（AI调用过工具且所有ToolMessage都回来了）之后，
          再遇到新的HumanMessage，才认为开启了新话题，开始新组

        Args:
            messages: 消息列表
            max_rounds: 最大保留轮数

        Returns:
            对话历史列表，每轮包含human、ai、tool_calls
        """
        if not messages:
            return []

        groups = []
        current = None
        last_round_has_completed_tool = False

        for msg in messages:
            if isinstance(msg, HumanMessage):
                # 关键判断：只有上一轮工具调用完整结束，才切新组
                if current is not None and last_round_has_completed_tool:
                    groups.append(current)
                    current = None
                    last_round_has_completed_tool = False

                if current is None:
                    current = {
                        "human": "",
                        "ai": "",
                        "tool_calls": []
                    }

                current["human"] = msg.content

            elif isinstance(msg, AIMessage) and current is not None:
                tool_calls = getattr(msg, "tool_calls", []) or []
                current["ai"] = msg.content or ""

                # 记录工具调用
                for tc in tool_calls:
                    current["tool_calls"].append({
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {})
                    })

                # 如果这次AI没有调用工具，也算一种"完成"
                if not tool_calls:
                    last_round_has_completed_tool = True

        # 添加最后一组
        if current is not None:
            groups.append(current)

        # 返回最近N轮
        if len(groups) > max_rounds:
            return groups[-max_rounds:]
        return groups

    def _extract_key_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取关键实体（使用EntityManager）

        根据路由决策的需求，提取各Agent的关键实体。
        使用类型安全的实体模型，而不是硬编码字段名。

        Args:
            entities: 完整的实体字典

        Returns:
            关键实体字典
        """
        if not entities:
            return {}

        # 【改进】使用EntityManager进行智能提取
        # 从所有实体模型中提取路由决策需要的字段
        from src.multi_agent.entities import (
            ProductAgentEntities,
            OrderAgentEntities,
            ConsultationAgentEntities,
            EntityManager
        )

        try:
            # 提取各Agent的实体模型
            product_entities = EntityManager.extract_product_entities(entities)
            order_entities = EntityManager.extract_order_entities(entities)
            consultation_entities = EntityManager.extract_consultation_entities(entities)

            # 合并关键实体
            key_entities = EntityManager.merge_entities(
                product_entities,
                order_entities,
                consultation_entities
            )

            logger.debug(f"提取关键实体: {EntityManager.summarize_entities(key_entities)}")
            return key_entities

        except Exception as e:
            logger.warning(f"使用EntityManager提取实体失败，降级为简单提取: {e}")
            # 降级策略：提取常见的路由字段
            key_fields = ["product_id", "product_ids", "order_id", "search_keyword", "quantity"]
            return {k: v for k, v in entities.items() if k in key_fields and v is not None}

    def _extract_recent_tool_calls(
        self,
        messages: List[BaseMessage],
        max_calls: int = 10
    ) -> List[Dict[str, Any]]:
        """
        提取最近工具调用摘要

        Args:
            messages: 消息列表
            max_calls: 最大提取数量

        Returns:
            工具调用摘要列表
        """
        tool_calls = []

        # 遍历消息，提取ToolMessage
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_call_id = msg.tool_call_id

                # 找到对应的AI消息（获取工具名称）
                for ai_msg in messages:
                    if isinstance(ai_msg, AIMessage):
                        for tc in (ai_msg.tool_calls or []):
                            if tc.get("id") == tool_call_id:
                                # 压缩tool result
                                summary = self._compress_tool_result(
                                    tc.get("name", ""),
                                    msg.content
                                )

                                tool_calls.append({
                                    "name": tc.get("name", ""),
                                    "summary": summary
                                })

                                break

                if len(tool_calls) >= max_calls:
                    break

        # 反转回原来的顺序（从旧到新）
        return list(reversed(tool_calls))

    def _compress_tool_result(self, tool_name: str, result: Any) -> str:
        """
        压缩tool result，只保留关键信息（包括所有ID等重要数据）

        使用通用的递归提取方法，自动识别和提取所有重要字段：
        - 所有包含"id"的字段（id, product_id, order_id等）
        - 关键状态字段（status, state, number等）
        - 递归处理嵌套结构和列表

        不针对特定工具类型硬编码，适用于所有工具返回的数据结构。

        Args:
            tool_name: 工具名称（用于日志，不影响提取逻辑）
            result: 工具返回结果（可以是字符串、字典、列表等）

        Returns:
            压缩后的字符串（JSON格式，包含所有重要ID信息）
        """
        try:
            # 尝试解析JSON字符串
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    # 如果不是JSON，直接返回截取的文本
                    return result.strip()[:300]

            # 使用通用方法递归提取所有重要字段
            extracted = self._extract_ids_and_key_fields(result)
            
            if extracted:
                # 将提取的结果转换为JSON字符串
                result_str = json.dumps(extracted, ensure_ascii=False)
                # 限制长度，避免上下文过长
                if len(result_str) > 1000:
                    return result_str[:1000] + "..."
                return result_str
            else:
                # 如果没有提取到关键字段，返回原始数据的简化版本
                if isinstance(result, (dict, list)):
                    # 对于复杂结构，返回前500个字符的字符串表示
                    return str(result)[:500]
                else:
                    # 对于简单类型，直接返回
                    return str(result)[:300]

        except Exception as e:
            logger.warning(f"压缩tool result失败（工具: {tool_name}）: {e}")
            return str(result)[:200]

    def _extract_ids_and_key_fields(self, data: Any, max_depth: int = 3) -> Dict[str, Any]:
        """
        递归提取所有ID字段和关键字段（通用方法，适用于所有数据结构）

        提取规则：
        1. 提取所有包含"id"的字段（如id, product_id, order_id, order_item_id等）
        2. 提取关键状态字段（status, state, number, order_number等）
        3. 提取关键业务字段（name, title等，用于上下文理解）
        4. 对于列表，递归处理每个元素，提取其中的ID和关键字段
        5. 对于嵌套字典，递归提取
        6. 限制深度避免无限递归

        Args:
            data: 要提取的数据（可以是dict、list、基本类型）
            max_depth: 最大递归深度

        Returns:
            提取的关键字段字典（如果输入是列表，返回列表；如果是字典，返回字典）
        """
        if max_depth <= 0:
            return {}

        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                key_lower = key.lower()
                
                # 1. 提取所有包含"id"的字段
                if any(id_keyword in key_lower for id_keyword in ["id", "_id"]):
                    result[key] = value
                # 2. 提取关键状态和编号字段
                elif key_lower in ["status", "state", "number", "order_number", "code"]:
                    result[key] = value
                # 3. 提取关键业务字段（用于上下文理解）
                elif key_lower in ["name", "title", "type", "category"]:
                    result[key] = value
                # 4. 对于列表，递归提取其中的ID和关键字段
                elif isinstance(value, list):
                    extracted_list = self._extract_ids_and_key_fields(value, max_depth - 1)
                    if extracted_list:
                        result[key] = extracted_list
                # 5. 对于嵌套字典，递归提取
                elif isinstance(value, dict):
                    extracted = self._extract_ids_and_key_fields(value, max_depth - 1)
                    if extracted:
                        result[key] = extracted
            return result

        elif isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, (dict, list)):
                    # 递归提取嵌套结构
                    extracted = self._extract_ids_and_key_fields(item, max_depth - 1)
                    if extracted:
                        result.append(extracted)
                # 对于基本类型，如果是数字或看起来像ID的字符串，也保留
                elif isinstance(item, (int, float)) or (isinstance(item, str) and item.strip()):
                    # 简单类型直接保留（可能是ID值）
                    result.append(item)
            
            # 如果列表为空，返回空字典；否则返回列表
            return result if result else {}

        # 对于基本类型（字符串、数字等），如果是顶层调用，包装成字典
        # 但在递归调用中，基本类型通常不会到达这里
        return {}
