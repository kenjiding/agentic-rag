"""意图识别节点实现

2025 企业级最佳实践：
- 支持初始意图识别和动态重识别
- 基于失败分析的意图重识别
- 多轮检索失败后重新分解问题
"""
from typing import Optional, Dict, Any
from colorama import Fore, Style

from src.agentic_rag.state import AgenticRAGState
from src.intent import IntentClassifier
from src.agentic_rag.threshold_config import ThresholdConfig


def create_intent_classification_node(
    intent_classifier: IntentClassifier,
    threshold_config: Optional[ThresholdConfig] = None
):
    """
    创建意图识别节点

    2025 企业级最佳实践：
    - 支持初始意图识别
    - 支持动态意图重识别（多轮检索失败后）
    - 基于失败分析调整分解策略

    Args:
        intent_classifier: 意图分类器
        threshold_config: 阈值配置

    Returns:
        意图识别节点函数
    """
    if threshold_config is None:
        threshold_config = ThresholdConfig.default()

    def intent_node(state: AgenticRAGState) -> AgenticRAGState:
        """
        意图识别节点：分析用户查询的意图

        支持两种模式：
        1. 初始识别：首次进入时进行完整意图识别
        2. 重识别：多轮检索失败后，基于失败分析重新识别

        Args:
            state: 当前状态

        Returns:
            更新后的状态（包含query_intent）
        """
        question = state["question"]
        next_action = state.get("next_action", "")
        failure_analysis = state.get("failure_analysis")
        reclass_count = state.get("intent_reclassification_count", 0)

        # 判断是否是重识别模式
        is_reclassification = (next_action == "reclassify_intent")

        if is_reclassification:
            print(f"\n{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 动态意图重识别 (第 {reclass_count} 次)...{Style.RESET_ALL}")
        else:
            print(f"\n{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 意图识别...{Style.RESET_ALL}")

        print(f"{Style.BRIGHT}{Fore.MAGENTA}查询: {question}{Style.RESET_ALL}")

        # 检查是否启用意图识别
        if not threshold_config.intent_classification.enable_intent_classification:
            print(f"{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 意图识别已禁用，跳过{Style.RESET_ALL}")
            return {"query_intent": None}

        try:
            if is_reclassification:
                # 动态重识别模式
                intent = _reclassify_intent(
                    intent_classifier=intent_classifier,
                    question=question,
                    failure_analysis=failure_analysis,
                    previous_intent=state.get("query_intent"),
                    reclass_count=reclass_count
                )
            else:
                # 初始识别模式
                intent = intent_classifier.classify(question)

            # 转换为字典格式
            intent_dict = intent.model_dump()

            # 打印识别结果
            _print_intent_result(intent, is_reclassification)

            return {
                "query_intent": intent_dict,
                "error_message": ""
            }

        except Exception as e:
            error_msg = f"意图识别错误: {str(e)}"
            print(f"{Style.BRIGHT}{Fore.YELLOW}🎯【intent节点】 ❌ {error_msg}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return {
                "query_intent": None,
                "error_message": error_msg
            }

    return intent_node


def _reclassify_intent(
    intent_classifier: IntentClassifier,
    question: str,
    failure_analysis: Optional[Dict[str, Any]],
    previous_intent: Optional[Dict[str, Any]],
    reclass_count: int
):
    """
    基于失败分析进行意图重识别

    策略：
    1. 如果之前的分解方式失败，尝试不同的分解类型
    2. 基于失败分析的建议调整检索策略
    3. 考虑缺失的信息方面
    """
    print(f"{Style.BRIGHT}{Fore.CYAN}🔄【重识别】 分析之前的失败原因...{Style.RESET_ALL}")

    # 构建重识别的上下文提示
    reclassification_context = _build_reclassification_context(
        failure_analysis=failure_analysis,
        previous_intent=previous_intent,
        reclass_count=reclass_count
    )

    if reclassification_context:
        print(f"{Style.BRIGHT}{Fore.CYAN}🔄【重识别】 上下文: {reclassification_context[:200]}...{Style.RESET_ALL}")

    # 使用增强的查询进行重识别
    # 将失败上下文附加到查询中，让 LLM 考虑之前的失败
    enhanced_query = f"""{question}

[重识别上下文]
{reclassification_context}

请基于以上失败分析，重新分析查询意图并尝试不同的分解策略。"""

    # 进行重识别
    intent = intent_classifier.classify(enhanced_query)

    # 如果之前的分解失败了，尝试强制使用不同的分解类型
    if previous_intent and previous_intent.get("needs_decomposition"):
        prev_decomposition_type = previous_intent.get("decomposition_type")
        if intent.decomposition_type == prev_decomposition_type:
            print(f"{Style.BRIGHT}{Fore.CYAN}🔄【重识别】 检测到相同的分解类型，尝试替换...{Style.RESET_ALL}")
            # 这里可以强制更换分解类型，但为了保持 LLM 的判断，我们只是记录
            pass

    return intent


def _build_reclassification_context(
    failure_analysis: Optional[Dict[str, Any]],
    previous_intent: Optional[Dict[str, Any]],
    reclass_count: int
) -> str:
    """构建重识别上下文"""
    context_parts = []

    if failure_analysis:
        failure_types = failure_analysis.get("failure_types", [])
        missing_aspects = failure_analysis.get("missing_aspects", [])
        suggested_refinements = failure_analysis.get("suggested_refinements", [])
        alternative_angles = failure_analysis.get("alternative_angles", [])

        if failure_types:
            context_parts.append(f"之前的检索失败类型: {', '.join(failure_types)}")

        if missing_aspects:
            context_parts.append(f"缺失的信息方面: {', '.join(missing_aspects[:3])}")

        if suggested_refinements:
            context_parts.append(f"建议的改进方向: {', '.join(suggested_refinements[:3])}")

        if alternative_angles:
            context_parts.append(f"替代的查询角度: {', '.join(alternative_angles[:3])}")

    if previous_intent:
        prev_type = previous_intent.get("intent_type", "unknown")
        prev_decomp = previous_intent.get("decomposition_type")
        prev_sub_queries = previous_intent.get("sub_queries", [])

        context_parts.append(f"之前识别的意图类型: {prev_type}")

        if prev_decomp:
            context_parts.append(f"之前的分解类型: {prev_decomp} (未能有效检索)")
            context_parts.append("请尝试使用不同的分解策略")

        if prev_sub_queries:
            prev_queries = [
                sq.get("query", sq) if isinstance(sq, dict) else str(sq)
                for sq in prev_sub_queries[:3]
            ]
            context_parts.append(f"之前的子查询: {prev_queries}")

    if reclass_count > 1:
        context_parts.append(f"这是第 {reclass_count} 次重识别，请尝试更激进的分解策略")

    return "\n".join(context_parts) if context_parts else ""


def _print_intent_result(intent, is_reclassification: bool):
    """打印意图识别结果"""
    prefix = "🔄【重识别】" if is_reclassification else "🎯【intent节点】"

    print(f"{Style.BRIGHT}{Fore.MAGENTA}{prefix} 意图类型: {intent.intent_type}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.MAGENTA}{prefix} 复杂度: {intent.complexity}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.MAGENTA}{prefix} 需要分解: {intent.needs_decomposition}{Style.RESET_ALL}")

    if intent.needs_decomposition:
        print(f"{Style.BRIGHT}{Fore.MAGENTA}{prefix} 分解类型: {intent.decomposition_type}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{Fore.MAGENTA}{prefix} 子查询数: {len(intent.sub_queries)}{Style.RESET_ALL}")
        for i, sq in enumerate(intent.sub_queries[:3]):
            sq_query = sq.query if hasattr(sq, 'query') else sq.get('query', str(sq))
            print(f"{Style.BRIGHT}{Fore.MAGENTA}  {i+1}. {sq_query[:50]}...{Style.RESET_ALL}")

    print(f"{Style.BRIGHT}{Fore.MAGENTA}{prefix} 推荐策略: {intent.recommended_retrieval_strategy}, k={intent.recommended_k}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.MAGENTA}{prefix} 置信度: {intent.confidence:.2f}{Style.RESET_ALL}")

    if intent.reasoning:
        print(f"{Style.BRIGHT}{Fore.MAGENTA}{prefix} 推理: {intent.reasoning[:100]}...{Style.RESET_ALL}")
