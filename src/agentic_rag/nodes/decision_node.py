"""决策节点实现

核心原则：答案优先 + 智能判断
- 先生成答案，再根据答案质量决定是否需要改进检索
- 使用 LLM 评估的 answer_type 来判断答案是"找到了"还是"没找到"
- 避免在生成答案前过度优化检索
"""
from typing import Optional, Dict, Any
from colorama import Fore, Style

from agentic_rag.advance_detector import AdvancedNeedsMoreInfoDetector
from src.agentic_rag.state import AgenticRAGState
from src.agentic_rag.threshold_config import ThresholdConfig


def create_decision_node(
    detector: AdvancedNeedsMoreInfoDetector,
    threshold_config: Optional[ThresholdConfig] = None
):
    """
    创建决策节点

    Args:
        detector: 信息需求检测器
        threshold_config: 阈值配置

    Returns:
        决策节点函数
    """
    if threshold_config is None:
        threshold_config = ThresholdConfig.default()

    detector.threshold_config = threshold_config

    def _should_use_adaptive_retrieval(
        iteration: int,
        adaptive_config
    ) -> bool:
        """判断是否应该使用自适应检索"""
        return (
            adaptive_config and 
            adaptive_config.enable_progressive_strategy and
            iteration < adaptive_config.max_retrieval_rounds
        )

    def _decide_retrieval_improvement(
        state: AgenticRAGState,
        iteration: int
    ) -> Dict[str, Any]:
        """决定如何改进检索（统一逻辑）"""
        adaptive_config = threshold_config.adaptive_retrieval

        # 优先使用 adaptive_retrieval 改进检索
        if _should_use_adaptive_retrieval(iteration, adaptive_config):
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 使用自适应检索改进检索策略{Style.RESET_ALL}")
            return {
                "next_action": "retrieve",
                "answer": "",
                "iteration_count": iteration + 1
            }

        # 回退策略：重新检索
        return {
            "next_action": "retrieve",
            "answer": "",
            "iteration_count": iteration + 1
        }

    def decision_node(state: AgenticRAGState) -> AgenticRAGState:
        """
        决策节点：决定下一步行动

        决策流程（答案优先）：
        1. 没有文档 → 检索
        2. 有文档没答案 → 生成答案
        3. 有答案且类型为 found → 完成
        4. 有答案但类型为 not_found 且检索质量低 → 改进检索
        5. 其他情况根据答案质量判断

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        retrieved_docs = state.get("retrieved_docs", [])
        answer = state.get("answer", "")
        retrieval_quality = state.get("retrieval_quality", 0.0)
        answer_quality = state.get("answer_quality", 0.0)
        answer_type = state.get("answer_type", "partial")  # found | not_found | partial

        answer_threshold = threshold_config.decision.answer_quality_threshold
        retrieval_threshold = threshold_config.decision.retrieval_quality_threshold

        # 检查是否达到最大迭代次数
        if iteration >= max_iterations:
            print(f"\n{Style.BRIGHT}{Fore.YELLOW}💭【decision】 达到最大迭代次数 ({max_iterations})，结束{Style.RESET_ALL}")
            return {"next_action": "finish"}

        print(f"\n{Style.BRIGHT}{Fore.YELLOW}💭【decision】 第 {iteration + 1} 轮决策{Style.RESET_ALL}")

        # ========== 步骤1：没有文档 → 检索 ==========
        if not retrieved_docs:
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 需要检索文档{Style.RESET_ALL}")
            return {"next_action": "retrieve"}

        # ========== 步骤2：有文档没答案 → 生成答案 ==========
        if not answer:
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 生成答案 (检索质量: {retrieval_quality:.2f}){Style.RESET_ALL}")
            return {"next_action": "generate"}

        # ========== 步骤3：根据答案类型和质量判断 ==========
        print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 答案类型: {answer_type}, 质量: {answer_quality:.2f}, 检索质量: {retrieval_quality:.2f}{Style.RESET_ALL}")

        # 答案类型为 "found"：成功找到答案
        if answer_type == "found":
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 答案已找到，完成{Style.RESET_ALL}")
            return {"next_action": "finish"}

        # 答案类型为 "not_found"：明确说没找到
        if answer_type == "not_found":
            # 检索质量低，可能是检索问题，尝试改进
            if retrieval_quality < retrieval_threshold and iteration < max_iterations - 1:
                print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 答案'未找到'且检索质量低 ({retrieval_quality:.2f})，尝试改进检索{Style.RESET_ALL}")
                return _decide_retrieval_improvement(state, iteration)
            else:
                # 检索质量已经够高，或已达最后一轮，接受"未找到"
                print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 检索质量已达标或已达最后轮，接受'未找到'答案{Style.RESET_ALL}")
                return {"next_action": "finish"}

        # 答案类型为 "partial"：部分回答
        if answer_quality >= answer_threshold:
            # 质量够高，接受部分答案
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 部分答案质量良好 ({answer_quality:.2f})，完成{Style.RESET_ALL}")
            return {"next_action": "finish"}

        # ========== 步骤4：答案质量不好 → 判断如何改进 ==========
        print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 答案质量不足 ({answer_quality:.2f} < {answer_threshold:.2f}){Style.RESET_ALL}")

        if iteration >= max_iterations - 1:
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 已达最后一轮，结束{Style.RESET_ALL}")
            return {"next_action": "finish"}

        # 检查失败分析是否建议重新进行意图识别
        failure_analysis = state.get("failure_analysis")
        intent_reclassification_count = state.get("intent_reclassification_count", 0)
        adaptive_config = threshold_config.adaptive_retrieval
        
        # 如果失败分析建议重新进行意图识别，且未超过最大重识别次数
        if (failure_analysis and 
            failure_analysis.get("needs_intent_reclassification") and
            adaptive_config and
            adaptive_config.enable_intent_reclassification and
            intent_reclassification_count < adaptive_config.max_reclassification_count and
            threshold_config.intent_classification.enable_intent_classification):
            
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 失败分析建议重新进行意图识别{Style.RESET_ALL}")
            return {
                "next_action": "reclassify_intent",
                "answer": "",
                "intent_reclassification_count": intent_reclassification_count + 1
            }

        # 使用 detector 判断是否需要更多信息
        question = state.get("question", "")
        needs_more_info = detector.needs_more_information(
            answer=answer,
            retrieved_docs=retrieved_docs,
            question=question,
            answer_quality=answer_quality
        )

        if not needs_more_info:
            # 信息足够，只是答案生成质量差，重新生成
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 信息充足，重新生成答案{Style.RESET_ALL}")
            return {
                "next_action": "generate",
                "answer": "",
                "iteration_count": iteration + 1
            }

        # 需要更多信息，尝试改进检索
        print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 需要更多信息，改进检索{Style.RESET_ALL}")
        return _decide_retrieval_improvement(state, iteration)

    return decision_node
