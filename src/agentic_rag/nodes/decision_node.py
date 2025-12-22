"""决策节点实现

核心原则：答案优先
- 先生成答案，再根据答案质量决定是否需要改进检索
- 避免在生成答案前过度优化检索
"""
from typing import Optional
from colorama import Fore, Style

from agentic_rag.advance_detector import AdvancedNeedsMoreInfoDetector
from src.agentic_rag.state import AgenticRAGState
from src.agentic_rag.threshold_config import ThresholdConfig
from src.agentic_rag.web_search import CorrectiveRAGHandler


def create_decision_node(
    detector: AdvancedNeedsMoreInfoDetector,
    crag_handler: Optional[CorrectiveRAGHandler] = None,
    threshold_config: Optional[ThresholdConfig] = None
):
    """
    创建决策节点

    Args:
        detector: 信息需求检测器
        crag_handler: CRAG 处理器（可选，启用 Web Search）
        threshold_config: 阈值配置

    Returns:
        决策节点函数
    """
    if threshold_config is None:
        threshold_config = ThresholdConfig.default()

    detector.threshold_config = threshold_config

    # 检查是否启用 Web Search
    enable_web_search = (
        crag_handler is not None and
        hasattr(crag_handler, 'web_search') and
        crag_handler.web_search.available
    )

    def decision_node(state: AgenticRAGState) -> AgenticRAGState:
        """
        决策节点：决定下一步行动

        决策流程（答案优先）：
        1. 没有文档 → 检索
        2. 有文档没答案 → 生成答案
        3. 有答案且质量好 → 完成
        4. 有答案但质量差 → 根据情况改进

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 5)
        retrieved_docs = state.get("retrieved_docs", [])
        answer = state.get("answer", "")
        retrieval_quality = state.get("retrieval_quality", 0.0)
        answer_quality = state.get("answer_quality", 0.0)
        web_search_count = state.get("web_search_count", 0)

        answer_threshold = threshold_config.decision.answer_quality_threshold

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

        # ========== 步骤3：有答案且质量好 → 完成 ==========
        if answer_quality >= answer_threshold:
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 答案质量良好 ({answer_quality:.2f})，完成{Style.RESET_ALL}")
            return {"next_action": "finish"}

        # ========== 步骤4：答案质量不好 → 判断如何改进 ==========
        print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 答案质量不足 ({answer_quality:.2f} < {answer_threshold:.2f}){Style.RESET_ALL}")

        # 检查是否还有改进空间
        if iteration >= max_iterations - 1:
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 已达最后一轮，结束{Style.RESET_ALL}")
            return {"next_action": "finish"}

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
                "answer": "",  # 清空答案，重新生成
                "iteration_count": iteration + 1
            }

        # 需要更多信息，尝试改进检索
        print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 需要更多信息，改进检索{Style.RESET_ALL}")

        # 优先尝试 Web Search（如果可用且未使用过）
        if enable_web_search and web_search_count < 1:
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision】 尝试 Web 搜索{Style.RESET_ALL}")
            return {
                "next_action": "web_search",
                "answer": "",  # 清空答案
                "iteration_count": iteration + 1
            }

        # 重新检索
        return {
            "next_action": "retrieve",
            "answer": "",  # 清空答案
            "iteration_count": iteration + 1
        }

    return decision_node
