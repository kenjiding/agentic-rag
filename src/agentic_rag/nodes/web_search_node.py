"""Web Search 节点实现 (Corrective RAG)"""
from typing import Optional
from colorama import Fore, Style

from src.agentic_rag.state import AgenticRAGState
from src.agentic_rag.threshold_config import ThresholdConfig
from src.agentic_rag.web_search import CorrectiveRAGHandler
from src.agentic_rag.retriever import IntelligentRetriever


def create_web_search_node(
    crag_handler: CorrectiveRAGHandler,
    retriever: Optional[IntelligentRetriever] = None,
    threshold_config: Optional[ThresholdConfig] = None
):
    """
    创建 Web Search 节点 (Corrective RAG)

    2025 最佳实践：当本地检索质量不足时，使用 Web 搜索获取外部信息

    Args:
        crag_handler: CRAG 处理器
        retriever: 智能检索器（用于评估检索质量）
        threshold_config: 阈值配置

    Returns:
        Web Search 节点函数
    """
    if threshold_config is None:
        threshold_config = ThresholdConfig.default()

    def web_search_node(state: AgenticRAGState) -> AgenticRAGState:
        """
        Web Search 节点：执行 Web 搜索并融合结果

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        question = state["question"]
        retrieved_docs = state.get("retrieved_docs", [])
        retrieval_quality = state.get("retrieval_quality", 0.0)
        iteration = state.get("iteration_count", 0)
        web_search_count = state.get("web_search_count", 0)

        print(f"\n{Style.BRIGHT}{Fore.CYAN}🌐【web_search节点】 执行 Web 搜索...{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{Fore.CYAN}查询: {question}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{Fore.CYAN}当前检索质量: {retrieval_quality:.2f}{Style.RESET_ALL}")

        try:
            # 直接执行 Web 搜索，不再重复判断
            # 注意：decision_node 已经判断过需要 web_search，这里直接执行
            # 避免 crag_handler.process() 内部的 should_trigger_web_search 再次判断导致死循环
            web_docs = crag_handler.perform_web_search(question, optimize_query=True)

            if web_docs:
                # 精炼并融合结果
                refined_docs = crag_handler.refine_web_results(question, web_docs)
                merged_docs = crag_handler.merge_results(retrieved_docs, refined_docs)
                used_web_search = True
                web_results_count = len(refined_docs)
            else:
                # Web 搜索没有返回结果
                merged_docs = retrieved_docs
                used_web_search = False  # 标记为未成功使用，但不会死循环因为 web_search_count 会增加
                web_results_count = 0

            if used_web_search:
                print(f"{Style.BRIGHT}{Fore.CYAN}🌐【web_search节点】 Web 搜索完成，获取 {web_results_count} 个结果{Style.RESET_ALL}")
                print(f"{Style.BRIGHT}{Fore.CYAN}🌐【web_search节点】 融合后共 {len(merged_docs)} 个文档{Style.RESET_ALL}")
            else:
                print(f"{Style.BRIGHT}{Fore.CYAN}🌐【web_search节点】 Web 搜索未触发或不可用{Style.RESET_ALL}")
            
            for doc in web_docs:
                print(f"{Style.BRIGHT}{Fore.CYAN}Web 搜索结果: {doc.page_content}{Style.RESET_ALL}")
            # 更新状态
            tools_used = state.get("tools_used", [])
            if used_web_search and "web_search" not in tools_used:
                tools_used.append("web_search")

            # 更新检索历史
            retrieval_history = state.get("retrieval_history", [])
            if used_web_search:
                retrieval_history.append(merged_docs)

            # 评估合并后的检索质量
            new_quality = retrieval_quality  # 默认保持原值
            if retriever and merged_docs:
                quality_threshold = threshold_config.retrieval.quality_threshold
                new_quality, _ = retriever.evaluate_retrieval_quality(
                    question,
                    merged_docs,
                    threshold=quality_threshold
                )
                print(f"{Style.BRIGHT}{Fore.CYAN}🌐【web_search节点】 更新检索质量: {retrieval_quality:.2f} → {new_quality:.2f}{Style.RESET_ALL}")

            return {
                "retrieved_docs": merged_docs,
                "retrieval_history": retrieval_history,
                "retrieval_quality": new_quality,  # 关键：更新检索质量
                "web_search_used": used_web_search,
                "web_search_results": web_docs if web_docs else [],
                # 关键修复：无论成功与否都增加计数，避免死循环
                # decision_node 用 web_search_count < 1 来判断是否还能触发
                "web_search_count": web_search_count + 1,
                "tools_used": tools_used,
                "error_message": ""
            }

        except Exception as e:
            error_msg = f"Web 搜索错误: {str(e)}"
            print(f"{Style.BRIGHT}{Fore.YELLOW}🌐【web_search节点】 ❌ {error_msg}{Style.RESET_ALL}")
            return {
                "error_message": error_msg,
                "web_search_used": False,
                # 即使出错也增加计数，避免死循环
                "web_search_count": web_search_count + 1
            }

    return web_search_node

