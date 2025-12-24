"""检索节点实现（简化版）

核心功能：
- 首轮：使用意图识别结果进行检索
- 第2轮：执行一次自适应检索（基于失败分析的策略调整）
- 第3轮及以后：不再检索，返回上一轮结果，让决策节点决定下一步

设计原则：
- 自适应检索只执行一次，避免过度检索
- 如果一次自适应检索还找不到，说明知识库中可能真的没有相关信息
"""
from typing import Optional, List, Dict, Any
from colorama import Fore, Style

from src.agentic_rag.state import AgenticRAGState
from src.agentic_rag.retriever import IntelligentRetriever
from src.agentic_rag.threshold_config import ThresholdConfig
from src.agentic_rag.adaptive_retrieval import (
    RetrievalFailureAnalyzer,
    SimpleProgressiveStrategy,
    FailureAnalysisResult
)


def create_retrieve_node(
    retriever: IntelligentRetriever,
    threshold_config: Optional[ThresholdConfig] = None
):
    """
    创建检索节点

    Args:
        retriever: 智能检索器
        threshold_config: 阈值配置

    Returns:
        检索节点函数
    """
    if threshold_config is None:
        threshold_config = ThresholdConfig.default()

    # 初始化失败分析器和渐进策略（如果启用）
    failure_analyzer = None
    progressive_strategy = None
    adaptive_config = threshold_config.adaptive_retrieval
    if adaptive_config and adaptive_config.enable_progressive_strategy:
        failure_analyzer = RetrievalFailureAnalyzer(threshold_config=threshold_config)
        progressive_strategy = SimpleProgressiveStrategy(threshold_config=threshold_config)
        print(f"{Style.BRIGHT}{Fore.GREEN}✅ 自适应检索已启用（简化版）{Style.RESET_ALL}")

    def _convert_failure_analysis_to_dict(failure_analysis: FailureAnalysisResult) -> Dict[str, Any]:
        """将失败分析结果转换为字典格式"""
        return {
            "failure_types": [ft.value for ft in failure_analysis.failure_types],
            "primary_failure": failure_analysis.primary_failure.value,
            "severity": failure_analysis.severity,
            "missing_aspects": failure_analysis.missing_aspects,
            "suggested_refinements": failure_analysis.suggested_refinements,
            "alternative_angles": failure_analysis.alternative_angles,
            "needs_intent_reclassification": failure_analysis.needs_intent_reclassification,
            "reasoning": failure_analysis.reasoning
        }


    def _fallback_retrieval(question: str, iteration: int) -> List:
        """回退检索策略"""
        strategies = ["hybrid"]
        k = min(threshold_config.retrieval.default_k + 3 * iteration, 15)
        print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve】 使用回退策略: {strategies}, k={k}{Style.RESET_ALL}")
        return retriever.retrieve(
            query=question,
            strategies=strategies,
            k=k,
            rewrite_query=True
        )

    def retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
        """
        检索节点：执行文档检索

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        question = state["question"]
        iteration = state.get("iteration_count", 0)
        query_intent = state.get("query_intent")

        print(f"\n{Style.BRIGHT}{Fore.BLUE}🔍【retrieve】 第 {iteration + 1} 轮检索{Style.RESET_ALL}")

        try:
            # 后续轮次：只在第2轮（iteration=1）执行一次自适应检索，之后不再继续
            if iteration > 0:
                # 获取上一轮的检索结果和质量
                previous_docs = state.get("retrieved_docs", [])
                previous_quality = state.get("retrieval_quality", 0.0)
                retrieval_history = state.get("retrieval_history", [])

                # 只在第2轮（iteration=1）执行一次自适应检索
                if (iteration == 1 and 
                    progressive_strategy and 
                    failure_analyzer and 
                    adaptive_config.enable_progressive_strategy):
                    print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve】 执行自适应检索（仅此一次）{Style.RESET_ALL}")
                    
                    # 分析上一轮的失败原因
                    previous_failure_analysis = failure_analyzer.analyze(
                        query=question,
                        retrieved_docs=previous_docs,
                        retrieval_quality=previous_quality,
                        query_intent=query_intent,
                        retrieval_history=retrieval_history,
                        iteration=0  # 分析首轮结果
                    )
                    
                    # 获取自适应检索配置（固定使用第2轮配置）
                    config = progressive_strategy.get_round_config(
                        round=1,  # 固定为第2轮
                        failure_analysis=previous_failure_analysis
                    )
                    
                    print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve】 自适应策略: {config.strategies}, k={config.k}, {config.description}{Style.RESET_ALL}")
                    
                    # 执行自适应检索
                    retrieved_docs = retriever.retrieve(
                        query=question,
                        strategies=config.strategies,
                        k=config.k,
                        rewrite_query=config.enable_rewrite
                    )
                else:
                    # 第3轮及以后：不再使用自适应检索
                    if iteration >= 2:
                        print(f"{Style.BRIGHT}{Fore.YELLOW}🔍【retrieve】 已执行过自适应检索（第2轮），不再继续检索{Style.RESET_ALL}")
                        # 返回上一轮的结果（不进行新检索）
                        retrieved_docs = previous_docs
                        # 使用上一轮的质量（不重新评估，避免重复计算）
                        quality = previous_quality
                        # 不更新检索历史（避免重复添加）
                        retrieval_history = state.get("retrieval_history", [])
                        
                        # 准备返回状态（跳过质量评估，但保留失败分析）
                        return_state = {
                            "retrieved_docs": retrieved_docs,
                            "retrieval_history": retrieval_history,
                            "retrieval_quality": quality,
                            "error_message": ""
                        }
                        
                        # 仍然进行失败分析（基于上一轮结果），供决策节点判断
                        if failure_analyzer and adaptive_config.enable_progressive_strategy:
                            failure_analysis = failure_analyzer.analyze(
                                query=question,
                                retrieved_docs=retrieved_docs,
                                retrieval_quality=quality,
                                query_intent=query_intent,
                                retrieval_history=retrieval_history,
                                iteration=iteration
                            )
                            return_state["failure_analysis"] = _convert_failure_analysis_to_dict(failure_analysis)
                        
                        return return_state
                    else:
                        # 未启用自适应检索，使用简单策略（向后兼容）
                        retrieved_docs = _fallback_retrieval(question, iteration)
            else:
                # 首轮：使用意图识别结果
                if query_intent:
                    strategies = query_intent.get("recommended_retrieval_strategy", ["semantic"])
                    if not isinstance(strategies, list):
                        strategies = [strategies] if strategies else ["semantic"]
                    k = query_intent.get("recommended_k", threshold_config.retrieval.default_k)

                    # 处理查询分解
                    split_queries = None
                    if query_intent.get("needs_decomposition") and query_intent.get("sub_queries"):
                        split_queries = _prepare_split_queries(
                            query_intent["sub_queries"],
                            strategies,
                            k
                        )
                        print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve】 查询分解：{len(split_queries)} 个子查询{Style.RESET_ALL}")

                    print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve】 意图: {query_intent.get('intent_type')}, 策略: {strategies}, k={k}{Style.RESET_ALL}")
                else:
                    strategies = ["semantic"]
                    k = threshold_config.retrieval.default_k
                    split_queries = None

                # 首轮检索
                retrieved_docs = retriever.retrieve(
                    query=question,
                    strategies=strategies,
                    k=k,
                    rewrite_query=False,
                    split_queries=split_queries if query_intent else None
                )

            print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve】 检索到 {len(retrieved_docs)} 个文档{Style.RESET_ALL}")

            # 评估检索质量
            quality, _ = retriever.evaluate_retrieval_quality(
                question,
                retrieved_docs,
                threshold=threshold_config.retrieval.quality_threshold
            )
            print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve】 检索质量: {quality:.2f}{Style.RESET_ALL}")

            # 更新检索历史
            retrieval_history = state.get("retrieval_history", [])
            retrieval_history.append(retrieved_docs)

            # 准备返回状态
            return_state = {
                "retrieved_docs": retrieved_docs,
                "retrieval_history": retrieval_history,
                "retrieval_quality": quality,
                "error_message": ""
            }

            # 如果启用自适应检索，更新失败分析信息（用于决策节点判断）
            if failure_analyzer and adaptive_config.enable_progressive_strategy:
                # 分析当前轮次的检索结果
                failure_analysis = failure_analyzer.analyze(
                    query=question,
                    retrieved_docs=retrieved_docs,
                    retrieval_quality=quality,
                    query_intent=query_intent,
                    retrieval_history=retrieval_history,
                    iteration=iteration
                )
                
                # 转换失败分析结果为字典
                failure_analysis_dict = _convert_failure_analysis_to_dict(failure_analysis)
                
                # 更新状态
                return_state["failure_analysis"] = failure_analysis_dict
                
                # 如果失败分析建议重新进行意图识别
                if failure_analysis.needs_intent_reclassification:
                    print(f"{Style.BRIGHT}{Fore.YELLOW}🔍【retrieve】 失败分析建议重新进行意图识别{Style.RESET_ALL}")

            return return_state

        except Exception as e:
            error_msg = f"检索错误: {str(e)}"
            print(f"{Style.BRIGHT}{Fore.RED}🔍【retrieve】 ❌ {error_msg}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return {
                "error_message": error_msg,
                "retrieved_docs": [],
                "retrieval_quality": 0.0
            }

    return retrieve_node


def _prepare_split_queries(
    sub_queries: List[Any],
    default_strategies: List[str],
    default_k: int
) -> List[Dict[str, Any]]:
    """准备分解查询列表"""
    split_queries = []

    for item in sub_queries:
        if isinstance(item, dict):
            split_queries.append({
                "query": item.get("query", ""),
                "strategy": item.get("recommended_strategy", default_strategies),
                "k": item.get("recommended_k", 3),
            })
        elif isinstance(item, str):
            split_queries.append({
                "query": item,
                "strategy": default_strategies,
                "k": max(1, default_k // len(sub_queries)),
            })
        else:
            # Pydantic 模型
            split_queries.append({
                "query": getattr(item, 'query', str(item)),
                "strategy": getattr(item, 'recommended_strategy', ["semantic"]),
                "k": getattr(item, 'recommended_k', 3),
            })

    return split_queries
