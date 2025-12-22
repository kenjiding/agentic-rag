"""检索节点实现

简洁设计：
- 首轮：使用意图识别结果进行检索
- 后续轮次：使用 hybrid 策略扩大检索范围
"""
from typing import Optional, List, Dict, Any
from colorama import Fore, Style

from src.agentic_rag.state import AgenticRAGState
from src.agentic_rag.retriever import IntelligentRetriever
from src.agentic_rag.threshold_config import ThresholdConfig


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
            # 确定检索策略和参数
            if query_intent:
                # 使用意图识别结果
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
                # 默认策略
                strategies = ["semantic"]
                k = threshold_config.retrieval.default_k
                split_queries = None

            # 后续轮次使用 hybrid 策略扩大范围
            if iteration > 0:
                if "hybrid" not in strategies:
                    strategies = ["hybrid"]
                k = min(k + 3, 15)  # 增加 k 值
                print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve】 后续轮次，升级策略: {strategies}, k={k}{Style.RESET_ALL}")

            # 执行检索
            retrieved_docs = retriever.retrieve(
                query=question,
                strategies=strategies,
                k=k,
                rewrite_query=(iteration > 0),  # 后续轮次启用查询改写
                split_queries=split_queries
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

            return {
                "retrieved_docs": retrieved_docs,
                "retrieval_history": retrieval_history,
                "retrieval_quality": quality,
                "error_message": ""
            }

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
