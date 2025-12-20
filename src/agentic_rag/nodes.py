"""Agentic RAG 节点实现"""
from typing import List, Optional
from colorama import Fore, Style, init

from agentic_rag.advance_detector import AdvancedNeedsMoreInfoDetector
from src.agentic_rag.state import AgenticRAGState
from src.agentic_rag.retriever import IntelligentRetriever
from src.agentic_rag.generator import IntelligentGenerator
from src.agentic_rag.intent_analyse import IntentClassifier
from src.agentic_rag.threshold_config import ThresholdConfig
from src.agentic_rag.intent_analyse import QueryOptimizer

def create_intent_classification_node(
    intent_classifier: IntentClassifier,
    threshold_config: Optional[ThresholdConfig] = None
):
    """
    创建意图识别节点
    
    基于2025-2026年最佳实践，在接收到用户问题后，首先进行意图识别。
    
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
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态（包含query_intent）
        """
        question = state["question"]
        
        print(f"\n{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 意图识别...{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{Fore.MAGENTA}查询: {question}{Style.RESET_ALL}")
        
        # 检查是否启用意图识别
        if not threshold_config.intent_classification.enable_intent_classification:
            print(f"{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 意图识别已禁用，跳过{Style.RESET_ALL}")
            return {"query_intent": None}
        
        try:
            # 进行意图识别
            intent = intent_classifier.classify(question)
            
            # 转换为字典格式（使用 Pydantic 的 model_dump 方法）
            intent_dict = intent.model_dump()
            
            print(f"{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 意图类型: {intent.intent_type}{Style.RESET_ALL}")
            print(f"{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 复杂度: {intent.complexity}{Style.RESET_ALL}")
            print(f"{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 是否对比: {intent.is_comparison}{Style.RESET_ALL}")
            print(f"{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 推荐策略: {intent.recommended_retrieval_strategy}, k={intent.recommended_k}{Style.RESET_ALL}")
            print(f"{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 置信度: {intent.confidence:.2f}{Style.RESET_ALL}")
            if intent.reasoning:
                print(f"{Style.BRIGHT}{Fore.MAGENTA}🎯【intent节点】 推理: {intent.reasoning[:100]}...{Style.RESET_ALL}")
            
            return {"query_intent": intent_dict}
            
        except Exception as e:
            error_msg = f"意图识别错误: {str(e)}"
            print(f"{Style.BRIGHT}{Fore.YELLOW}🎯【intent节点】 ❌ {error_msg}{Style.RESET_ALL}")
            return {
                "query_intent": None,
                "error_message": error_msg
            }
    
    return intent_node


def create_retrieve_node(
    retriever: IntelligentRetriever,
    threshold_config: Optional[ThresholdConfig] = None
):
    """
    创建检索节点
    
    Args:
        retriever: 智能检索器
        threshold_config: 阈值配置（如果为None，使用默认配置）
        
    Returns:
        检索节点函数
    """
    # 使用默认配置如果未提供
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
        
        print(f"\n{Style.BRIGHT}{Fore.BLUE}🔍【retrieve节点】 迭代 {iteration + 1} - 检索操作{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{Fore.BLUE}查询: {question}{Style.RESET_ALL}")
        
        try:
            # 优先使用意图识别结果
            query_intent = state.get("query_intent")
            rewrite_query = False
            strategies = None
            split_queries = None
            
            if query_intent:
                # 根据意图识别结果调整检索策略（现在是列表）
                strategies = query_intent.get("recommended_retrieval_strategy", ["semantic"])
                # 确保strategies是列表
                if not isinstance(strategies, list):
                    strategies = [strategies] if strategies else ["semantic"]
                k = query_intent.get("recommended_k", threshold_config.retrieval.default_k)
                
                # 如果是对比查询，直接使用comparison_items作为拆分查询
                intent_type = query_intent.get("intent_type")
                is_comparison = query_intent.get("is_comparison", False)
                comparison_items = query_intent.get("comparison_items", [])
                print("query_intent意图识别结果:", query_intent)
                
                if (intent_type == "comparison" or is_comparison) and comparison_items:
                    # comparison_items 已经包含拆分后的完整查询
                    split_queries = comparison_items
                    print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve节点】 对比查询：使用 {len(split_queries)} 个拆分查询{Style.RESET_ALL}")
                else:
                    split_queries = None
                
                print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve节点】 使用意图识别结果: strategies={strategies}, k={k}{Style.RESET_ALL}")
            else:
                # 回退到原有逻辑
                strategies = ["semantic"]
                k = threshold_config.retrieval.default_k
                
                if iteration > 0:
                    # 第二轮及以后，尝试改写查询或使用混合检索
                    quality_threshold = threshold_config.retrieval.quality_for_hybrid_search
                    if state.get("retrieval_quality", 1.0) < quality_threshold:
                        strategies = ["hybrid"]
                        rewrite_query = True
            
            # 准备上下文（只在需要时使用，避免误导查询改写）
            context = None
            # 只有在明确需要改写且已有部分检索结果时才提供上下文
            if not query_intent and rewrite_query and iteration > 0 and state.get("retrieved_docs"):
                # 使用配置的上下文长度
                context_length = threshold_config.retrieval.context_length_for_rewrite
                context = state["retrieved_docs"][0].page_content[:context_length]
            
            # 执行检索（使用意图识别建议的k值，如果没有则使用默认值）
            final_k = k if query_intent else threshold_config.retrieval.default_k
            final_rewrite = rewrite_query if not query_intent else False
            retrieved_docs = retriever.retrieve(
                query=question,
                strategies=strategies,
                context=context,
                k=final_k,
                rewrite_query=final_rewrite,
                split_queries=split_queries
            )
            
            print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve节点】 检索到 {len(retrieved_docs)} 个文档{Style.RESET_ALL}")
            
            # 评估检索质量（使用动态阈值）
            quality_threshold = threshold_config.retrieval.quality_threshold
            quality, meets_threshold = retriever.evaluate_retrieval_quality(
                question,
                retrieved_docs,
                threshold=quality_threshold
            )
            
            print(f"{Style.BRIGHT}{Fore.BLUE}🔍【retrieve节点】 检索质量: {quality:.2f} (阈值: {quality_threshold:.2f}, {'通过' if meets_threshold else '未通过'}){Style.RESET_ALL}")
            
            # 更新状态
            retrieval_history = state.get("retrieval_history", [])
            retrieval_history.append(retrieved_docs)
            
            # 更新迭代计数（检索操作算一次迭代）
            current_iteration = state.get("iteration_count", 0)
            
            return {
                "retrieved_docs": retrieved_docs,
                "retrieval_history": retrieval_history,
                "retrieval_quality": quality,
                "retrieval_strategy": str(strategies) if strategies else "semantic",  # 保存策略列表
                "iteration_count": current_iteration + 1,
                "error_message": ""
            }
            
        except Exception as e:
            error_msg = f"检索错误: {str(e)}"
            print(f"{Style.BRIGHT}{Fore.YELLOW}🔍【retrieve]点】 ❌ {error_msg}{Style.RESET_ALL}")
            return {
                "error_message": error_msg,
                "retrieved_docs": [],
                "retrieval_quality": 0.0
            }
    
    return retrieve_node


def create_generate_node(
    generator: IntelligentGenerator,
    threshold_config: Optional[ThresholdConfig] = None
):
    """
    创建生成节点
    
    Args:
        generator: 智能生成器
        threshold_config: 阈值配置（如果为None，使用默认配置）
        
    Returns:
        生成节点函数
    """
    # 使用默认配置如果未提供
    if threshold_config is None:
        threshold_config = ThresholdConfig.default()
    
    def generate_node(state: AgenticRAGState) -> AgenticRAGState:
        """
        生成节点：基于检索结果生成答案
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        question = state["question"]
        retrieved_docs = state.get("retrieved_docs", [])
        previous_answer = state.get("answer", "")
        
        print(f"\n{Style.BRIGHT}{Fore.GREEN}🚢【generator节点】 生成答案...{Style.RESET_ALL}")
        
        try:
            # 格式化上下文
            context = generator.format_context(retrieved_docs)
            
            # 决定生成模式
            if previous_answer and state.get("iteration_count", 0) > 0:
                # 改进模式：生成反馈，然后改进答案
                print(f"{Style.BRIGHT}{Fore.GREEN}🚢【generator节点】 改进模式{Style.RESET_ALL}")
                feedback = generator.generate_feedback(question, previous_answer, context)
                print(f"{Style.BRIGHT}{Fore.GREEN}🚢【generator节点】 改进反馈: {feedback[:100]}...{Style.RESET_ALL}")
                
                answer = generator.generate(
                    question=question,
                    context=context,
                    previous_answer=previous_answer,
                    feedback=feedback
                )
            else:
                # 首次生成模式（传递意图信息）
                print(f"{Style.BRIGHT}{Fore.GREEN}🚢【generator节点】 首次生成模式{Style.RESET_ALL}")
                query_intent = state.get("query_intent")
                answer = generator.generate(
                    question=question,
                    context=context,
                    query_intent=query_intent
                )
            
            print(f"{Style.BRIGHT}{Fore.GREEN}🚢【generator节点】 生成答案长度: {len(answer)} 字符{Style.RESET_ALL}")
            print(f"{Style.BRIGHT}{Fore.GREEN}🚢【generator节点】 答案预览: {answer[:200]}...{Style.RESET_ALL}")
            
            # 评估答案质量
            quality_threshold = threshold_config.generation.answer_quality_threshold
            quality, meets_threshold, feedback = generator.evaluate_answer_quality(
                question,
                answer,
                context,
                threshold=quality_threshold
            )
            
            print(f"{Style.BRIGHT}{Fore.GREEN}🚢【generator节点】 答案质量: {quality:.2f} (阈值: {quality_threshold:.2f}, {'通过' if meets_threshold else '未通过'}){Style.RESET_ALL}")
            if feedback:
                print(f"{Style.BRIGHT}{Fore.GREEN}🚢【generator节点】 评估反馈: {feedback[:100]}...{Style.RESET_ALL}")
            
            # 更新生成历史
            generation_history = state.get("generation_history", [])
            generation_history.append(answer)
            
            # 更新迭代计数（生成操作算一次迭代）
            current_iteration = state.get("iteration_count", 0)
            
            return {
                "answer": answer,
                "generation_history": generation_history,
                "answer_quality": quality,
                "evaluation_feedback": feedback,
                "iteration_count": current_iteration + 1,
                "error_message": ""
            }
            
        except Exception as e:
            error_msg = f"生成错误: {str(e)}"
            print(f"{Style.BRIGHT}{Fore.YELLOW}🚢【generator]点】 ❌ {error_msg}{Style.RESET_ALL}")
            return {
                "error_message": error_msg,
                "answer": "",
                "answer_quality": 0.0
            }
    
    return generate_node


def create_decision_node(
    detector: AdvancedNeedsMoreInfoDetector,
    query_optimizer: QueryOptimizer,
    threshold_config: Optional[ThresholdConfig] = None
):
    """
    创建决策节点
    
    Args:
        detector: 信息需求检测器
        threshold_config: 阈值配置（如果为None，使用默认配置）
    
    Returns:
        决策节点函数
    """
    # 将 threshold_config 附加到 detector 上，以便在决策节点中使用
    if threshold_config:
        detector.threshold_config = threshold_config
    
    def decision_node(state: AgenticRAGState) -> AgenticRAGState:
        """
        决策节点：决定下一步行动
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态（包含 next_action）
        """
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 5)
        retrieved_docs = state.get("retrieved_docs", [])
        answer = state.get("answer", "")
        retrieval_quality = state.get("retrieval_quality", 0.0)
        answer_quality = state.get("answer_quality", 0.0)
        
        print(f"\n{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 第 {iteration + 1} 轮决策{Style.RESET_ALL}")
        
        # 如果超过最大迭代次数，结束
        if iteration >= max_iterations:
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 达到最大迭代次数，结束{Style.RESET_ALL}")
            return {"next_action": "finish"}
        
        # 如果没有检索过，先检索
        if not retrieved_docs:
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 需要检索文档{Style.RESET_ALL}")
            return {"next_action": "retrieve"}
        
        # 如果检索质量不够，继续检索（尝试不同策略），但避免无限循环
        # 使用配置的阈值（从 detector 中获取，如果没有则使用默认值 0.7）
        retrieval_threshold = 0.7  # 默认值，如果 detector 有 threshold_config 则使用配置值
        if hasattr(detector, 'threshold_config') and detector.threshold_config:
            retrieval_threshold = detector.threshold_config.decision.retrieval_quality_threshold
        
        if (retrieval_quality < retrieval_threshold) and iteration < 2 and len(retrieved_docs) > 0:
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 检索质量不足 ({retrieval_quality:.2f})，继续检索{Style.RESET_ALL}")
            # 使用意图识别对用户问题进行优化
            query_intent = state.get("query_intent")
            if query_intent:
                origin_question = query_intent.get("query")
                optimized_query = query_optimizer.optimize(origin_question, query_intent)
                print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 优化后的查询: {optimized_query.primary_query}{Style.RESET_ALL}")
                return {"next_action": "retrieve", "question": optimized_query.primary_query}
            return {"next_action": "retrieve"}
        
        # 如果检索失败（0个文档）且已尝试多次，尝试生成或结束
        if len(retrieved_docs) == 0 and iteration >= 2:
            if not answer:
                print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 检索失败，尝试生成答案{Style.RESET_ALL}")
                return {"next_action": "generate"}
            else:
                print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 检索失败且已有答案，结束{Style.RESET_ALL}")
                return {"next_action": "finish"}
        
        # 如果没有生成过答案，生成
        if not answer:
            print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 需要生成答案{Style.RESET_ALL}")
            return {"next_action": "generate"}
        
        # 如果答案质量不够，需要判断是检索问题还是生成问题
        # 使用配置的阈值
        answer_threshold = 0.7  # 默认值
        retrieval_threshold_for_decision = 0.7  # 默认值
        if hasattr(detector, 'threshold_config') and detector.threshold_config:
            answer_threshold = detector.threshold_config.decision.answer_quality_threshold
            retrieval_threshold_for_decision = detector.threshold_config.decision.retrieval_quality_threshold
        
        if answer_quality < answer_threshold:
            question = state.get("question", "")
            
            # 关键逻辑：如果检索质量已经很高，但答案质量仍然很低，
            # 说明问题不在检索，而在生成，应该优先重新生成
            if retrieval_quality >= retrieval_threshold_for_decision:
                # 检索质量高但答案质量低，优先重新生成
                # 可能是生成器没有正确利用上下文，或者需要改进生成策略
                print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 检索质量高 ({retrieval_quality:.2f}) 但答案质量低 ({answer_quality:.2f})，重新生成{Style.RESET_ALL}")
                return {"next_action": "generate"}
            
            # 检索质量不够，检查是否需要更多信息
            needs_more_info = detector.needs_more_information(
                answer=answer,
                retrieved_docs=retrieved_docs,
                question=question,
                answer_quality=answer_quality
            )
            
            # 如果确实需要更多信息，且还有迭代次数，继续检索
            if needs_more_info and iteration < max_iterations - 1:
                print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 答案质量不足 ({answer_quality:.2f})，检索质量不足 ({retrieval_quality:.2f})，需要更多信息，继续检索{Style.RESET_ALL}")
                return {"next_action": "retrieve"}
            else:
                # 不需要更多信息，或已达到最大迭代次数，重新生成
                print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 答案质量不足 ({answer_quality:.2f})，重新生成{Style.RESET_ALL}")
                return {"next_action": "generate"}
        
        # 质量足够，完成
        print(f"{Style.BRIGHT}{Fore.YELLOW}💭【decision节点】 答案质量良好 ({answer_quality:.2f})，完成{Style.RESET_ALL}")
        return {"next_action": "finish"}
    
    return decision_node
