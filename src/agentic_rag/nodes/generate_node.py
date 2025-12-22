"""生成节点实现"""
from typing import Optional
from colorama import Fore, Style

from src.agentic_rag.state import AgenticRAGState
from src.agentic_rag.generator import IntelligentGenerator
from src.agentic_rag.threshold_config import ThresholdConfig


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
            
            # 注意：iteration_count 应该在 decision_node 中管理，不在 generate_node 中增加
            # 这样可以准确反映决策循环的次数，而不是每个节点执行的次数
            
            return {
                "answer": answer,
                "generation_history": generation_history,
                "answer_quality": quality,
                "evaluation_feedback": feedback,
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

