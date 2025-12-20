"""Agentic RAG 完整图实现"""
from agentic_rag.advance_detector import AdvancedNeedsMoreInfoDetector
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from typing import Optional
from src.agentic_rag.state import AgenticRAGState
from src.agentic_rag.retriever import IntelligentRetriever
from src.agentic_rag.generator import IntelligentGenerator
from src.agentic_rag.nodes import (
    create_intent_classification_node,
    create_retrieve_node,
    create_generate_node,
    create_decision_node
)
from src.agentic_rag.intent_analyse import IntentClassifier, QueryOptimizer
from src.agentic_rag.threshold_config import ThresholdConfig


def create_agentic_rag_graph(
    vectorstore: Chroma,
    llm: ChatOpenAI = None,
    max_iterations: int = 5,
    threshold_config: Optional[ThresholdConfig] = None
):
    """
    创建完整的 Agentic RAG 图
    
    Args:
        vectorstore: 向量数据库
        llm: LLM 实例
        max_iterations: 最大迭代次数
        threshold_config: 阈值配置（如果为None，使用默认配置）
        
    Returns:
        编译后的图
    """
    # 使用默认配置如果未提供
    if threshold_config is None:
        threshold_config = ThresholdConfig.default()
    
    # 初始化组件（传递 threshold_config）
    if llm is None:
        # 使用配置的生成器温度
        generator_temp = threshold_config.generator.default_temperature
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=generator_temp)
    
    retriever = IntelligentRetriever(
        vectorstore=vectorstore, 
        llm=llm,
        threshold_config=threshold_config
    )
    generator = IntelligentGenerator(
        llm=llm,
        threshold_config=threshold_config
    )
    detector = AdvancedNeedsMoreInfoDetector(
        llm=llm,
        vectorstore=vectorstore,
        use_llm=True,  # 如果llm可用，使用更准确的LLM判断
        use_embedding=True,  # 同时使用嵌入向量作为快速检查
        threshold_config=threshold_config
    )
    
    # 创建意图分类器（如果启用）
    intent_classifier = None
    if threshold_config.intent_classification.enable_intent_classification:
        intent_classifier = IntentClassifier(
            llm=llm,
            threshold_config=threshold_config
        )
    
    query_optimizer = QueryOptimizer(llm=llm)
    
    # 创建节点（传递 threshold_config）
    intent_node = None
    if intent_classifier:
        intent_node = create_intent_classification_node(intent_classifier, threshold_config=threshold_config)
    retrieve_node = create_retrieve_node(retriever, threshold_config=threshold_config)
    generate_node = create_generate_node(generator, threshold_config=threshold_config)
    decision_node = create_decision_node(
      detector,
      query_optimizer=query_optimizer,
      threshold_config=threshold_config)
    
    # 创建状态图
    graph = StateGraph(AgenticRAGState)
    
    # 添加节点
    if intent_node:
        graph.add_node("intent", intent_node)
    graph.add_node("decision", decision_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    # 设置入口点：如果启用意图识别，先进行意图识别，否则直接进入决策节点
    if intent_node:
        graph.set_entry_point("intent")
        # 意图识别后进入决策节点
        graph.add_edge("intent", "decision")
    else:
        graph.set_entry_point("decision")
    
    # 添加条件边
    graph.add_conditional_edges(
        "decision",
        lambda state: state.get("next_action", "finish"),
        {
            "retrieve": "retrieve",
            "generate": "generate",
            "finish": END
        }
    )
    
    # 检索后回到决策节点
    graph.add_edge("retrieve", "decision")
    
    # 生成后回到决策节点
    graph.add_edge("generate", "decision")
    
    # 编译图
    compiled_graph = graph.compile()
    # compiled_graph.get_graph().draw_mermaid_png(output_file="graph.png")
    
    return compiled_graph


def create_initial_state(
    question: str,
    max_iterations: int = 5
) -> AgenticRAGState:
    """
    创建初始状态
    
    Args:
        question: 用户问题
        max_iterations: 最大迭代次数
        
    Returns:
        初始状态
    """
    return {
        "question": question,
        "query_intent": None,  # 意图识别结果（将在intent节点中填充）
        "retrieved_docs": [],
        "retrieval_history": [],
        "retrieval_quality": 0.0,
        "retrieval_strategy": "semantic",
        "answer": "",
        "generation_history": [],
        "answer_quality": 0.0,
        "evaluation_feedback": "",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "next_action": None,
        "error_message": "",
        "tools_used": []
    }
