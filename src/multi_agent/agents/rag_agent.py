"""Agentic RAG Agent - 封装RAG搜索能力

本模块将现有的AgenticRAG系统封装为多Agent框架中的一个Agent。
这样可以在多Agent系统中使用RAG的搜索和知识检索能力。
"""
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from src.multi_agent.agents.base_agent import BaseAgent
from src.multi_agent.state import MultiAgentState
from src.agentic_rag.agentic_rag import AgenticRAG
import logging

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """Agentic RAG Agent - 提供知识检索和搜索能力
    
    此Agent封装了AgenticRAG系统，提供以下能力：
    1. 从向量数据库中检索相关信息
    2. 基于检索结果生成高质量答案
    3. 支持多轮迭代优化检索和生成质量
    4. 支持Web搜索作为补充（Corrective RAG）
    
    使用场景：
    - 需要从知识库中检索信息的问题
    - 需要基于文档内容回答的问题
    - 需要多轮检索优化的问题
    
    2025-2026 最佳实践：
    - 封装现有系统，保持向后兼容
    - 提供统一的Agent接口
    - 支持异步执行（未来扩展）
    - 详细的日志记录
    """
    
    def __init__(
        self,
        rag_system: Optional[AgenticRAG] = None,
        llm: Optional[ChatOpenAI] = None,
        persist_directory: str = "./tmp/chroma_db/agentic_rag",
        model_name: str = "gpt-4o-mini",
        max_iterations: int = 3
    ):
        """
        初始化RAG Agent
        
        Args:
            rag_system: 已初始化的AgenticRAG实例，如果为None则创建新实例
            llm: 语言模型实例
            persist_directory: 向量数据库持久化目录
            model_name: 模型名称
            max_iterations: RAG最大迭代次数
        """
        super().__init__(
            name="rag_agent",
            llm=llm,
            description="专门用于从知识库中检索信息并生成答案的Agent。适用于需要基于文档内容回答的问题。"
        )
        
        # 初始化RAG系统
        if rag_system:
            self.rag_system = rag_system
        else:
            logger.info(f"初始化RAG系统，持久化目录: {persist_directory}")
            self.rag_system = AgenticRAG(
                model_name=model_name,
                max_iterations=max_iterations,
                persist_directory=persist_directory,
                skip_intent_classification=True  # 从multi_agent进入，已做过意图识别
            )
    
    async def execute(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        执行RAG搜索和生成
        
        从用户消息中提取问题，使用RAG系统检索和生成答案。
        
        Args:
            state: 当前的多Agent系统状态
            
        Returns:
            包含以下字段的字典：
            - result: RAG查询结果，包含answer、answer_quality等
            - messages: 新增的AI消息（包含答案）
            - metadata: 包含检索质量、迭代次数等元数据
        """
        try:
            # 从消息历史中提取最后一个用户问题
            user_message = None
            for msg in reversed(state.messages):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break
            
            if not user_message:
                return {
                    "result": None,
                    "messages": [],
                    "metadata": {"error": "未找到用户问题"},
                    "error": "未找到用户问题"
                }
            
            # 从state中提取意图识别结果
            # 注意：query_intent 已经通过 intent.model_dump() 包含了所有字段，包括 sub_queries
            query_intent = state.query_intent
            original_question = state.original_question or user_message
            
            # 确保sub_queries是列表格式（如果存在）
            if query_intent and query_intent.get("sub_queries") and not isinstance(query_intent["sub_queries"], list):
                query_intent = query_intent.copy()  # 避免修改原始状态
                query_intent["sub_queries"] = []
            
            logger.info(f"RAG Agent执行查询: {user_message}")
            if query_intent:
                logger.info(f"RAG Agent使用意图识别结果: {query_intent.get('intent_type', 'unknown')}")
                if query_intent.get("needs_decomposition"):
                    sub_queries = query_intent.get("sub_queries", [])
                    logger.info(f"RAG Agent检测到需要分解，子查询数: {len(sub_queries)}")
                    for i, sq in enumerate(sub_queries[:3], 1):  # 只打印前3个
                        sq_text = sq.get("query", str(sq)) if isinstance(sq, dict) else str(sq)
                        logger.info(f"  子查询 {i}: {sq_text[:50]}...")
            
            # 调用RAG系统查询，传入意图识别结果
            # 使用original_question作为主查询，query_intent中包含sub_queries用于检索
            rag_result = self.rag_system.query(
                question=original_question,  # 使用原始问题（用于生成最终答案）
                verbose=False,
                query_intent=query_intent  # 传入意图识别结果（包含sub_queries用于检索）
            )
            
            # 提取答案
            answer = rag_result.get("answer", "")
            if not answer:
                answer = "抱歉，我无法从知识库中找到相关信息。"
            
            # 创建AI消息
            ai_message = AIMessage(content=answer)
            
            # 构建返回结果
            result = {
                "result": rag_result,
                "messages": [ai_message],
                "metadata": {
                    "agent": self.name,
                    "question": user_message,
                    "answer_quality": rag_result.get("answer_quality", 0.0),
                    "retrieval_quality": rag_result.get("retrieval_quality", 0.0),
                    "iteration_count": rag_result.get("iteration_count", 0),
                    "retrieval_history_count": len(rag_result.get("retrieval_history", [])),
                    "web_search_used": rag_result.get("web_search_used", False)
                }
            }
            
            logger.info(f"RAG Agent执行完成，答案质量: {rag_result.get('answer_quality', 0.0):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"RAG Agent执行错误: {str(e)}", exc_info=True)
            return {
                "result": None,
                "messages": [AIMessage(content=f"执行RAG搜索时出现错误: {str(e)}")],
                "metadata": {"error": str(e)},
                "error": str(e)
            }

