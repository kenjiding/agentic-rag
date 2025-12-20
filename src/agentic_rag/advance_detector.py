from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import json
import numpy as np
from src.agentic_rag.threshold_config import ThresholdConfig, DetectorThresholds

class MultilingualNeedsMoreInfoDetector:
    """多语言的信息需求检测器"""
    
    def __init__(
        self, 
        embeddings: OpenAIEmbeddings,
        threshold_config: Optional[ThresholdConfig] = None
    ):
        self.embeddings = embeddings
        # 获取检测器阈值配置
        self.detector_thresholds = threshold_config.detector if threshold_config else DetectorThresholds()
        
        # 多语言的不确定表述（嵌入向量）
        self.uncertain_phrases = [
            # 中文
            "不知道", "无法确定", "信息不足", "不清楚", "没有找到相关信息",
            # 英文
            "I don't know", "cannot determine", "insufficient information", 
            "unclear", "not found", "lacking information", "need more information",
            # 其他语言可以继续添加
        ]
        self._uncertain_embeddings = None
    
    def _get_uncertain_embeddings(self):
        """获取不确定表述的嵌入向量（懒加载）"""
        if self._uncertain_embeddings is None:
            self._uncertain_embeddings = self.embeddings.embed_documents(self.uncertain_phrases)
        return self._uncertain_embeddings
    
    def needs_more_information(
        self, 
        answer: str, 
        retrieved_docs: List[Document],
        threshold: Optional[float] = None
    ) -> bool:
        """
        使用语义相似度判断是否需要更多信息
        
        注意：不检查答案与文档的相似度，因为：
        1. answer_quality 中的 accuracy 已经评估了答案是否基于上下文
        2. retrieved_docs 已经通过 embedding 相似度检索出来
        3. 重复计算 embedding 是冗余的
        
        Args:
            answer: 当前答案
            retrieved_docs: 已检索的文档（用于长度检查，不用于相似度计算）
            threshold: 相似度阈值（如果为None，使用配置的阈值）
            
        Returns:
            是否需要更多信息
        """
        # 使用配置的阈值（如果未提供）
        if threshold is None:
            threshold = self.detector_thresholds.embedding_similarity_threshold
        
        # 1. 快速检查：答案长度（使用配置的最小长度）
        if len(answer.strip()) < self.detector_thresholds.min_answer_length_embedding:
            return True
        
        # 2. 检查答案与不确定表述的语义相似度
        # 这是核心功能：检测答案是否表达了不确定性
        answer_embedding = np.array(self.embeddings.embed_query(answer))
        uncertain_embeddings = np.array(self._get_uncertain_embeddings())
        
        # 计算余弦相似度
        similarities = np.dot(uncertain_embeddings, answer_embedding) / (
            np.linalg.norm(uncertain_embeddings, axis=1) * np.linalg.norm(answer_embedding)
        )
        
        # 如果有很高的相似度，说明答案表达了不确定性
        max_similarity = np.max(similarities)
        if max_similarity > threshold:
            return True
        
        # 注意：不再检查 answer 和 retrieved_docs 的相似度
        # 因为 answer_quality.accuracy 已经在 generate_node 中评估了答案是否基于上下文
        # 这样可以避免重复计算 embedding，提高效率
        
        return False


class AdvancedNeedsMoreInfoDetector:
    """高级多语言信息需求检测器"""
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        vectorstore: Optional[Chroma] = None,
        use_llm: bool = True,
        use_embedding: bool = True,
        threshold_config: Optional[ThresholdConfig] = None
    ):
        """
        Args:
            llm: 用于LLM判断的模型
            vectorstore: 向量数据库（用于获取嵌入模型）
            use_llm: 是否使用LLM判断（更准确但更慢）
            use_embedding: 是否使用嵌入向量判断（更快但可能不够精确）
            threshold_config: 阈值配置
        """
        self.llm = llm
        self.embeddings = vectorstore._embedding_function if vectorstore else None
        self.use_llm = use_llm
        self.use_embedding = use_embedding
        self.threshold_config = threshold_config
        
        # 获取检测器阈值配置
        self.detector_thresholds = threshold_config.detector if threshold_config else DetectorThresholds()
        
        # 初始化嵌入向量检测器
        if self.use_embedding and self.embeddings:
            self.embedding_detector = MultilingualNeedsMoreInfoDetector(
                self.embeddings,
                threshold_config=threshold_config
            )
    
    def needs_more_information(
        self,
        answer: str,
        retrieved_docs: List[Document],
        question: Optional[str] = None,
        answer_quality: Optional[float] = None
    ) -> bool:
        """
        判断是否需要更多信息（混合方法，优化版）
        
        策略（按优先级）：
        1. 快速检查：答案长度、基本启发式
        2. 语义检查：使用嵌入向量检测不确定表述（快速，多语言支持好）
        3. LLM检查：如果需要更准确判断（较慢但更准确）
        
        注意：不检查 answer 与 retrieved_docs 的相似度，因为：
        - answer_quality.accuracy 已经在 generate_node 中评估了答案是否基于上下文
        - 避免重复计算 embedding，提高效率
        
        Args:
            answer: 当前答案
            retrieved_docs: 已检索的文档
            question: 原始问题（用于LLM检查）
            answer_quality: 答案质量分数（可选，用于快速判断）
        """
        # 快速检查1：答案长度（使用配置的最小长度）
        if len(answer.strip()) < self.detector_thresholds.min_answer_length_quick:
            return True
        
        # 快速检查2：如果答案质量很低，很可能需要更多信息
        # 但这里不直接返回，因为我们需要判断是"信息不足"还是"生成质量问题"
        if answer_quality is not None and answer_quality < self.detector_thresholds.low_quality_threshold:
            # 质量极低时，优先检查是否是不确定表述
            if self._quick_heuristic_check(answer):
                return True
        
        # 快速检查3：基本启发式（多语言关键词）
        if self._quick_heuristic_check(answer):
            return True
        
        # 方法1：嵌入向量检查（快速，多语言支持好）
        # 只检查不确定表述，不检查与文档的相似度
        if self.use_embedding and self.embedding_detector and isinstance(self.embedding_detector, MultilingualNeedsMoreInfoDetector):
            embedding_result = self.embedding_detector.needs_more_information(
                answer, retrieved_docs, threshold=self.detector_thresholds.embedding_similarity_threshold
            )
            # 如果嵌入向量判断明确，直接返回
            if embedding_result:
                return True
        
        # 方法2：LLM检查（准确，但需要question上下文）
        if self.use_llm and self.llm and question:
            try:
                llm_result = self._llm_check(answer, question, retrieved_docs)
                return llm_result
            except Exception as e:
                print(f"[警告] LLM检查失败，使用嵌入向量结果: {e}")
                # 如果LLM失败，回退到嵌入向量结果（使用回退阈值）
                if self.use_embedding and hasattr(self, 'embedding_detector'):
                    return self.embedding_detector.needs_more_information(
                        answer, retrieved_docs, threshold=self.detector_thresholds.embedding_similarity_threshold_fallback
                    )
        
        # 默认返回False
        return False
    
    def _quick_heuristic_check(self, answer: str) -> bool:
        """快速启发式检查（多语言关键词）"""
        answer_lower = answer.lower()
        
        # 多语言不确定表述（更全面的列表）
        uncertain_patterns = [
            # 中文
            "不知道", "无法确定", "信息不足", "不清楚", "没有找到",
            "无法回答", "不能确定", "缺少信息",
            # 英文
            "i don't know", "don't know", "cannot determine", "cannot tell",
            "insufficient information", "lack of information", "not enough information",
            "unclear", "unsure", "uncertain", "not found", "no information",
            "need more", "requires more", "additional information needed",
            # 法语
            "je ne sais pas", "ne peut pas déterminer", "information insuffisante",
            # 德语
            "weiß nicht", "kann nicht bestimmen", "unzureichende information",
            # 日语
            "わかりません", "確定できません", "情報不足",
        ]
        
        return any(pattern in answer_lower for pattern in uncertain_patterns)
    
    def _llm_check(
        self, 
        answer: str, 
        question: str, 
        retrieved_docs: List[Document]
    ) -> bool:
        """使用LLM进行准确判断"""
        template = """判断给定的答案是否表明需要更多信息才能正确回答问题。

问题: {question}

答案: {answer}

已检索文档数: {doc_count}

请判断答案是否表现出不确定性、信息不足或需要更多上下文。

只返回JSON格式：
{{
    "needs_more_info": true/false,
    "confidence": 0.0-1.0,
    "reason": "简要说明（使用与答案相同的语言）"
}}

需要更多信息的答案示例：
- "I don't know" / "我不知道"
- "Insufficient information" / "信息不足"
- "Cannot determine" / "无法确定"
- 非常简短的答案（<30词）且未回答问题

不需要更多信息的答案示例：
- 包含具体信息的详细答案
- 即使提到限制但已完整回答的问题"""
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "doc_count": len(retrieved_docs)
        })
        
        result = json.loads(response)
        return result.get("needs_more_info", False)