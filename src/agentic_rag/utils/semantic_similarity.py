"""语义相似度工具 - 企业级最佳实践

本模块提供通用的语义相似度计算功能，避免在多个地方重复实现。
支持embedding语义相似度计算，统一处理余弦相似度计算逻辑。

2025-2026 最佳实践：
- 统一的语义相似度计算接口
- 支持多种使用场景
- 错误处理和回退机制
- 性能优化（批量计算等）
"""
from typing import List, Optional, Union
import numpy as np
from langchain_core.documents import Document


class SemanticSimilarityCalculator:
    """语义相似度计算器 - 企业级最佳实践
    
    提供统一的语义相似度计算接口，避免代码重复。
    支持单文本、文本对、文本与文档列表等多种场景。
    """
    
    def __init__(self, embeddings):
        """
        初始化语义相似度计算器
        
        Args:
            embeddings: Embedding模型实例
        """
        self.embeddings = embeddings
    
    def calculate_similarity(
        self,
        text1: str,
        text2: str,
        normalize: bool = True
    ) -> float:
        """
        计算两个文本的语义相似度（企业级最佳实践）
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            normalize: 是否将相似度从[-1, 1]映射到[0, 1]
            
        Returns:
            语义相似度分数 (0-1 如果normalize=True，否则 -1到1)
        """
        try:
            # 获取embedding
            embedding1 = np.array(self.embeddings.embed_query(text1))
            embedding2 = np.array(self.embeddings.embed_query(text2))
            
        # 计算余弦相似度
        cosine_sim = self._cosine_similarity(embedding1, embedding2)
        
        # 如果需要，将[-1, 1]映射到[0, 1]
        # 注意：确保返回 Python 原生 float 类型，而不是 numpy 类型
        if normalize:
            return float((cosine_sim + 1) / 2)
        return float(cosine_sim)
            
        except Exception as e:
            # 如果计算失败，返回0（表示不相似）
            return 0.0
    
    def calculate_max_similarity(
        self,
        query: str,
        documents: List[Union[Document, str]],
        max_doc_length: int = 1000
    ) -> float:
        """
        计算查询与文档列表的最大语义相似度
        
        Args:
            query: 查询文本
            documents: 文档列表（可以是Document对象或字符串）
            max_doc_length: 每个文档的最大长度（用于性能优化）
            
        Returns:
            最大相似度分数 (0-1)
        """
        if not documents:
            return 0.0
        
        try:
            # 获取查询的embedding
            query_embedding = np.array(self.embeddings.embed_query(query))
            
            max_similarity = 0.0
            
            for doc in documents:
                # 提取文档内容
                if isinstance(doc, Document):
                    doc_content = doc.page_content[:max_doc_length]
                else:
                    doc_content = str(doc)[:max_doc_length]
                
                # 获取文档的embedding
                doc_embedding = np.array(self.embeddings.embed_query(doc_content))
                
                # 计算余弦相似度
                cosine_sim = self._cosine_similarity(query_embedding, doc_embedding)
                similarity = float((cosine_sim + 1) / 2)  # 映射到[0, 1]，确保是 Python float
                
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
            
        except Exception as e:
            return 0.0
    
    def calculate_similarities(
        self,
        query: str,
        documents: List[Union[Document, str]],
        max_doc_length: int = 1000
    ) -> List[float]:
        """
        计算查询与文档列表中每个文档的相似度
        
        Args:
            query: 查询文本
            documents: 文档列表
            max_doc_length: 每个文档的最大长度
            
        Returns:
            相似度分数列表
        """
        if not documents:
            return []
        
        try:
            # 获取查询的embedding
            query_embedding = np.array(self.embeddings.embed_query(query))
            
            similarities = []
            
            for doc in documents:
                # 提取文档内容
                if isinstance(doc, Document):
                    doc_content = doc.page_content[:max_doc_length]
                else:
                    doc_content = str(doc)[:max_doc_length]
                
                # 获取文档的embedding
                doc_embedding = np.array(self.embeddings.embed_query(doc_content))
                
                # 计算余弦相似度
                cosine_sim = self._cosine_similarity(query_embedding, doc_embedding)
                similarity = float((cosine_sim + 1) / 2)  # 映射到[0, 1]，确保是 Python float
                
                similarities.append(similarity)
            
            return similarities
            
        except Exception as e:
            return [0.0] * len(documents)
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        计算两个向量的余弦相似度（内部方法）
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            余弦相似度 (-1 到 1)
        """
        # 计算点积
        dot_product = np.dot(vec1, vec2)
        
        # 计算向量长度
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # 避免除零
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 计算余弦相似度，确保返回 Python float 类型
        return float(dot_product / (norm1 * norm2))
    
    def batch_calculate_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: List[np.ndarray]
    ) -> List[float]:
        """
        批量计算相似度（性能优化版本）
        
        当需要计算查询与多个文档的相似度时，可以预先计算embedding，
        然后批量计算相似度，提高性能。
        
        Args:
            query_embedding: 查询的embedding向量
            document_embeddings: 文档的embedding向量列表
            
        Returns:
            相似度分数列表
        """
        if not document_embeddings:
            return []
        
        try:
            query_vec = np.array(query_embedding)
            similarities = []
            
            for doc_vec in document_embeddings:
                doc_vec_array = np.array(doc_vec)
                cosine_sim = self._cosine_similarity(query_vec, doc_vec_array)
                similarity = float((cosine_sim + 1) / 2)  # 映射到[0, 1]，确保是 Python float
                similarities.append(similarity)
            
            return similarities
            
        except Exception as e:
            return [0.0] * len(document_embeddings)

