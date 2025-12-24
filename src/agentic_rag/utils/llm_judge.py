"""LLM判断工具 - 企业级最佳实践

本模块提供通用的LLM判断功能，避免在多个地方重复实现。
使用with_structured_output确保输出格式正确，支持多种判断场景。

2025-2026 最佳实践：
- 统一的LLM判断接口
- 使用结构化输出确保格式正确
- 支持多种判断场景
- 错误处理和回退机制
"""
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class CoverageResult(BaseModel):
    """文档覆盖查询的判断结果"""
    is_covered: bool = Field(..., description="文档是否覆盖了查询")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    reason: str = Field(..., description="判断原因")


class ValidationResult(BaseModel):
    """验证结果结构"""
    is_valid: bool = Field(..., description="是否有效/合理")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    reason: str = Field(..., description="判断原因")


class LLMJudge:
    """LLM判断工具 - 企业级最佳实践
    
    提供统一的LLM判断接口，支持多种判断场景。
    使用with_structured_output确保输出格式正确。
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        初始化LLM判断工具
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
    
    def check_coverage(
        self,
        query: str,
        documents: List[Document],
        max_docs: int = 5,
        max_doc_length: int = 500,
        confidence_threshold: float = 0.5
    ) -> bool:
        """
        使用LLM判断文档是否覆盖查询（企业级最佳实践）
        
        Args:
            query: 查询文本
            documents: 文档列表
            max_docs: 最多检查的文档数量
            max_doc_length: 每个文档的最大长度
            confidence_threshold: 置信度阈值
            
        Returns:
            如果文档覆盖了查询，返回True
        """
        if not self.llm or not documents:
            return False
        
        try:
            # 构建文档内容摘要
            doc_contents = "\n\n".join([
                f"[文档 {i+1}]\n{doc.page_content[:max_doc_length]}"
                for i, doc in enumerate(documents[:max_docs])
            ])
            
            template = """判断以下文档是否包含回答查询所需的信息。

查询: {query}

文档内容:
{doc_contents}

请判断文档是否包含足够的信息来回答查询。考虑语义相关性，而不仅仅是关键词匹配。"""
            
            prompt = ChatPromptTemplate.from_template(template)
            structured_llm = self.llm.with_structured_output(CoverageResult)
            chain = prompt | structured_llm
            
            result = chain.invoke({
                "query": query,
                "doc_contents": doc_contents
            })
            
            # 如果置信度较高，返回判断结果
            return result.is_covered and result.confidence > confidence_threshold
            
        except Exception as e:
            # LLM判断失败，返回False（保守策略）
            return False
    
    def validate_rewritten_query(
        self,
        original_query: str,
        rewritten_query: str,
        confidence_threshold: float = 0.5
    ) -> bool:
        """
        使用LLM验证改写后的查询是否合理（企业级最佳实践）
        
        Args:
            original_query: 原始查询
            rewritten_query: 改写后的查询
            confidence_threshold: 置信度阈值
            
        Returns:
            如果改写合理，返回True
        """
        if not self.llm:
            return True  # 如果没有LLM，默认认为合理
        
        try:
            template = """判断改写后的查询是否保持了原始查询的核心意图。

原始查询: {original_query}
改写后的查询: {rewritten_query}

请判断改写后的查询是否：
1. 保持了原始查询的核心意图和主题
2. 没有偏离原意或改变问题本质
3. 只是优化了表述，而不是完全改变问题"""
            
            prompt = ChatPromptTemplate.from_template(template)
            structured_llm = self.llm.with_structured_output(ValidationResult)
            chain = prompt | structured_llm
            
            result = chain.invoke({
                "original_query": original_query,
                "rewritten_query": rewritten_query
            })
            
            # 如果置信度较高，返回判断结果
            return result.is_valid and result.confidence > confidence_threshold
            
        except Exception as e:
            # LLM判断失败，返回True（保守策略，允许改写）
            return True
    
    def judge(
        self,
        prompt_template: str,
        result_model: BaseModel,
        **kwargs
    ) -> BaseModel:
        """
        通用的LLM判断方法（企业级最佳实践）
        
        允许自定义prompt和结果模型，提供最大的灵活性。
        
        Args:
            prompt_template: 提示词模板
            result_model: 结果Pydantic模型
            **kwargs: 传递给prompt的参数
            
        Returns:
            判断结果（Pydantic模型实例）
        """
        if not self.llm:
            raise ValueError("LLM未初始化")
        
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            structured_llm = self.llm.with_structured_output(result_model)
            chain = prompt | structured_llm
            
            result = chain.invoke(kwargs)
            return result
            
        except Exception as e:
            raise RuntimeError(f"LLM判断失败: {str(e)}") from e

