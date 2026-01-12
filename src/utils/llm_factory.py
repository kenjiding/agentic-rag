"""LLM 工厂函数 - 统一创建和管理 LLM 实例

2025-2026 企业级最佳实践：
- 统一的 LLM 创建接口
- 支持配置化和环境变量
- 类型安全
- 便于测试和切换模型
"""
from typing import Optional
from langchain_core.language_models import BaseChatModel
import os
import logging

from src.agentic_rag.llm import LLM

logger = logging.getLogger(__name__)

# 延迟加载默认配置（避免模块导入时的环境变量读取问题）
def _get_default_model_name() -> str:
    """获取默认模型名称（延迟加载环境变量）"""
    return os.getenv("LLM_MODEL_NAME", "openai:gpt-4o-mini")

def _get_default_temperature() -> float:
    """获取默认温度（延迟加载环境变量）"""
    return float(os.getenv("LLM_TEMPERATURE", "0.1"))


def create_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> BaseChatModel:
    """创建 LLM 实例的统一工厂函数
    
    这是系统中创建 LLM 实例的推荐方式。支持：
    - 环境变量配置
    - 参数覆盖
    - 类型安全
    
    Args:
        model_name: 模型名称，格式为 provider:model_name
                   如果为 None，使用环境变量 LLM_MODEL_NAME 或默认值
        temperature: 温度参数，如果为 None，使用环境变量 LLM_TEMPERATURE 或默认值
        **kwargs: 其他模型特定参数
        
    Returns:
        BaseChatModel 实例，可直接用于 LangChain
        
    Examples:
        # 使用默认配置
        llm = create_llm()
        
        # 指定模型
        llm = create_llm(model_name="anthropic:claude-3-5-sonnet-20241022")
        
        # 指定温度和模型
        llm = create_llm(model_name="openai:gpt-4o", temperature=0.7)
        
        # 使用自定义 API key
        llm = create_llm(model_name="openai:gpt-4o-mini", api_key="sk-...")
    """
    # 使用参数或环境变量或默认值（延迟加载）
    final_model_name = model_name or _get_default_model_name()
    final_temperature = temperature if temperature is not None else _get_default_temperature()
    
    logger.debug(f"创建 LLM 实例: {final_model_name}, temperature={final_temperature}")
    
    llm_wrapper = LLM(
        model_name=final_model_name,
        temperature=final_temperature,
        **kwargs
    )
    
    return llm_wrapper.get_llm()


def get_default_llm() -> BaseChatModel:
    """获取默认 LLM 实例（向后兼容）"""
    return create_llm()


def create_llm_for_intent_classification(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None
) -> BaseChatModel:
    """创建用于意图分类的 LLM 实例
    
    意图分类通常需要更低的温度以获得更稳定的结果。
    
    Args:
        model_name: 模型名称，如果为 None 使用默认值
        temperature: 温度参数，默认 0.0（更稳定）
        
    Returns:
        BaseChatModel 实例
    """
    final_temperature = temperature if temperature is not None else 0.0
    return create_llm(
        model_name=model_name or os.getenv("INTENT_LLM_MODEL_NAME", _get_default_model_name()),
        temperature=final_temperature
    )


def create_llm_for_agent(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None
) -> BaseChatModel:
    """创建用于 Agent 的 LLM 实例
    
    Agent 通常需要适中的温度以平衡创造性和稳定性。
    
    Args:
        model_name: 模型名称，如果为 None 使用默认值
        temperature: 温度参数，默认 0.1
        
    Returns:
        BaseChatModel 实例
    """
    return create_llm(
        model_name=model_name,
        temperature=temperature
    )


def create_llm_for_rag(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None
) -> BaseChatModel:
    """创建用于 RAG 的 LLM 实例
    
    RAG 生成通常需要较低的温度以保持准确性。
    
    Args:
        model_name: 模型名称，如果为 None 使用默认值
        temperature: 温度参数，默认 0.1
        
    Returns:
        BaseChatModel 实例
    """
    return create_llm(
        model_name=model_name,
        temperature=temperature
    )