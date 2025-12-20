"""Agentic RAG 测试"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from src.agentic_rag.agentic_rag import AgenticRAG
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def rag_system():
    """创建测试用的 RAG 系统"""
    rag = AgenticRAG(max_iterations=3)
    
    # 加载测试文档
    test_docs = [
        Document(page_content="LangGraph 是一个用于构建 Agent 的库。它基于图模型，支持状态管理和循环。"),
        Document(page_content="传统 RAG 系统有局限性：检索质量依赖查询质量，无法处理复杂查询。"),
        Document(page_content="Agentic RAG 通过迭代优化、自我反思和工具调用改进了传统 RAG。")
    ]
    rag.build_vectorstore(test_docs)
    
    return rag


def test_basic_query(rag_system):
    """测试基本查询"""
    result = rag_system.query("LangGraph 是什么？", verbose=False)
    
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert result["iteration_count"] > 0


def test_retrieval_quality(rag_system):
    """测试检索质量"""
    result = rag_system.query("LangGraph 是什么？", verbose=False)
    
    assert result["retrieval_quality"] >= 0.0
    assert result["retrieval_quality"] <= 1.0


def test_iteration_limit(rag_system):
    """测试迭代限制"""
    result = rag_system.query("LangGraph 是什么？", verbose=False)
    
    assert result["iteration_count"] <= rag_system.max_iterations


def test_answer_quality(rag_system):
    """测试答案质量"""
    result = rag_system.query("传统 RAG 有什么局限性？", verbose=False)
    
    assert result["answer_quality"] >= 0.0
    assert result["answer_quality"] <= 1.0
    assert "answer" in result
    assert len(result["answer"]) > 0


def test_retrieval_history(rag_system):
    """测试检索历史"""
    result = rag_system.query("Agentic RAG 如何改进传统 RAG？", verbose=False)
    
    assert "retrieval_history" in result
    assert isinstance(result["retrieval_history"], list)


def test_error_handling(rag_system):
    """测试错误处理"""
    # 测试空问题（应该能处理，不崩溃）
    result = rag_system.query("", verbose=False)
    
    # 应该返回结果（可能为空或错误信息）
    assert isinstance(result, dict)
    assert "answer" in result or "error_message" in result


@pytest.mark.skip(reason="需要实际的向量数据库和API连接")
def test_vectorstore_persistence():
    """测试向量数据库持久化"""
    persist_dir = "./test_data/chroma_test"
    
    # 创建第一个系统并添加文档
    rag1 = AgenticRAG(persist_directory=persist_dir)
    docs = [Document(page_content="测试文档")]
    rag1.build_vectorstore(docs)
    
    # 创建第二个系统，应该能加载已存在的向量数据库
    rag2 = AgenticRAG(persist_directory=persist_dir)
    result = rag2.query("测试", verbose=False)
    
    assert "answer" in result