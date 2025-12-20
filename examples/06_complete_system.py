"""完整 Agentic RAG 系统测试"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from langchain_core.documents import Document
from src.agentic_rag.agentic_rag import AgenticRAG

load_dotenv()


def main():
    """主函数"""
    print("=" * 60)
    print("完整 Agentic RAG 系统测试")
    print("=" * 60)
    
    # 1. 初始化系统
    print("\n[步骤 1] 初始化 Agentic RAG 系统...")
    rag = AgenticRAG(
        model_name="gpt-3.5-turbo",
        max_iterations=5,
        persist_directory="./data/chroma_db/agentic_rag"
    )
    
    # 2. 准备文档
    print("\n[步骤 2] 加载文档...")
    sample_docs = [
        Document(page_content="""
        LangGraph 是 LangChain 的一个扩展库，专门用于构建有状态的、多参与者的应用程序。
        
        核心特性：
        1. 基于图的执行模型：使用节点和边来定义工作流
        2. 状态管理：自动管理和传递状态
        3. 循环支持：支持条件循环和迭代
        4. 检查点：可以保存和恢复执行状态
        
        LangGraph 特别适合构建：
        - 聊天机器人
        - 多步骤推理系统
        - Agentic RAG 系统
        - 复杂的工作流应用
        """),
        Document(page_content="""
        传统 RAG 系统使用线性流程：用户问题 → 向量化 → 检索相似文档 → 拼接上下文 → LLM 生成回答。
        
        传统 RAG 的局限性：
        1. 检索质量依赖查询质量：如果初始查询不够好，检索结果就会很差
        2. 无法处理复杂查询：无法分解多步骤问题
        3. 缺乏反馈机制：检索失败时无法自我调整
        4. 上下文利用有限：无法根据生成的中间结果进行动态检索
        """),
        Document(page_content="""
        Agentic RAG 将 Agent（智能体）的思想引入 RAG 系统，使其能够：
        1. 主动决策：根据当前状态决定下一步行动
        2. 迭代优化：可以多轮检索和生成
        3. 工具调用：可以使用多种工具（检索器、计算器、代码执行器等）
        4. 自我反思：能够评估结果质量并进行改进
        
        Agentic RAG 的核心组件：
        - 决策引擎：决定下一步应该做什么
        - 检索器：多种检索策略，根据上下文调整
        - 生成器：基于检索结果生成回答，可改进
        - 评估器：评估检索和生成的质量
        - 状态管理器：维护对话历史和中间结果
        """),
        Document(page_content="""
        LangGraph 与传统 LangChain 的主要区别：
        - LangChain 主要面向链式调用，适合简单的顺序流程
        - LangGraph 支持更复杂的控制流，包括循环和条件分支
        - LangGraph 基于图的执行模型，更适合构建 Agentic 系统
        - LangGraph 提供了内置的状态管理和检查点功能
        
        使用 LangGraph 构建 Agentic RAG 的优势：
        1. 清晰的状态管理
        2. 灵活的流程控制
        3. 易于调试和可视化
        4. 支持复杂的决策逻辑
        """)
    ]
    
    # 3. 构建向量数据库
    print("\n[步骤 3] 构建向量数据库...")
    rag.build_vectorstore(sample_docs)
    
    # 4. 测试查询
    print("\n[步骤 4] 测试查询...")
    
    test_questions = [
        "LangGraph 的核心特性是什么？",
        "传统 RAG 有什么局限性？",
        "Agentic RAG 如何改进传统 RAG？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(test_questions)}")
        print(f"{'='*60}\n")
        
        result = rag.query(question, verbose=True)
        
        print(f"\n最终答案:\n{result['answer']}\n")
        
        if result.get("retrieval_history"):
            print(f"检索轮数: {len(result['retrieval_history'])}")
            for j, docs in enumerate(result["retrieval_history"], 1):
                print(f"  第 {j} 轮检索到 {len(docs)} 个文档块")
        
        print(f"\n质量评分:")
        print(f"  检索质量: {result.get('retrieval_quality', 0.0):.2f}")
        print(f"  答案质量: {result.get('answer_quality', 0.0):.2f}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
