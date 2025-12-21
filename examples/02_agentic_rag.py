"""Agentic RAG 基础示例"""
import sys
from pathlib import Path
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_absolute_path
from src.agentic_rag.docs_parser import DocsParser
from dotenv import load_dotenv
from langchain_core.documents import Document
from src.agentic_rag.agentic_rag import AgenticRAG
from src.agentic_rag.parser import PDFParser

# 加载环境变量
load_dotenv()


def main():
    """主函数"""
    print("=" * 60)
    print("Agentic RAG 系统示例")
    print("=" * 60)
    print("\n本示例展示 Agentic RAG 如何通过迭代优化改进检索和生成质量。")
    
    # 1. 初始化 Agentic RAG 系统
    print("\n[步骤 1] 初始化 Agentic RAG 系统...")
    rag = AgenticRAG(
        model_name="gpt-4o-mini",
        max_iterations=5,
        persist_directory="./tmp/chroma_db/agentic_rag"
    )

    current_file = os.path.dirname(__file__)
    # md_file = get_absolute_path(current_file, "../README.md")
    # docs_parser = DocsParser(file_path=md_file)

    # pdf_path = get_absolute_path(current_file, "../kenjiding.pdf")
    # pdf_parser = DocsParser(file_path=pdf_path)
    
    # # 2. 准备文档
    # print("\n[步骤 2] 加载文档...")
    # sample_docs = [
    #     *pdf_parser.docs,
    #     *docs_parser.docs,
    #     Document(page_content="""
    #     LangGraph 是 LangChain 的一个扩展库，专门用于构建有状态的、多参与者的应用程序。
        
    #     核心特性：
    #     1. 基于图的执行模型：使用节点和边来定义工作流
    #     2. 状态管理：自动管理和传递状态
    #     3. 循环支持：支持条件循环和迭代
    #     4. 检查点：可以保存和恢复执行状态
        
    #     LangGraph 特别适合构建：
    #     - 聊天机器人
    #     - 多步骤推理系统
    #     - Agentic RAG 系统
    #     - 复杂的工作流应用
    #     """),
    #     Document(page_content="""
    #     传统 RAG 系统使用线性流程：用户问题 → 向量化 → 检索相似文档 → 拼接上下文 → LLM 生成回答。
        
    #     传统 RAG 的局限性：
    #     1. 检索质量依赖查询质量：如果初始查询不够好，检索结果就会很差
    #     2. 无法处理复杂查询：无法分解多步骤问题
    #     3. 缺乏反馈机制：检索失败时无法自我调整
    #     4. 上下文利用有限：无法根据生成的中间结果进行动态检索
    #     """),
    #     Document(page_content="""
    #     Agentic RAG 将 Agent（智能体）的思想引入 RAG 系统，使其能够：
    #     1. 主动决策：根据当前状态决定下一步行动
    #     2. 迭代优化：可以多轮检索和生成
    #     3. 工具调用：可以使用多种工具（检索器、计算器、代码执行器等）
    #     4. 自我反思：能够评估结果质量并进行改进
        
    #     Agentic RAG 的核心组件：
    #     - 决策引擎：决定下一步应该做什么
    #     - 检索器：多种检索策略，根据上下文调整
    #     - 生成器：基于检索结果生成回答，可改进
    #     - 评估器：评估检索和生成的质量
    #     - 状态管理器：维护对话历史和中间结果
    #     """)
    # ]
    
    # pdf_path = get_absolute_path(current_file, "../uber_10q_march_2022_page26.pdf")
    # pdf_parser = PDFParser()
    # pdf_chunks = pdf_parser.parse_pdf_to_documents(pdf_path, refresh=True)
    # pdf_parser = DocsParser(file_path=pdf_path)
    # # 3. 构建向量数据库
    # print("\n[步骤 3] 构建向量数据库...")
    # # rag.build_vectorstore(sample_docs)
    # rag.add_documents(pdf_chunks)
    
    # 4. 测试查询 - 展示迭代优化
    print("\n[步骤 4] 测试查询（展示 Agentic RAG 的迭代优化能力）...")
    questions = [
        "Uber 2021年和2022年Legal, tax, and regulatory reserve changes and settlements 业务的调整后EBITDA分别是多少?",
        # "2022年福布斯富豪榜杰夫·贝索斯财富是多少?",
        # "2019年福布斯富豪榜杰夫·贝索斯财富是多少?",
        # "2019年, 2020,2021年福布斯富豪榜杰夫·贝索斯财富是上升了还是下降了? 请给出具体数据.",
        # "kenjiding的low code项目是在哪家公司做的?",
        # "kenjiding有哪些公司工作过?",
        # "据你的了解,kenjiding最厉害的经历是哪些?",
        # "gentic-agent项目结构是怎样的?",
        # "LangGraph 的核心特性是什么？",
        # "传统 RAG 有什么局限性？",
        # "Agentic RAG 如何改进传统 RAG？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(questions)}: {question}")
        print(f"{'='*60}\n")
        
        result = rag.query(question, verbose=True)
        
        print(f"\n📊 执行统计:")
        print(f"  总迭代次数: {result.get('iteration_count', 0)}")
        print(f"  检索轮数: {len(result.get('retrieval_history', []))}")
        print(f"  最终检索质量: {result.get('retrieval_quality', 0.0):.2f}")
        print(f"  最终答案质量: {result.get('answer_quality', 0.0):.2f}")
        
        print(f"\n💡 最终答案:")
        print(f"{result['answer']}\n")
        
        # 展示检索历史
        if result.get("retrieval_history"):
            print("📚 检索历史:")
            for j, docs in enumerate(result["retrieval_history"], 1):
                print(f"  第 {j} 轮: 检索到 {len(docs)} 个文档块")
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Agentic RAG 示例完成！")
    print("\n💡 观察要点:")
    print("1. Agentic RAG 会根据质量自动决定是否需要更多迭代")
    print("2. 如果检索质量不够，系统会自动尝试不同的检索策略")
    print("3. 如果答案质量不够，系统会尝试改进或获取更多信息")
    print("=" * 60)


if __name__ == "__main__":
    main()