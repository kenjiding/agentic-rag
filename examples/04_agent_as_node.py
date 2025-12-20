"""示例：LangChain Agent 作为 Node vs 普通函数"""

from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


# === 状态定义 ===
class ProcessingState(TypedDict):
    input: str
    processed_data: str
    result: str


# === 示例 1：使用普通函数作为 Node ===
def simple_process_node(state: ProcessingState) -> ProcessingState:
    """
    普通函数节点：简单、直接、高效
    
    适用场景：
    - 逻辑简单明确
    - 不需要 LLM 推理
    - 需要精确控制
    """
    input_text = state["input"]
    # 简单的字符串处理
    processed = input_text.upper().strip()
    return {"processed_data": processed}


# === 示例 2：使用 LangChain Agent 作为 Node ===

# 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网络工具（示例）"""
    return f"搜索结果: {query}"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "计算错误"

def create_agent_node(llm: ChatOpenAI):
    """
    创建使用 Agent 的节点
    
    适用场景：
    - 需要复杂的工具调用
    - 需要 LLM 驱动的决策
    - 需要多步骤推理
    """
    # 创建 Agent
    tools = [search_web, calculate]
    agent = create_react_agent(llm, tools)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def agent_node(state: ProcessingState) -> ProcessingState:
        """
        Agent 节点：使用 LangChain Agent 处理复杂任务
        
        特点：
        - Agent 可以自动决定调用哪些工具
        - 可以进行多步推理
        - LLM 驱动的决策过程
        """
        # 调用 Agent 处理输入
        agent_result = executor.invoke({"input": state["input"]})
        
        return {
            "result": agent_result["output"],
            "processed_data": state.get("processed_data", "")
        }
    
    return agent_node


# === 示例 3：混合使用 ===
def create_hybrid_graph():
    """
    混合使用普通函数和 Agent 的图
    
    最佳实践：
    - 简单逻辑用普通函数
    - 复杂推理用 Agent
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # 创建 Agent 节点
    agent_node = create_agent_node(llm)
    
    # 创建图
    graph = StateGraph(ProcessingState)
    
    # 添加节点
    graph.add_node("simple_process", simple_process_node)  # 普通函数
    graph.add_node("agent_process", agent_node)            # Agent
    
    # 设置流程
    graph.set_entry_point("simple_process")
    
    # 条件边：决定是否需要 Agent 处理
    def should_use_agent(state: ProcessingState) -> Literal["agent", "end"]:
        """简单规则：如果输入包含问号，使用 Agent"""
        if "?" in state.get("input", ""):
            return "agent"
        return "end"
    
    graph.add_conditional_edges(
        "simple_process",
        should_use_agent,
        {
            "agent": "agent_process",
            "end": END
        }
    )
    
    graph.add_edge("agent_process", END)
    
    return graph.compile()


# === 使用示例 ===
def main():
    print("=" * 60)
    print("LangChain Agent 作为 Node 示例")
    print("=" * 60)
    
    # 示例 1：只使用普通函数
    print("\n1. 只使用普通函数：")
    simple_graph = StateGraph(ProcessingState)
    simple_graph.add_node("process", simple_process_node)
    simple_graph.set_entry_point("process")
    simple_graph.add_edge("process", END)
    simple_graph = simple_graph.compile()
    
    result1 = simple_graph.invoke({"input": "hello world"})
    print(f"输入: hello world")
    print(f"输出: {result1}")
    
    # 示例 2：混合使用
    print("\n2. 混合使用（普通函数 + Agent）：")
    hybrid_graph = create_hybrid_graph()
    
    result2 = hybrid_graph.invoke({"input": "hello world"})
    print(f"输入: hello world (简单处理)")
    print(f"输出: {result2}")
    
    # 注意：如果要测试 Agent，需要有效的 API key
    # result3 = hybrid_graph.invoke({"input": "2+2等于多少?"})
    # print(f"输入: 2+2等于多少? (使用 Agent)")
    # print(f"输出: {result3}")


if __name__ == "__main__":
    main()
