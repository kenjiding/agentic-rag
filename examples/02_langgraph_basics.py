"""LangGraph 基础示例"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END


# 1. 定义状态
class CounterState(TypedDict):
    """计数器状态"""
    counter: int
    message: str


# 2. 定义节点函数
def increment_node(state: CounterState) -> CounterState:
    """增加计数器"""
    new_counter = state["counter"] + 1
    print(f"计数器增加到: {new_counter}")
    return {"counter": new_counter}


def decrement_node(state: CounterState) -> CounterState:
    """减少计数器"""
    new_counter = state["counter"] - 1
    print(f"计数器减少到: {new_counter}")
    return {"counter": new_counter}


def print_node(state: CounterState) -> CounterState:
    """打印当前状态"""
    print(f"当前计数器值: {state['counter']}")
    return state


# 3. 定义条件函数（用于条件边）
def should_continue(state: CounterState) -> Literal["increment", "decrement", "end"]:
    """决定下一步"""
    counter = state["counter"]
    
    if counter < 5:
        return "increment"
    elif counter > 0:
        return "decrement"
    else:
        return "end"


# 4. 构建图
def create_counter_graph():
    """创建计数器图"""
    # 创建状态图
    graph = StateGraph(CounterState)
    
    # 添加节点
    graph.add_node("increment", increment_node)
    graph.add_node("decrement", decrement_node)
    graph.add_node("print", print_node)
    
    # 设置入口点
    graph.set_entry_point("print")
    
    # 添加条件边
    graph.add_conditional_edges(
        "print",
        should_continue,
        {
            "increment": "increment",
            "decrement": "decrement",
            "end": END
        }
    )
    
    # 添加循环边
    graph.add_edge("increment", "print")
    graph.add_edge("decrement", "print")
    
    # 编译图
    return graph.compile()


def main():
    """主函数"""
    print("=" * 60)
    print("LangGraph 基础示例：计数器")
    print("=" * 60)
    
    # 创建图
    graph = create_counter_graph()
    
    # 初始状态
    initial_state = {
        "counter": 0,
        "message": "开始"
    }
    
    # 运行图
    print("\n开始执行图...\n")
    final_state = graph.invoke(initial_state)
    
    print(f"\n最终状态: {final_state}")
    print("=" * 60)


if __name__ == "__main__":
    main()
