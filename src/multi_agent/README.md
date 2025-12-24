# 多Agent系统框架 - 2025-2026 企业级最佳实践

## 概述

本框架基于LangGraph 1.x构建，提供了一个可扩展、可维护的多Agent智能体系统。

## 核心特性

### 1. Supervisor模式
- 智能路由：使用LLM分析用户意图，自动选择最合适的Agent
- 可解释决策：提供路由决策的原因说明
- 降级策略：当LLM路由失败时，使用基于关键词的降级策略

### 2. 模块化设计
- **BaseAgent**: 所有Agent的基础抽象类，确保统一接口
- **RAGAgent**: 封装AgenticRAG系统，提供知识检索能力
- **ChatAgent**: 通用对话Agent，支持工具调用
- **ToolRegistry**: 统一管理MCP工具

### 3. 状态管理
- 使用TypedDict定义清晰的状态结构
- 支持消息历史管理
- 记录Agent执行轨迹
- 工具使用追踪

### 4. 可扩展性
- 易于添加新的Agent
- 支持动态工具注册
- 灵活的配置管理

## 架构设计

```
MultiAgentGraph (主图)
    ├── SupervisorAgent (路由决策)
    │   ├── 分析用户意图
    │   ├── 选择Agent
    │   └── 管理执行流程
    │
    ├── RAGAgent (知识检索)
    │   └── 封装AgenticRAG系统
    │
    ├── ChatAgent (对话处理)
    │   └── 支持工具调用
    │
    └── ToolRegistry (工具管理)
        └── 统一管理MCP工具
```

## 使用示例

### 基础使用

```python
from src.multi_agent import MultiAgentGraph

# 初始化系统
graph = MultiAgentGraph(
    rag_persist_directory="./tmp/chroma_db/agentic_rag",
    max_iterations=10
)

# 执行查询
result = graph.invoke("你的问题")

# 获取答案
answer = result["messages"][-1].content
print(answer)
```

### 添加自定义Agent

```python
from src.multi_agent.agents.base_agent import BaseAgent
from src.multi_agent.state import MultiAgentState

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="custom_agent",
            description="自定义Agent描述"
        )
    
    async def execute(self, state: MultiAgentState) -> Dict[str, Any]:
        # 实现Agent逻辑
        return {
            "result": "执行结果",
            "messages": [AIMessage(content="回复内容")],
            "metadata": {}
        }

# 使用自定义Agent
custom_agent = CustomAgent()
graph = MultiAgentGraph(agents=[custom_agent])
```

### 注册工具

```python
from src.multi_agent.tools.tool_registry import ToolRegistry
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """工具描述"""
    return "工具执行结果"

# 注册工具
registry = ToolRegistry()
registry.register_tool("my_tool", my_tool, category="custom")

# 在ChatAgent中使用
chat_agent = ChatAgent(tool_registry=registry)
```

## 文件结构

```
src/multi_agent/
├── __init__.py              # 模块导出
├── state.py                  # 状态定义
├── supervisor.py             # Supervisor实现
├── graph.py                  # 主图实现
├── config.py                 # 配置管理
├── agents/
│   ├── __init__.py
│   ├── base_agent.py         # 基础Agent类
│   ├── rag_agent.py          # RAG Agent
│   └── chat_agent.py         # Chat Agent
└── tools/
    ├── __init__.py
    └── tool_registry.py      # 工具注册表
```

## 最佳实践

### 1. Agent设计
- 继承BaseAgent并实现execute方法
- 提供清晰的Agent描述，帮助Supervisor进行路由
- 处理错误并返回有意义的错误信息

### 2. 状态管理
- 只更新必要的状态字段
- 保持状态结构的一致性
- 记录重要的执行信息

### 3. 工具集成
- 通过ToolRegistry统一管理工具
- 提供清晰的工具描述
- 追踪工具使用情况

### 4. 错误处理
- 在Agent中捕获和处理错误
- 返回有意义的错误信息
- 使用降级策略提高系统鲁棒性

## 扩展指南

### 添加新Agent

1. 创建Agent类，继承BaseAgent
2. 实现execute方法
3. 在初始化MultiAgentGraph时传入Agent实例
4. 在Supervisor的路由提示词中添加Agent描述（可选）

### 集成MCP工具

1. 创建或获取工具实例
2. 使用ToolRegistry注册工具
3. 将ToolRegistry传递给需要工具的Agent

### 自定义路由逻辑

1. 修改SupervisorAgent的route方法
2. 更新路由提示词
3. 调整降级策略

## 配置

通过环境变量或配置文件设置：

```bash
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1
RAG_PERSIST_DIRECTORY=./tmp/chroma_db/agentic_rag
MAX_ITERATIONS=10
ENABLE_WEB_SEARCH=true
LOG_LEVEL=INFO
```

## 注意事项

1. **异步执行**: 当前实现使用同步方式调用异步方法，在生产环境中建议使用完全异步的实现
2. **错误处理**: 确保所有Agent都有适当的错误处理
3. **性能优化**: 对于高并发场景，考虑使用异步执行和连接池
4. **监控**: 建议添加监控和日志记录，追踪系统性能

## 未来改进

- [ ] 完全异步实现
- [ ] 支持Agent之间的直接通信
- [ ] 添加Agent执行缓存
- [ ] 支持动态Agent加载
- [ ] 增强监控和可观测性
- [ ] 支持分布式部署

