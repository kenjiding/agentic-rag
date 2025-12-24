import asyncio
import time
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.state import RunnableConfig
from tools.rag import get_rag_tools

def format_debug_output(step_name: str, content: str, is_tool_call = False) -> None:
  if is_tool_call:
    print(f'🔄 【工具调用】 {step_name}')
    print("-" * 40)
    print(content.strip())
    print("-" * 40)
  else:
    print(f"💭 【{step_name}】")
    print("-" * 40)
    print(content.strip())
    print("-" * 40)

async def create_chat_agent(tools: list):
  return create_agent(
    tools=tools,
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks. you should use agentic rag tools to answer questions first."
  )

async def run_chat_agent():
  tools = await get_rag_tools()
  agent = await create_chat_agent(tools)
  session_id = "1"
  config = RunnableConfig(configurable={"thread_id": session_id}, recursion_limit=100)
  input_messages = [("user", "2019年福布斯富豪榜杰夫·贝索斯财富是多少?")]
  res = agent.astream(input={"messages": input_messages}, config=config)
  async for chunk in res:
        print("=" * 60)
        items = chunk.items();
        for item in items:
          (model_name, model_output) = item;
          messages = model_output.get("messages", []);

          for message in messages:
            if isinstance(message, AIMessage):
              response_metadata = message.response_metadata
              print("完成原因: ", f"{response_metadata.get("finish_reason", "")}, ", "使用的模型: ", f"{response_metadata.get("model_name", "")}");
              tool_calls = message.tool_calls;
              if tool_calls and len(tool_calls) > 0:
                for tool_call in tool_calls:
                  tool_name = tool_call.get("name", "");
                  tool_input = tool_call.get("args", {});
                  print(f"工具名称: {tool_name}")
                  print(f"工具输入: {tool_input}")
                  print("-" * 60)

              print("\n")
              print("AI 回答: ")
              print("-" * 60)
              print(message.content)
              print("-" * 60)
              usage_metadata = message.usage_metadata;
              print("input_tokens: ", f"{usage_metadata.get("input_tokens", "")}, ", "output_tokens: ", f"{usage_metadata.get("output_tokens", "")}");
          
            if isinstance(message, ToolMessage):
              # 从 ToolMessage 中获取工具名称
              tool_name = getattr(message, 'name', '未知工具')
              tool_result = f"""
  🔧 工具：{tool_name}
  📤 结果：
  {message.content}
  ✅ 状态：执行完成，可以开始下一个任务
              """
              format_debug_output("工具执行结果", tool_result, is_tool_call=True)

if __name__ == "__main__":
  asyncio.run(run_chat_agent())