from src.mcp.rag import search_from_rag
from src.utils.mcp import create_mcp_stdio_client

async def get_web_search_tools():
  params = {
    "command": "npx",
    "args": ["open-websearch@latest"]
  }

  tools, client = await create_mcp_stdio_client("web_search_tools", params)

  return tools
