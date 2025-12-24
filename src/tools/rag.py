from src.mcp.rag import search_from_rag
from src.utils.mcp import create_mcp_stdio_client

async def get_rag_tools():
  params = {
    "command": "uv",
    "args": [
      "run",
      "python",
      "-m",
      "src.mcp.rag"
    ]
  }

  tools, client = await create_mcp_stdio_client("agentic_rag_tools", params)

  return tools
