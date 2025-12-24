import os
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from src.agentic_rag.agentic_rag import AgenticRAG

mcp = FastMCP(name="agentic_rag")

current_file = os.path.dirname(__file__)
persist_directory = os.path.join(current_file, "../../tmp/chroma_db/agentic_rag")

rag = AgenticRAG(
  model_name="gpt-4o-mini",
  max_iterations=3,
  persist_directory=persist_directory
)

@mcp.tool(name="search_from_rag", description="Search the RAG database for the answer to the question")
def search_from_rag(question: Annotated[
  str,
  Field(
    description="The query to search the RAG database",
    examples=["What is the capital of France?"]
  )]) -> str:
  if not question:
    raise ValueError("Question is required")
  result = rag.query(question, verbose=True)
  if not result["answer"]:
    raise ValueError("No answer found from the RAG database")
  return result["answer"]


if __name__ == "__main__":
  mcp.run(transport="stdio")