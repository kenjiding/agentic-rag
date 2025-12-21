from agentic_rag.llm import LLM
from src.agentic_rag.agentic_rag import AgenticRAG

def run_agentic_rag():
  print("Running Agentic RAG...")
  print("=" * 60)
  llm = LLM(model_name="openai:gpt-4o-mini", temperature=0.1)
  agentic_rag = AgenticRAG(llm=llm.get_llm())
  agentic_rag.run("What is the capital of France?")
