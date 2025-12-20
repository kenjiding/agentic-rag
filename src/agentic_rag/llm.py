from typing import Generator
from langchain_openai import ChatOpenAI

class LLM:
  def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1) -> None:
    self.llm = ChatOpenAI(model=model_name, temperature=temperature)

  def generate(self, prompt: str) -> str:
    return self.llm.invoke(prompt)
  
  def stream(self, prompt: str) -> Generator[str, None, None]:
    return self.llm.stream(prompt)
