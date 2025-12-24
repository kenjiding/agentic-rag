from typing import Generator, Optional, Any
from langchain_openai import ChatOpenAI
import os

# Optional dependencies
try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    ChatOllama = None  # type: ignore

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    ChatGoogleGenerativeAI = None  # type: ignore


class LLM:
  def __init__(self, model_name: str = "openai:gpt-4o-mini", temperature: float = 0.1) -> None:
    self.type = model_name.split(":")[0]
    self.model_name = model_name.split(":")[1]
    self.temperature = temperature
    if self.type == "openai":
      self.llm = self.init_openai(self.model_name, temperature)
    elif self.type == "ollama":
      if not HAS_OLLAMA:
        raise ImportError("langchain_ollama is required for Ollama models. Install it with: pip install langchain-ollama")
      self.llm = self.init_ollama(self.model_name, temperature)
    elif self.type == "gemini":
      if not HAS_GEMINI:
        raise ImportError("langchain-google-genai is required for Gemini models. Install it with: pip install langchain-google-genai")
      self.llm = self.init_gemini(self.model_name, temperature)
    else:
      raise ValueError(f"Unsupported model type: {self.type}")

  def init_openai(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=temperature)

  def init_ollama(self, model_name: str = "qwen3:8b", temperature: float = 0.2) -> Any:
    if not HAS_OLLAMA or ChatOllama is None:
      raise ImportError("langchain_ollama is required")
    return ChatOllama(model=model_name, temperature=temperature)

  def init_gemini(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.2) -> Any:
    if not HAS_GEMINI or ChatGoogleGenerativeAI is None:
      raise ImportError("langchain-google-genai is required")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, api_key=os.getenv("GOOGLE_API_KEY"))

  def generate(self, prompt: str) -> str:
    return self.llm.invoke(prompt)

  def stream(self, prompt: str) -> Generator[str, None, None]:
    return self.llm.stream(prompt)

  def get_llm(self) -> ChatOpenAI | ChatOllama | ChatGoogleGenerativeAI:
    return self.llm
