from typing import List
import pickle
import hashlib
import asyncio
import time
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.agentic_rag.llm import LLM

# 1. 定义 Markdown 标题切分器
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    # 其他切分符
    ("\n\n", "Paragraph"),
    ("\n", "Line Break"),
    ("。", "Sentence End"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 2. 这里的 text_splitter 用于处理单个章节过长的情况
chunk_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000, 
  chunk_overlap=200,
  separators=[
      "\n\n",      # 段落分隔（最重要）
      "\n",        # 换行
      "。",        # 中文句号
      ". ",        # 英文句号+空格
      "；",        # 中文分号（可选）
      "; ",        # 英文分号（可选）
      "，",        # 中文逗号（可选，可能切分太细）
      ", ",        # 英文逗号（可选，可能切分太细）
      " ",         # 空格
      ""   
  ])

class PDFParser:
  def __init__(self, cache_dir: str = "tmp/pdf_cache"):
    self.converter = DocumentConverter()
    self.llm = LLM(model_name="openai:gpt-3.5-turbo", temperature=0.2).get_llm()
    self.cache_dir = Path(cache_dir)
    self.cache_dir.mkdir(parents=True, exist_ok=True)

  def parse_pdf_to_documents(self, pdf_path: str, refresh: bool = False) -> List[Document]:
    md_text = self.parse_pdf_to_markdown(pdf_path)
    return self.intellgent_chunking_pdf(md_text, pdf_path, refresh=refresh)

  def parse_pdf_to_markdown(self, pdf_path: str) -> str:
    result = self.converter.convert(pdf_path)

    return result.document.export_to_markdown()

  def _get_cache_path(self, pdf_path: str) -> Path:
    """生成缓存文件路径"""
    # 使用PDF路径的哈希值作为缓存文件名，避免路径中的特殊字符问题
    pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    return self.cache_dir / f"{pdf_hash}.pkl"

  def intellgent_chunking_pdf(self, md_text: str, pdf_path: str, refresh: bool = False) -> List[Document]:
    cache_path = self._get_cache_path(pdf_path)
    
    # 如果不需要刷新且缓存存在，直接加载缓存
    if not refresh and cache_path.exists():
      print(f"Loading cached chunks from {cache_path}")
      try:
        with open(cache_path, 'rb') as f:
          return pickle.load(f)
      except Exception as e:
        print(f"Error loading cache: {e}, regenerating...")
    
    # 需要重新生成：first physical chunking by markdown headers
    header_splits = markdown_splitter.split_text(md_text)

    final_chunks = []

    # 定义语境增强的 Prompt
    # 这里的逻辑是：给 LLM 看整个文档摘要(或父章节)，让它解释这个切片属于哪里。
    # 为节省 Token，这里简化为让 LLM 基于切片元数据(Headers)生成一句话概括。
    context_prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. 
        Here is a chunk of text from a document. The metadata shows it belongs to these sections: {headers}.
        
        Please generate a short, one-sentence context description that explains what this chunk is about and where it fits in the document.
        Start with: "This section discusses..."
        
        Chunk content:
        {content}
        """
    )
    context_chain = context_prompt | self.llm | StrOutputParser()
    print(f"Processing {len(header_splits)} logical sections...")

    # 收集所有需要处理的 chunk 数据
    chunks_to_process = []
    for split in header_splits:
      # 如果章节本身太长，进一步物理切分
      sub_splits = chunk_splitter.split_documents([split])
      for sub_split in sub_splits:
        chunks_to_process.append({
          "original_content": sub_split.page_content,
          "headers": sub_split.metadata,
          "pdf_path": pdf_path
        })

    # 异步并发处理所有 chunk
    async def process_chunk_async(chunk_data: dict, semaphore: asyncio.Semaphore, chunk_idx: int) -> Document:
      """异步处理单个 chunk 的上下文生成"""
      async with semaphore:  # 控制并发数，避免过多并发导致问题
        
        original_content = chunk_data["original_content"]
        headers = chunk_data["headers"]
        pdf_path = chunk_data["pdf_path"]
        
        try:
          # 使用异步调用
          context_desc = await context_chain.ainvoke({"headers": headers, "content": original_content})
        except Exception as e:
          print(f"Error generating context for chunk {chunk_idx}: {e}")
          context_desc = "Context generation failed."

        # 将生成的上下文拼接到原始内容前面，用于向量检索，但保留原始内容用于展示
        # 这种技术叫做 "Document Shadowing" 或 "Contextual Retrieval"
        enhanced_content = f"Context: {context_desc}\n\nOriginal Content:\n{original_content}"
        return Document(
            page_content=enhanced_content, # 检索时用这个包含上下文的内容
            metadata={
                **headers,
                "source": pdf_path,
                "type": "text/markdown",
                "original_content": original_content # 后面生成答案时用纯净内容
            }
        )

    # 并发处理所有 chunk
    async def process_all_chunks_async():
      # 使用 semaphore 控制并发数（设置为 10，可以根据实际情况调整）
      # 这样可以避免过多并发导致 HTTP 连接池耗尽
      semaphore = asyncio.Semaphore(10)
      # 创建所有任务，确保它们真正并发执行
      tasks = [asyncio.create_task(process_chunk_async(chunk_data, semaphore, idx)) 
               for idx, chunk_data in enumerate(chunks_to_process)]
      results = await asyncio.gather(*tasks)
      print(f"All tasks completed at {time.time():.2f}s")
      return results

    # 在同步方法中运行异步代码
    print(f"Processing {len(chunks_to_process)} chunks concurrently (max 10 at a time)...")
    final_chunks = asyncio.run(process_all_chunks_async())
    
    # 保存缓存
    try:
      with open(cache_path, 'wb') as f:
        pickle.dump(final_chunks, f)
      print(f"Cached chunks saved to {cache_path}")
    except Exception as e:
      print(f"Error saving cache: {e}")
    
    return final_chunks