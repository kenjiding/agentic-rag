
from typing import List
import os
from pathlib import Path
from langchain_core.documents import Document
import mimetypes
import PyPDF2

class DocsParser:
  def __init__(self, type: str = None, file_path: str = "") -> None:
    if not file_path:
      raise ValueError("file_path is required")

    if not type:
      self.type = self.detect_type(file_path)
    else: 
      self.type = type

    if self.type == "application/pdf":
      self.docs = self.parse_pdf(file_path)
    elif self.type == "text/txt":
      self.docs = self.parse_txt(file_path)
    elif self.type == "text/markdown":
      self.docs = self.parse_markdown(file_path)
    elif self.type == "application/plain":
      self.docs = self.parse_plain(file_path)
    else:
      raise ValueError(f"Unsupported file type: {self.type}")
    
  def _get_absolute_path(self, file_path: str) -> str:
      """将路径转换为规范化的绝对路径"""
      path = Path(file_path)
      return str(path.resolve())

  def detect_type(self, file_path: str) -> str:
    path = self._get_absolute_path(file_path)
    if not os.path.exists(path):
      raise FileNotFoundError(f"File not found: {file_path}")
    mime_type, encoding = mimetypes.guess_type(self._get_absolute_path(file_path))
    return mime_type or "text/plain"

  def parse_pdf(self, file_path: str) -> List[Document]:
      from langchain_community.document_loaders import PyPDFLoader

      loader = PyPDFLoader(file_path)
      documents = loader.load()
      
      # 添加类型和规范化元数据
      for i, doc in enumerate(documents):
          doc.metadata.update({
              "type": self.type,
              "source": file_path,
              "page": i + 1  # 添加页码
          })
      
      return documents
    # with open(self._get_absolute_path(file_path), "rb") as pdf:
    #   reader = PyPDF2.PdfReader(pdf)
    #   return [Document(
    #     page_content=page.extract_text(),
    #     metadata={"source": file_path, "type": self.type}) for page in reader.pages
    #   ]

  def parse_txt(self, file_path: str) -> List[Document]:
    with open(self._get_absolute_path(file_path), "r", encoding="utf-8") as file:
      return [Document(page_content=file.read(), metadata={"source": file_path, "type": self.type})]

  def parse_markdown(self, file_path: str) -> List[Document]:
    file_path = self._get_absolute_path(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
      return [Document(page_content=file.read(), metadata={"source": file_path, "type": self.type})]

  def parse_plain(self, file_path: str) -> List[Document]:
    return []