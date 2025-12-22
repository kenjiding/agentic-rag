"""结构感知分块器实现 - 针对 Markdown 等结构化文档

2025 最佳实践：先按结构分块，再按大小细分
"""
from typing import List
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document


class StructureAwareChunker:
    """结构感知分块器 - 针对 Markdown 等结构化文档

    2025 最佳实践：先按结构分块，再按大小细分
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Markdown 标题分块器
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ]
        )

        # 递归分块器（用于细分过大的块）
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        """分割 Markdown 文本"""
        try:
            # 1. 先按 Markdown 结构分块
            md_docs = self.md_splitter.split_text(text)

            # 2. 对过大的块进行细分
            result = []
            for doc in md_docs:
                content = doc.page_content
                if len(content) > self.chunk_size:
                    # 需要细分
                    sub_chunks = self.recursive_splitter.split_text(content)
                    result.extend(sub_chunks)
                else:
                    result.append(content)

            return result

        except Exception as e:
            print(f"[结构分块] 失败，回退到递归分块: {e}")
            return self.recursive_splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_method": "structure_aware"
                    }
                )
                result.append(new_doc)
        return result

