from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocsSplitter:
  def __init__(self, docs_type: str = None, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
    if not docs_type:
      raise ValueError("docs_type is required")

    self.splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      separators=self.get_separators(docs_type)
    )

  def get_separators(self, docs_type: str = None):
    if docs_type == "text/markdown":
      return [
          "\n## ",    # 二级标题
          "\n### ",   # 三级标题
          "\n\n",     # 段落
          "\n",       # 换行
          "。",
          ". ",
          " ",
          ""
      ]
    elif docs_type == "text/txt":
      return [
          "\n\n",  # 段落分隔符
          "\n",    # 换行符
          "。",    # 中文句号
          ". ",    # 英文句号+空格
          "，",    # 中文逗号
          ", ",    # 英文逗号
          " ",     # 空格
          ""       # 字符（最后选择）
      ]
    elif docs_type == "application/pdf":
      return [
        # 简历和结构化文档的常见分隔符
        "\n\n\n\n",      # 大节分隔（如工作经历、教育背景之间）
        "\n\n\n",        # 节内段落分隔
        "\n\n",          # 段落分隔
        "\n• ",          # 项目符号列表
        "\n- ",          # 破折号列表
        "\n* ",          # 星号列表
        "\n  ",          # 缩进（子项）
        "\n",            # 换行
        "。",            # 中文句号
        ". ",            # 英文句号+空格
        "：",            # 中文冒号（如"姓名："）
        ": ",            # 英文冒号
        "；",            # 中文分号
        "; ",            # 英文分号
        "，",            # 中文逗号
        ", ",            # 英文逗号
        " ",             # 空格
        ""   
      ]
    else:
      return ["\n\n", "\n", " ", ""]

  def get_splitter(self):
    return self.splitter