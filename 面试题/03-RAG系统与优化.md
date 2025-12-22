# RAG系统与优化相关面试题

## 问题4: 介绍一下RAG的整体流程。在Agent落地场景中,RAG会遇到哪些延迟和正确率问题?你怎么优化召回链路?

### 问题解析

这个问题需要深入理解：
- **RAG（Retrieval-Augmented Generation）**的完整流程
- **Agent场景**下的特殊挑战
- **延迟问题**的来源和优化方法
- **正确率问题**的原因和解决方案
- **召回链路**的优化策略

### 详细答案

#### 1. RAG整体流程详解

##### 1.1 RAG的核心思想

**RAG（Retrieval-Augmented Generation，检索增强生成）**的核心思想是：
- **不依赖模型内部知识**：模型不需要记住所有知识
- **动态检索相关知识**：从外部知识库中检索相关信息
- **基于检索结果生成**：结合检索到的上下文生成答案

**为什么需要RAG？**
- **知识更新问题**：模型训练后知识就固定了，无法获取最新信息
- **知识容量限制**：模型无法记住所有领域的详细知识
- **幻觉问题**：模型可能生成不准确的信息
- **可解释性**：RAG可以提供答案的来源，增强可信度

##### 1.2 RAG的完整流程

**阶段1: 知识库构建（Offline）**

```
原始文档
    ↓
文档解析（PDF、Word、网页等）
    ↓
文档切分（Chunking）
    ↓
向量化（Embedding）
    ↓
向量数据库存储（Vector Store）
    ↓
知识库就绪
```

**详细步骤**：

1. **文档解析**
   ```python
   # 示例：解析PDF文档
   from langchain.document_loaders import PyPDFLoader
   
   loader = PyPDFLoader("document.pdf")
   documents = loader.load()
   ```

2. **文档切分**
   ```python
   # 示例：文本切分
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=500,      # 每个chunk的字符数
       chunk_overlap=50     # chunk之间的重叠
   )
   chunks = text_splitter.split_documents(documents)
   ```
   
   **切分策略**：
   - **固定长度切分**：简单但可能切断语义
   - **语义切分**：按语义单元切分，保持语义完整
   - **结构感知切分**：考虑文档结构（段落、章节等）

3. **向量化**
   ```python
   # 示例：生成嵌入向量
   from langchain.embeddings import OpenAIEmbeddings
   
   embeddings = OpenAIEmbeddings()
   vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
   ```

4. **存储到向量数据库**
   ```python
   # 示例：存储到Chroma
   from langchain.vectorstores import Chroma
   
   vectorstore = Chroma.from_documents(
       documents=chunks,
       embedding=embeddings,
       persist_directory="./vector_db"
   )
   ```

**阶段2: 查询处理（Online）**

```
用户查询
    ↓
查询理解/改写（Query Understanding/Rewriting）
    ↓
向量化查询（Query Embedding）
    ↓
检索（Retrieval）
    ├─ 语义检索（Dense Retrieval）
    ├─ 关键词检索（Sparse Retrieval，如BM25）
    └─ 混合检索（Hybrid Retrieval）
    ↓
重排序（Reranking）
    ↓
上下文构建（Context Construction）
    ↓
生成答案（Generation）
    ↓
后处理（Post-processing）
    ↓
返回答案
```

**详细步骤**：

1. **查询理解/改写**
   ```python
   # 示例：查询改写
   def rewrite_query(original_query, llm):
       prompt = f"""
       将以下用户查询改写为更适合检索的形式，保持原意但使用更准确的关键词：
       
       原始查询：{original_query}
       
       改写后的查询：
       """
       rewritten = llm(prompt)
       return rewritten
   ```
   
   **改写目的**：
   - 将口语化查询转为正式查询
   - 补充隐含信息
   - 提取关键实体和概念

2. **检索（Retrieval）**
   
   **a) 语义检索（Dense Retrieval）**
   ```python
   # 使用向量相似度检索
   query_vector = embeddings.embed_query(query)
   results = vectorstore.similarity_search_with_score(
       query_vector,
       k=10  # 返回top-10
   )
   ```
   
   **b) 关键词检索（Sparse Retrieval）**
   ```python
   # 使用BM25检索
   from rank_bm25 import BM25Okapi
   
   # 构建BM25索引
   corpus = [chunk.page_content for chunk in chunks]
   bm25 = BM25Okapi(corpus)
   
   # 检索
   tokenized_query = query.split()
   scores = bm25.get_scores(tokenized_query)
   top_indices = np.argsort(scores)[::-1][:10]
   ```
   
   **c) 混合检索（Hybrid Retrieval）**
   ```python
   # 结合语义检索和关键词检索
   def hybrid_retrieval(query, vectorstore, bm25_index, k=10):
       # 语义检索
       dense_results = vectorstore.similarity_search(query, k=k*2)
       
       # 关键词检索
       sparse_results = bm25_index.search(query, k=k*2)
       
       # 融合结果（Reciprocal Rank Fusion）
       fused_results = reciprocal_rank_fusion(
           dense_results,
           sparse_results
       )
       
       return fused_results[:k]
   ```

3. **重排序（Reranking）**
   ```python
   # 使用Cross-Encoder重排序
   from sentence_transformers import CrossEncoder
   
   reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
   
   def rerank(query, documents):
       pairs = [[query, doc.page_content] for doc in documents]
       scores = reranker.predict(pairs)
       
       # 按分数排序
       ranked_indices = np.argsort(scores)[::-1]
       return [documents[i] for i in ranked_indices]
   ```

4. **上下文构建**
   ```python
   def build_context(retrieved_docs, max_length=2000):
       context = ""
       for doc in retrieved_docs:
           if len(context) + len(doc.page_content) <= max_length:
               context += f"\n\n{doc.page_content}"
           else:
               break
       return context
   ```

5. **生成答案**
   ```python
   def generate_answer(query, context, llm):
       prompt = f"""
       基于以下上下文回答问题。如果上下文中没有相关信息，请说明。
       
       上下文：
       {context}
       
       问题：{query}
       
       答案：
       """
       answer = llm(prompt)
       return answer
   ```

##### 1.3 RAG流程图

```
┌─────────────────────────────────────────────────────────┐
│                    RAG系统架构                            │
└─────────────────────────────────────────────────────────┘

离线阶段（知识库构建）：
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ 原始文档 │ --> │ 文档切分 │ --> │ 向量化   │ --> │ 向量存储 │
└──────────┘     └──────────┘     └──────────┘     └──────────┘

在线阶段（查询处理）：
┌──────────┐
│ 用户查询 │
└────┬─────┘
     │
     ├─> ┌──────────────┐
     │   │ 查询改写     │
     │   └──────┬───────┘
     │          │
     │          ├─> ┌──────────────┐
     │          │   │ 向量化查询   │
     │          │   └──────┬───────┘
     │          │          │
     │          │          ├─> ┌──────────────┐
     │          │          │   │ 语义检索     │
     │          │          │   └──────┬───────┘
     │          │          │          │
     │          │          ├─> ┌──────────────┐
     │          │          │   │ 关键词检索   │
     │          │          │   └──────┬───────┘
     │          │          │          │
     │          │          └─> ┌──────────────┐
     │          │              │ 混合检索融合 │
     │          │              └──────┬───────┘
     │          │                     │
     │          │                     ├─> ┌──────────────┐
     │          │                     │   │ 重排序       │
     │          │                     │   └──────┬───────┘
     │          │                     │          │
     │          │                     └─> ┌──────────────┐
     │          │                         │ 上下文构建   │
     │          │                         └──────┬───────┘
     │          │                                │
     │          └─> ┌──────────────┐            │
     │              │ LLM生成答案  │ <───────────┘
     │              └──────┬───────┘
     │                    │
     └────────────────────┘
                    │
              ┌─────▼─────┐
              │  返回答案  │
              └───────────┘
```

#### 2. Agent落地场景中的挑战

##### 2.1 Agent场景的特点

**Agent场景**与普通RAG应用的区别：

1. **多轮对话**：
   - 需要维护对话历史
   - 需要理解上下文依赖
   - 查询可能不完整

2. **实时性要求高**：
   - 用户期望快速响应
   - 延迟直接影响用户体验

3. **准确性要求高**：
   - Agent需要做出决策
   - 错误信息可能导致错误决策

4. **动态查询**：
   - 查询可能根据对话动态调整
   - 需要自适应检索策略

##### 2.2 延迟问题分析

**延迟来源分析**：

```
总延迟 = 查询处理延迟 + 检索延迟 + 生成延迟 + 网络延迟

1. 查询处理延迟：
   - 查询改写：100-500ms（调用LLM）
   - 查询向量化：50-200ms（Embedding API）

2. 检索延迟：
   - 向量检索：50-300ms（取决于数据库规模和硬件）
   - 关键词检索：20-100ms（BM25）
   - 混合检索融合：10-50ms

3. 重排序延迟：
   - Cross-Encoder重排序：100-500ms（取决于候选数量）

4. 生成延迟：
   - LLM生成：500-3000ms（取决于答案长度和模型）

5. 网络延迟：
   - API调用：50-200ms（每次）
```

**典型延迟分布**（总延迟约2-5秒）：

| 阶段 | 延迟 | 占比 |
|------|------|------|
| 查询处理 | 150-700ms | 10-20% |
| 检索 | 80-450ms | 5-15% |
| 重排序 | 100-500ms | 5-15% |
| 生成 | 500-3000ms | 60-80% |
| 网络 | 50-200ms | 2-5% |

**延迟问题的具体表现**：

1. **查询改写延迟**
   - 每次查询都需要调用LLM改写
   - LLM API调用通常需要100-500ms
   - 对于简单查询，改写可能不必要

2. **多路检索延迟**
   - 语义检索和关键词检索并行执行
   - 但都需要等待最慢的那个完成
   - 重排序需要等待所有检索完成

3. **生成延迟**
   - LLM生成是主要延迟来源
   - 答案越长，延迟越大
   - 无法并行优化

##### 2.3 正确率问题分析

**正确率问题的来源**：

1. **检索不准确**
   - **语义不匹配**：查询和文档的语义相似但实际不相关
   - **关键词缺失**：文档包含相关信息但缺少查询中的关键词
   - **上下文丢失**：切分时丢失了重要上下文

2. **上下文不足**
   - 检索到的文档不完整
   - 缺少关键信息
   - 多个文档信息冲突

3. **生成错误**
   - 模型基于错误上下文生成
   - 模型产生幻觉
   - 模型理解错误

**正确率问题的具体表现**：

**示例1: 检索不准确**

```
用户查询："如何配置Redis集群？"

检索结果：
1. "Redis单机配置"（语义相似但不相关）
2. "MySQL集群配置"（包含"集群"关键词但不相关）
3. "Redis集群配置"（相关，但排名靠后）

问题：相关文档没有被优先检索到
```

**示例2: 上下文不足**

```
用户查询："Redis集群的故障转移机制"

检索到的文档片段：
"Redis集群使用主从复制实现故障转移。当主节点故障时..."

问题：文档片段不完整，缺少关键细节（如故障检测时间、切换流程等）
```

**示例3: 生成错误**

```
检索到的上下文："Redis集群支持自动故障转移，故障检测时间约为15秒。"

模型生成："Redis集群的故障转移是手动的，需要管理员干预。"

问题：模型忽略了上下文中的信息，生成了错误答案
```

#### 3. 召回链路优化策略

##### 3.1 延迟优化

**策略1: 查询改写优化**

**a) 缓存常见查询改写**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query_rewrite(query):
    """缓存查询改写结果"""
    return rewrite_query(query, llm)

# 使用
rewritten_query = cached_query_rewrite(user_query)
```

**b) 智能判断是否需要改写**
```python
def should_rewrite_query(query):
    """判断查询是否需要改写"""
    # 简单查询（短且包含关键词）可能不需要改写
    if len(query.split()) <= 5 and has_keywords(query):
        return False
    
    # 复杂查询（长或模糊）需要改写
    if len(query.split()) > 10 or is_ambiguous(query):
        return True
    
    return False
```

**c) 异步改写**
```python
import asyncio

async def async_query_rewrite(query, llm):
    """异步查询改写"""
    # 如果改写耗时，可以异步执行
    # 同时进行其他操作
    rewritten = await llm.ainvoke(rewrite_prompt)
    return rewritten
```

**策略2: 检索优化**

**a) 向量数据库优化**
```python
# 1. 使用更快的向量数据库
# Chroma -> Milvus（更快的检索速度）
# 或使用本地向量数据库（减少网络延迟）

# 2. 索引优化
# 使用HNSW索引（Hierarchical Navigable Small World）
# 平衡检索速度和准确性

# 3. 批量检索
# 如果有多个查询，批量处理
def batch_retrieve(queries, vectorstore, batch_size=10):
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        batch_results = vectorstore.batch_similarity_search(batch)
        results.extend(batch_results)
    return results
```

**b) 混合检索优化**
```python
# 1. 并行执行语义检索和关键词检索
import asyncio

async def parallel_hybrid_retrieval(query, vectorstore, bm25_index):
    # 并行执行
    dense_task = asyncio.create_task(
        vectorstore.asimilarity_search(query, k=10)
    )
    sparse_task = asyncio.create_task(
        bm25_index.asearch(query, k=10)
    )
    
    dense_results, sparse_results = await asyncio.gather(
        dense_task, sparse_task
    )
    
    # 融合结果
    return reciprocal_rank_fusion(dense_results, sparse_results)
```

**c) 早期终止**
```python
def early_termination_retrieval(query, vectorstore, min_score=0.7):
    """如果找到高质量结果，提前终止"""
    results = []
    for doc, score in vectorstore.similarity_search_with_score(query, k=20):
        results.append((doc, score))
        # 如果前3个结果分数都很高，提前终止
        if len(results) >= 3 and all(s >= min_score for _, s in results[:3]):
            break
    return results
```

**策略3: 重排序优化**

**a) 减少重排序候选数量**
```python
def optimized_rerank(query, documents, reranker, top_k=5):
    """只对top-k候选重排序"""
    # 先使用快速方法筛选
    # 只对前10个候选重排序，而不是全部
    candidates = documents[:10]
    
    # 重排序
    reranked = rerank(query, candidates)
    
    # 返回top-k
    return reranked[:top_k]
```

**b) 使用更快的重排序模型**
```python
# 使用更小更快的重排序模型
# cross-encoder/ms-marco-MiniLM-L-6-v2 (更快)
# vs
# cross-encoder/ms-marco-MiniLM-L-12-v2 (更准确但更慢)
```

**c) 缓存重排序结果**
```python
@lru_cache(maxsize=5000)
def cached_rerank(query_hash, doc_hashes):
    """缓存重排序结果（基于查询和文档的hash）"""
    # 如果查询和文档组合相同，直接返回缓存结果
    pass
```

**策略4: 生成优化**

**a) 流式生成**
```python
def stream_generate(query, context, llm):
    """流式生成，用户可以看到部分结果"""
    for chunk in llm.stream(prompt):
        yield chunk
        # 用户可以立即看到部分结果，感知延迟降低
```

**b) 限制生成长度**
```python
def generate_with_length_limit(query, context, llm, max_tokens=200):
    """限制生成长度，减少生成时间"""
    prompt = f"""
    基于上下文回答问题，答案要简洁，不超过{max_tokens}字。
    
    上下文：{context}
    问题：{query}
    答案：
    """
    return llm(prompt, max_tokens=max_tokens)
```

**c) 使用更快的模型**
```python
# 对于简单查询，使用更快的模型
def select_model(query_complexity):
    if query_complexity == 'simple':
        return 'gpt-3.5-turbo'  # 更快
    else:
        return 'gpt-4'  # 更准确但更慢
```

##### 3.2 正确率优化

**策略1: 检索准确性优化**

**a) 查询扩展（Query Expansion）**
```python
def expand_query(query, llm):
    """扩展查询，增加相关关键词"""
    prompt = f"""
    为以下查询生成3个相关的查询变体，用于提高检索准确性：
    
    原始查询：{query}
    
    查询变体：
    1.
    2.
    3.
    """
    variants = llm(prompt)
    return [query] + variants
```

**b) 多轮检索（Multi-Round Retrieval）**
```python
def multi_round_retrieval(query, vectorstore, max_rounds=3):
    """多轮检索，逐步细化"""
    all_results = []
    current_query = query
    
    for round in range(max_rounds):
        # 检索
        results = vectorstore.similarity_search(current_query, k=10)
        all_results.extend(results)
        
        # 分析检索结果质量
        quality = evaluate_retrieval_quality(query, results)
        
        if quality >= 0.8:  # 质量足够高，停止
            break
        
        # 如果质量不够，改写查询
        current_query = refine_query(query, results, llm)
    
    # 去重并返回
    return deduplicate(all_results)
```

**c) 语义切分优化**
```python
# 使用语义切分，保持语义完整
from langchain.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    chunk_size=500,
    chunk_overlap=100
)

# 语义切分可以保持上下文完整，提高检索准确性
chunks = semantic_splitter.split_documents(documents)
```

**策略2: 上下文质量优化**

**a) 动态上下文长度**
```python
def build_dynamic_context(query, retrieved_docs, max_length=2000):
    """根据查询复杂度动态调整上下文长度"""
    query_complexity = estimate_complexity(query)
    
    if query_complexity == 'simple':
        max_length = 1000
    elif query_complexity == 'complex':
        max_length = 3000
    
    context = ""
    for doc in retrieved_docs:
        if len(context) + len(doc.page_content) <= max_length:
            context += f"\n\n{doc.page_content}"
        else:
            break
    
    return context
```

**b) 上下文去重和排序**
```python
def optimize_context(query, retrieved_docs):
    """优化上下文：去重、排序、过滤"""
    # 1. 去重
    unique_docs = deduplicate(retrieved_docs)
    
    # 2. 按相关性排序
    sorted_docs = sort_by_relevance(query, unique_docs)
    
    # 3. 过滤低质量文档
    filtered_docs = [doc for doc in sorted_docs 
                     if relevance_score(query, doc) >= 0.5]
    
    return filtered_docs
```

**c) 上下文增强**
```python
def enhance_context(query, retrieved_docs, llm):
    """增强上下文：补充相关信息"""
    # 提取关键实体
    entities = extract_entities(query)
    
    # 为每个实体检索补充信息
    enhanced_docs = list(retrieved_docs)
    for entity in entities:
        entity_docs = vectorstore.similarity_search(entity, k=2)
        enhanced_docs.extend(entity_docs)
    
    # 去重
    return deduplicate(enhanced_docs)
```

**策略3: 生成准确性优化**

**a) 提示工程优化**
```python
def build_optimized_prompt(query, context):
    """构建优化的提示"""
    prompt = f"""
    你是一个专业的AI助手。请基于以下上下文回答问题。
    
    **重要要求**：
    1. 只使用上下文中的信息回答问题
    2. 如果上下文中没有相关信息，明确说明"根据提供的信息，无法回答此问题"
    3. 不要编造信息
    4. 如果信息不完整，说明哪些部分无法确定
    
    上下文：
    {context}
    
    问题：{query}
    
    答案：
    """
    return prompt
```

**b) 答案验证**
```python
def verify_answer(query, answer, context, llm):
    """验证答案的准确性"""
    verification_prompt = f"""
    验证以下答案是否基于提供的上下文，且没有编造信息：
    
    问题：{query}
    上下文：{context}
    答案：{answer}
    
    请判断：
    1. 答案是否基于上下文？
    2. 答案是否有编造的信息？
    3. 答案是否完整？
    
    验证结果：
    """
    verification = llm(verification_prompt)
    
    if "有编造" in verification or "不基于上下文" in verification:
        # 重新生成或标记为不确定
        return None
    
    return answer
```

**c) 多候选生成和选择**
```python
def generate_multiple_candidates(query, context, llm, n=3):
    """生成多个候选答案，选择最好的"""
    candidates = []
    
    for i in range(n):
        # 使用不同的温度生成不同风格的答案
        answer = llm.generate(
            prompt=build_prompt(query, context),
            temperature=0.7 + i * 0.1
        )
        candidates.append(answer)
    
    # 评估每个候选的质量
    best_answer = max(candidates, key=lambda a: evaluate_answer_quality(query, a, context))
    
    return best_answer
```

##### 3.3 综合优化策略

**策略1: 自适应检索策略**

```python
class AdaptiveRetriever:
    def __init__(self, vectorstore, bm25_index, reranker):
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.reranker = reranker
    
    def retrieve(self, query, latency_budget=2000):
        """根据延迟预算自适应选择检索策略"""
        start_time = time.time()
        
        # 1. 快速评估查询复杂度
        complexity = self.assess_complexity(query)
        
        if complexity == 'simple' and latency_budget < 1000:
            # 简单查询 + 低延迟预算：只用语义检索
            results = self.vectorstore.similarity_search(query, k=5)
            return results
        
        elif complexity == 'medium':
            # 中等复杂度：混合检索，但跳过重排序
            dense_results = self.vectorstore.similarity_search(query, k=10)
            sparse_results = self.bm25_index.search(query, k=10)
            results = reciprocal_rank_fusion(dense_results, sparse_results)
            return results[:5]
        
        else:
            # 复杂查询：完整流程
            dense_results = self.vectorstore.similarity_search(query, k=20)
            sparse_results = self.bm25_index.search(query, k=20)
            results = reciprocal_rank_fusion(dense_results, sparse_results)
            
            # 检查剩余时间
            elapsed = (time.time() - start_time) * 1000
            if elapsed < latency_budget - 500:  # 留500ms给生成
                # 有时间，进行重排序
                results = self.reranker.rerank(query, results[:10])
            
            return results[:5]
```

**策略2: 缓存策略**

```python
class SmartCache:
    def __init__(self):
        self.query_cache = {}  # 查询 -> 结果
        self.semantic_cache = {}  # 语义相似查询 -> 结果
    
    def get(self, query):
        # 1. 精确匹配
        if query in self.query_cache:
            return self.query_cache[query]
        
        # 2. 语义相似匹配
        for cached_query, result in self.semantic_cache.items():
            similarity = compute_similarity(query, cached_query)
            if similarity > 0.9:  # 高度相似
                return result
        
        return None
    
    def put(self, query, result):
        self.query_cache[query] = result
        # 也存储语义版本
        query_embedding = embed(query)
        self.semantic_cache[query_embedding] = result
```

**策略3: 异步流水线**

```python
async def async_rag_pipeline(query, vectorstore, llm):
    """异步RAG流水线，最大化并行度"""
    # 1. 并行执行查询改写和向量化
    rewrite_task = asyncio.create_task(async_rewrite_query(query, llm))
    embed_task = asyncio.create_task(async_embed_query(query))
    
    rewritten_query, query_vector = await asyncio.gather(rewrite_task, embed_task)
    
    # 2. 并行执行语义检索和关键词检索
    dense_task = asyncio.create_task(
        vectorstore.asimilarity_search_by_vector(query_vector, k=10)
    )
    sparse_task = asyncio.create_task(
        bm25_index.asearch(rewritten_query, k=10)
    )
    
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
    
    # 3. 融合和重排序
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
    reranked_results = await asyncio.create_task(
        async_rerank(rewritten_query, fused_results[:10])
    )
    
    # 4. 生成答案
    context = build_context(reranked_results[:5])
    answer = await llm.agenerate(build_prompt(query, context))
    
    return answer
```

#### 4. 优化效果评估

**优化前后对比**：

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **平均延迟** | 3.5s | 1.2s | 66% ↓ |
| **P95延迟** | 5.8s | 2.1s | 64% ↓ |
| **检索准确率** | 72% | 85% | 13% ↑ |
| **答案准确率** | 68% | 82% | 14% ↑ |
| **用户满意度** | 3.2/5 | 4.3/5 | 34% ↑ |

#### 5. 总结

**RAG整体流程**：
1. 离线阶段：文档解析 → 切分 → 向量化 → 存储
2. 在线阶段：查询处理 → 检索 → 重排序 → 上下文构建 → 生成

**Agent场景的挑战**：
- **延迟问题**：查询处理、检索、重排序、生成各阶段都有延迟
- **正确率问题**：检索不准确、上下文不足、生成错误

**优化策略**：
- **延迟优化**：缓存、并行、早期终止、流式生成
- **正确率优化**：查询扩展、多轮检索、上下文优化、答案验证
- **综合优化**：自适应策略、智能缓存、异步流水线

**最佳实践**：
1. 根据查询复杂度自适应选择策略
2. 建立多级缓存机制
3. 使用异步流水线最大化并行度
4. 持续监控和优化各环节性能

---

