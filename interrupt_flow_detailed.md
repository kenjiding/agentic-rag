# `return interrupt(confirmation_data)` 详细代码流转过程

本文档详细追踪从 `order_agent.py:720` 调用 `interrupt()` 到恢复执行的完整流程。

---

## 📍 起点：order_agent.py:720

```720:720:src/multi_agent/agents/order_agent.py
            return interrupt(confirmation_data)
```

**此时的状态：**
- `confirmation_data` 包含：
  - `action_type`: 操作类型（如 "create_order"）
  - `action_data`: 操作数据（订单信息等）
  - `display_message`: 显示给用户的消息
  - `display_data`: UI 展示数据
  - `confirmation_pending`: 确认信息（包含 confirmation_id）
  - `conversation_phase`: "order_creating"
- 已创建确认记录（保存在 ConfirmationManager）
- 已在 `state.entities` 中设置 interrupt 数据

---

## 🔄 阶段 1：interrupt() 调用 → 抛出异常

### 1.1 interrupt() 函数执行
**位置：** `langgraph.types.interrupt`

**行为：**
- `interrupt(confirmation_data)` 接收字典参数
- **立即抛出 `GraphInterrupt` 异常**
- 异常对象包含 `Interrupt(value=confirmation_data)`

**关键点：**
- ❌ **不会返回任何值**（因为抛出异常）
- ✅ **异常是 LangGraph 识别"需要暂停"的信号**

---

## 🔄 阶段 2：graph_nodes.py 捕获异常

### 2.1 异常传播路径
```
order_agent._handle_llm_tool_calls()
  ↓ return interrupt(confirmation_data) 抛出异常
graph_nodes._execute_agent_node()
  ↓ 捕获 GraphInterrupt 异常
```

### 2.2 异常处理代码
**位置：** `src/multi_agent/graph_nodes.py:201-242`

```201:242:src/multi_agent/graph_nodes.py
        except GraphInterrupt as e:
            # LangGraph 1.x interrupt() 机制
            # 捕获 interrupt() 调用，转换为状态更新传递给客户端
            # GraphInterrupt 的结构: (Interrupt(value={...}),)
            # 需要从 e.args[0].value 获取实际的值
            interrupt_value = None
            if e.args and len(e.args) > 0:
                interrupt_obj = e.args[0]
                if hasattr(interrupt_obj, 'value'):
                    interrupt_value = interrupt_obj.value
                else:
                    # 如果 args[0] 本身就是值（向后兼容）
                    interrupt_value = interrupt_obj

            # 【关键修复】GraphInterrupt 的 value 可能是 (Interrupt(...),) 这样的 tuple
            # 需要继续解析获取实际的字典值
            if interrupt_value and isinstance(interrupt_value, tuple) and len(interrupt_value) > 0:
                first_element = interrupt_value[0]
                if hasattr(first_element, 'value'):
                    # Interrupt 对象，获取其 value 属性
                    interrupt_value = first_element.value
                    logger.info(f"{agent_name} 从 tuple 中提取 Interrupt.value")
                elif isinstance(first_element, dict):
                    # 直接是字典
                    interrupt_value = first_element
                    logger.info(f"{agent_name} 从 tuple 中提取 dict")

            logger.info(f"{agent_name} 触发 interrupt: {type(interrupt_value)}")

            # 【调试日志】记录 interrupt 信息
            if isinstance(interrupt_value, dict):
                logger.info(f"[graph_nodes] interrupt_value 是 dict，keys: {list(interrupt_value.keys())}")
                if "confirmation_pending" in interrupt_value:
                    logger.info(f"[graph_nodes] ✅ interrupt_value 包含 confirmation_pending")
            else:
                logger.warning(f"[graph_nodes] ❌ interrupt_value 不是 dict，type: {type(interrupt_value)}")

            # 【关键修复】GraphInterrupt 会被 LangGraph 捕获，不会将返回值包含在 stream 输出中
            # 所以我们需要重新抛出异常，让 LangGraph 处理
            # LangGraph 会将 interrupt 信息保存到 checkpointer，客户端可以通过 get_state() 获取
            logger.info(f"[graph_nodes] 重新抛出 GraphInterrupt 异常，让 LangGraph 处理")
            raise
```

**处理步骤：**
1. ✅ 从异常中提取 `interrupt_value`（即 `confirmation_data`）
2. ✅ 记录日志（用于调试）
3. ✅ **重新抛出异常**（`raise`）- 让 LangGraph 框架处理

**为什么重新抛出？**
- LangGraph 框架需要这个异常来：
  - 保存状态到 checkpointer（持久化）
  - 生成 `__interrupt__` 节点输出
  - 停止图的执行

---

## 🔄 阶段 3：LangGraph 框架处理异常

### 3.1 框架捕获异常
**位置：** LangGraph 内部（框架代码）

**执行的操作：**
1. ✅ **保存状态到 checkpointer**
   - 保存当前图的状态
   - 保存 interrupt 数据
   - 标记图处于"中断"状态

2. ✅ **生成 `__interrupt__` 节点输出**
   - 节点名称：`"__interrupt__"`（LangGraph 框架定义的特殊节点名）
   - 节点值：`(Interrupt(value=confirmation_data),)`
   - **注意**：虽然框架使用字符串 `"__interrupt__"`，但项目代码中使用 `LANGGRAPH_INTERRUPT_KEY` 常量来引用它

3. ✅ **停止图的执行**
   - 不再执行后续节点
   - 图进入"等待恢复"状态

### 3.2 状态流输出
**格式：** `{"__interrupt__": (Interrupt(value=confirmation_data),)}`

**说明：**
- LangGraph 框架输出时使用字符串 `"__interrupt__"` 作为节点名
- 项目代码中使用 `LANGGRAPH_INTERRUPT_KEY` 常量来检测和处理这个节点
- 这是最佳实践，避免在代码中硬编码框架定义的字符串

---

## 🔄 阶段 4：chat.py 路由处理 interrupt

### 4.1 导入常量（最佳实践）
**位置：** `src/api/routes/chat.py:8-13`

```8:13:src/api/routes/chat.py
from src.api.formatters import (
    format_state_update,
    format_step_name,
    format_step_detail,
    LANGGRAPH_INTERRUPT_KEY,
)
```

**说明：**
- ✅ 使用 `LANGGRAPH_INTERRUPT_KEY` 常量而不是魔术字符串 `"__interrupt__"`
- ✅ 这是最佳实践，避免硬编码字符串，提高代码可维护性
- ✅ 常量定义在 `src/api/formatters.py:25`：`LANGGRAPH_INTERRUPT_KEY = "__interrupt__"`

### 4.2 检测 interrupt 节点
**位置：** `src/api/routes/chat.py:109-128`

```109:128:src/api/routes/chat.py
                # 【关键修复】处理 __interrupt__ 节点
                # 当 interrupt() 被调用时，LangGraph 生成一个 __interrupt__ 节点
                # node_update 是一个 tuple，包含 Interrupt 对象
                if node_name == LANGGRAPH_INTERRUPT_KEY and isinstance(node_update, tuple) and len(node_update) > 0:
                    logger.info(f"[chat路由] ✅ 检测到 {LANGGRAPH_INTERRUPT_KEY} 节点，提取 interrupt 信息")

                    # 从 tuple 中提取 Interrupt 对象
                    interrupt_obj = node_update[0]
                    if hasattr(interrupt_obj, 'value'):
                        interrupt_value = interrupt_obj.value
                        logger.info(f"[chat路由] interrupt_value keys: {list(interrupt_value.keys()) if isinstance(interrupt_value, dict) else 'N/A'}")

                        # 格式化并发送 interrupt 信息
                        if isinstance(interrupt_value, dict):
                            formatted = format_state_update({LANGGRAPH_INTERRUPT_KEY: interrupt_value}, None, 0)
                            formatted["data"]["execution_steps"] = execution_steps
                            formatted["data"]["step_details"] = step_details
                            yield f"data: {json.dumps(formatted, ensure_ascii=False)}\n\n"
                            logger.info(f"[chat路由] ✅ 已发送 interrupt 信息到前端（包含 confirmation_pending）")
                            continue  # 跳过后续处理
                    else:
                        logger.warning(f"[chat路由] ❌ interrupt_obj 没有 value 属性")
```

**处理步骤：**
1. ✅ 检测到 `node_name == LANGGRAPH_INTERRUPT_KEY`（使用常量，非魔术字符串）
2. ✅ 从 tuple 中提取 `Interrupt` 对象
3. ✅ 获取 `interrupt_obj.value`（即 `confirmation_data`）
4. ✅ 调用 `format_state_update({LANGGRAPH_INTERRUPT_KEY: interrupt_value}, ...)` 格式化数据
5. ✅ 通过 SSE 流发送给前端

**最佳实践说明：**
- ✅ 使用 `LANGGRAPH_INTERRUPT_KEY` 常量替代硬编码字符串
- ✅ 所有对 `__interrupt__` 的引用都通过常量管理
- ✅ 便于维护和重构，符合企业级代码规范

### 4.3 检查 final_state 中的 interrupt（备用检测）
**位置：** `src/api/routes/chat.py:202-217`

```202:217:src/api/routes/chat.py
            # 【关键修复】检查 snapshot.values 中是否有 __interrupt__ 字段
            if final_snapshot and hasattr(final_snapshot, 'values') and final_snapshot.values:
                from src.multi_agent.utils import state_to_dict
                final_state_dict = state_to_dict(final_snapshot.values)
                logger.info(f"[chat路由] final_state_dict keys: {list(final_state_dict.keys())}")

                # 检查是否有 __interrupt__ 字段
                if LANGGRAPH_INTERRUPT_KEY in final_state_dict:
                    logger.info(f"[chat路由] ✅ final_state 包含 {LANGGRAPH_INTERRUPT_KEY}，格式化并发送")
                    # 格式化并发送 interrupt 信息
                    formatted = format_state_update(final_state_dict, None, 0)
                    formatted["data"]["execution_steps"] = execution_steps
                    formatted["data"]["step_details"] = step_details
                    yield f"data: {json.dumps(formatted, ensure_ascii=False)}\n\n"
                else:
                    logger.warning(f"[chat路由] ❌ final_state 不包含 {LANGGRAPH_INTERRUPT_KEY}")
```

**说明：**
- ✅ 这是备用检测机制，用于检查最终状态中是否包含 interrupt
- ✅ 使用 `LANGGRAPH_INTERRUPT_KEY in final_state_dict` 而不是硬编码字符串
- ✅ 确保即使流式输出中未捕获到 interrupt，也能从最终状态中检测到

### 4.4 格式化 interrupt 数据
**位置：** `src/api/formatters.py:143-193`

**常量定义：**
```24:25:src/api/formatters.py
# LangGraph 原生的 interrupt 键名（这是 LangGraph 定义的，不是我们定义的）
LANGGRAPH_INTERRUPT_KEY = "__interrupt__"
```

**使用常量检测 interrupt：**
```143:149:src/api/formatters.py
    # 【LangGraph interrupt() 机制】优先检测 interrupt 信号
    # interrupt 是优先级最高的状态，会暂停图执行
    interrupt_data = None
    if node_update and isinstance(node_update, dict):
        interrupt_data = node_update.get(LANGGRAPH_INTERRUPT_KEY)
    if not interrupt_data:
        interrupt_data = state_update.get(LANGGRAPH_INTERRUPT_KEY)
```

**解析 interrupt 数据：**
```151:193:src/api/formatters.py
    # 【关键修复】interrupt_data 可能是 Interrupt 对象或包含 Interrupt 的 tuple
    # 需要解析获取实际的字典值
    if interrupt_data:
        original_interrupt_data = interrupt_data  # 保存原始值用于日志

        # 如果是 tuple，提取第一个元素
        if isinstance(interrupt_data, tuple) and len(interrupt_data) > 0:
            interrupt_data = interrupt_data[0]
            logger.info(f"[format_state_update] 从 tuple 提取 interrupt_data")

        # 如果是 Interrupt 对象（非 tuple），获取其 value 属性
        if not isinstance(interrupt_data, (tuple, dict)) and hasattr(interrupt_data, 'value'):
            interrupt_data = interrupt_data.value
            logger.info(f"[format_state_update] 从 Interrupt 对象提取 value")

        logger.info(f"[format_state_update] 解析后的 interrupt_data type: {type(interrupt_data)}, original: {type(original_interrupt_data)}")

    # 只有当 interrupt_data 是 dict 时才继续处理
    if interrupt_data and isinstance(interrupt_data, dict):
        # 解析 interrupt 数据
        parsed = InterruptData.from_dict(interrupt_data) if isinstance(interrupt_data, dict) else None

        result["data"]["interrupt"] = {
            "interrupt_type": parsed.interrupt_type.value if parsed else interrupt_data.get("interrupt_type", "confirmation"),
            "action_type": parsed.action_type if parsed else interrupt_data.get("action_type", ""),
            "display_message": parsed.display_message if parsed else interrupt_data.get("display_message", ""),
            "display_data": parsed.display_data if parsed else interrupt_data.get("display_data"),
        }
        result["data"]["response_type"] = "interrupt"
        result["data"]["role"] = "system"
        logger.info(f"[format_state_update] 检测到 interrupt 信号: type={result['data']['interrupt']['interrupt_type']}")

        # 【关键修复】从 interrupt_data 中提取额外的状态字段（如 confirmation_pending）
        # 因为当 GraphInterrupt 异常被抛出时，LangGraph 不会使用节点函数的返回值
        # 所以 confirmation_pending 等字段需要直接从 interrupt_data 中提取
        if isinstance(interrupt_data, dict):
            # 提取 confirmation_pending（前端显示用）
            if "confirmation_pending" in interrupt_data:
                result["data"]["confirmation_pending"] = interrupt_data["confirmation_pending"]
                logger.info(f"[format_state_update] 从 interrupt_data 提取 confirmation_pending: {interrupt_data['confirmation_pending']}")
            # 提取 conversation_phase（会话阶段）
            if "conversation_phase" in interrupt_data:
                result["data"]["conversation_phase"] = interrupt_data["conversation_phase"]
```

**格式化结果：**
```json
{
  "type": "state_update",
  "data": {
    "response_type": "interrupt",
    "interrupt": {
      "interrupt_type": "confirmation",
      "action_type": "create_order",
      "display_message": "确认创建订单？",
      "display_data": {...}
    },
    "confirmation_pending": {
      "confirmation_id": "xxx",
      "action_type": "create_order",
      "display_message": "确认创建订单？",
      "display_data": {...}
    },
    "conversation_phase": "order_creating"
  }
}
```

---

## 🔄 阶段 5：前端接收并显示确认界面

### 5.1 前端接收 SSE 消息
- 收到 `response_type: "interrupt"` 的消息
- 提取 `confirmation_pending` 数据
- 显示确认界面（按钮：确认/取消）

### 5.2 用户操作
- 用户点击"确认"或"取消"按钮
- 前端调用 `/api/confirmation/resolve` 接口

---

## 🔄 阶段 6：confirmation.py 路由处理确认

### 6.1 接收确认请求
**位置：** `src/api/routes/confirmation.py:26-72`

```26:72:src/api/routes/confirmation.py
@router.post("/confirmation/resolve")
async def resolve_confirmation(request: ConfirmationResolveRequest):
    """解析确认操作并恢复执行

    LangGraph 1.x interrupt() 机制：
    1. 使用 Command(resume=...) 恢复被中断的图
    2. interrupt() 调用会返回 resume 的值
    3. 图从中断点继续执行
    """
    try:
        from src.api.graph_manager import get_graph

        manager = get_confirmation_manager()
        confirmation = await manager.get_confirmation(request.confirmation_id)
        if not confirmation:
            raise ValueError("确认不存在")

        session_id = confirmation.session_id

        logger.info(
            f"[interrupt] 用户确认请求: confirmation_id={request.confirmation_id}, "
            f"confirmed={request.confirmed}, session_id={session_id}"
        )

        graph = await get_graph()
        config = {
            "configurable": {"thread_id": session_id, "session_id": session_id},
            "recursion_limit": 25
        }

        # 【LangGraph interrupt() 机制】resume 数据
        # 这个值会被 interrupt() 调用返回给 Agent
        resume_data = {
            "confirmed": request.confirmed,
            "confirmation_id": request.confirmation_id
        }

        # 解析确认（执行确认操作）
        result = await manager.resolve_confirmation(
            request.confirmation_id,
            request.confirmed,
        )

        logger.info(
            f"[interrupt] 确认操作已执行: confirmation_id={request.confirmation_id}, "
            f"status={result.status}"
        )
```

**处理步骤：**
1. ✅ 获取确认记录
2. ✅ 解析确认（更新确认状态）
3. ✅ 准备 `resume_data`（包含 `confirmed` 和 `confirmation_id`）

### 6.2 恢复图执行
**位置：** `src/api/routes/confirmation.py:75-98`

```75:98:src/api/routes/confirmation.py
        async def stream_response():
            """流式返回恢复执行的结果"""
            try:
                action_text = "已确认" if request.confirmed else "已取消"
                yield f"data: {json.dumps({'type': 'confirmation_resolved', 'message': f'{action_text}，正在继续处理...'}, ensure_ascii=False)}\n\n"

                # 【核心】使用 Command(resume=...) 恢复被 interrupt() 暂停的图
                resume_command = Command(resume=resume_data)

                logger.info(f"[interrupt] 调用 graph.astream(resume={resume_data})")

                async for formatted in accumulate_and_format_state_updates(
                    graph.astream(
                        command=resume_command,
                        config=config,
                        stream_mode="updates",
                        session_id=session_id
                    )
                ):
                    json_str = json.dumps(formatted, ensure_ascii=False)
                    yield f"data: {json_str}\n\n"

                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                logger.info(f"[interrupt] resume 流结束")
```

**关键操作：**
- ✅ 创建 `Command(resume=resume_data)`
- ✅ 调用 `graph.astream(command=resume_command, ...)`
- ✅ LangGraph 从 checkpointer 恢复状态
- ✅ **`interrupt()` 调用返回 `resume_data`**

---

## 🔄 阶段 7：order_agent.py 恢复执行

### 7.1 检测 interrupt 恢复
**位置：** `src/multi_agent/agents/order_agent.py:747-790`

```747:790:src/multi_agent/agents/order_agent.py
        # 【LangGraph interrupt() 机制】
        # 检查是否有待处理的 interrupt（从 resume 恢复）
        # 使用 InterruptState 类而不是魔术字符串
        if InterruptState.has_pending_interrupt(state.entities):
            interrupt_data = InterruptState.get_interrupt_data(state.entities)

            # 这是 interrupt 后的恢复执行
            # interrupt() 调用会返回 resume 的值
            resume_value = interrupt(interrupt_data)
            logger.info(f"[interrupt] resume 值: {resume_value}")

            # 使用框架函数检查 resume 值
            confirmed = is_resume_confirm(resume_value)

            if confirmed is True:
                # 用户确认，执行确认操作
                return await self._execute_confirmed_action(
                    state, interrupt_data, session_id
                )
            elif confirmed is False:
                # 用户取消
                updated_entities = {**state.entities}
                InterruptState.clear_interrupt_data(updated_entities)
                return {
                    "messages": messages + [
                        AIMessage(content="👌 已取消操作，请问还有其他需要帮助的吗？")
                    ],
                    "current_agent": self.name,
                    "confirmation_pending": None,
                    "entities": updated_entities,
                }
            else:
                # resume 值无效，取消操作
                logger.warning(f"[interrupt] resume 值无效: {resume_value}，取消操作")
                updated_entities = {**state.entities}
                InterruptState.clear_interrupt_data(updated_entities)
                return {
                    "messages": messages + [
                        AIMessage(content="确认信息无效，操作已取消。请问还有其他需要帮助的吗？")
                    ],
                    "current_agent": self.name,
                    "confirmation_pending": None,
                    "entities": updated_entities,
                }
```

**关键点：**
1. ✅ 检测到 `state.entities` 中有 interrupt 数据
2. ✅ **再次调用 `interrupt(interrupt_data)`**
3. ✅ **这次 `interrupt()` 返回 `resume_data`**（不是抛出异常！）
4. ✅ 从 `resume_value` 中提取 `confirmed` 状态
5. ✅ 根据确认结果执行相应操作

**为什么再次调用 `interrupt()`？**
- 第一次调用（720行）：抛出异常，暂停执行
- 第二次调用（755行）：返回 resume 值，恢复执行
- 这是 LangGraph 的设计：`interrupt()` 在恢复时会返回 resume 值

---

## 🔄 阶段 8：执行确认操作

### 8.1 用户确认（confirmed=True）
**位置：** `order_agent._execute_confirmed_action()`

- 执行实际的订单创建操作
- 返回成功消息
- 清除 interrupt 数据

### 8.2 用户取消（confirmed=False）
- 返回取消消息
- 清除 interrupt 数据
- 不执行操作

---

## 📊 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1: order_agent.py:720                                      │
│ return interrupt(confirmation_data)                              │
│   ↓                                                              │
│ 抛出 GraphInterrupt 异常                                         │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 2: graph_nodes.py:201                                      │
│ except GraphInterrupt as e:                                      │
│   - 提取 interrupt_value                                         │
│   - 记录日志                                                      │
│   - raise  # 重新抛出                                            │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 3: LangGraph 框架                                          │
│   - 保存状态到 checkpointer                                      │
│   - 生成 __interrupt__ 节点                                      │
│   - 停止图执行                                                    │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 4: chat.py:112                                             │
│ if node_name == LANGGRAPH_INTERRUPT_KEY:                        │
│   - 使用常量检测（最佳实践，避免魔术字符串）                      │
│   - 提取 interrupt_value                                         │
│   - 格式化数据                                                    │
│   - 发送 SSE 消息给前端                                          │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 5: 前端                                                     │
│   - 接收 interrupt 消息                                          │
│   - 显示确认界面                                                  │
│   - 用户点击确认/取消                                             │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 6: confirmation.py:26                                      │
│ POST /api/confirmation/resolve                                  │
│   - 创建 Command(resume=resume_data)                            │
│   - graph.astream(command=resume_command)                       │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 7: order_agent.py:750                                      │
│ if InterruptState.has_pending_interrupt():                      │
│   resume_value = interrupt(interrupt_data)  # 返回 resume 值    │
│   - 处理 resume_value                                            │
│   - 执行确认操作或取消操作                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 关键理解点

### 1. interrupt() 的双重行为
- **第一次调用（暂停时）**：抛出 `GraphInterrupt` 异常
- **第二次调用（恢复时）**：返回 `resume_data` 值

### 2. 为什么需要抛出异常？
- 异常是 LangGraph 识别"需要暂停"的唯一信号
- 框架通过捕获异常来：
  - 保存状态
  - 生成 interrupt 输出
  - 停止执行

### 3. 状态持久化
- interrupt 数据保存在 `state.entities` 中
- LangGraph 通过 checkpointer 持久化整个图状态
- 恢复时从 checkpointer 恢复状态

### 4. 数据传递路径
```
confirmation_data (order_agent)
  ↓ interrupt(confirmation_data) 抛出异常
GraphInterrupt 异常
  ↓ graph_nodes 提取
interrupt_value
  ↓ LangGraph 生成
__interrupt__ 节点（框架使用字符串 "__interrupt__"）
  ↓ chat.py 使用 LANGGRAPH_INTERRUPT_KEY 常量检测
前端 JSON 消息
  ↓ 用户确认
resume_data
  ↓ Command(resume=...)
interrupt() 返回 resume_data
  ↓ order_agent 处理
执行确认操作
```

### 5. 使用常量而非魔术字符串（最佳实践）
- ✅ **定义常量**：`LANGGRAPH_INTERRUPT_KEY = "__interrupt__"`（在 `formatters.py` 中）
- ✅ **统一使用**：所有代码都通过常量引用 `__interrupt__`，而不是硬编码字符串
- ✅ **优势**：
  - 统一管理：如需修改，只需改一处
  - 易于维护：避免拼写错误和重复
  - 类型安全：IDE 可提供自动补全
  - 符合企业级代码规范

---

## 📝 总结

`return interrupt(confirmation_data)` 的完整流程：

1. **抛出异常** → 暂停执行
2. **框架捕获** → 保存状态、生成输出
3. **路由处理** → 发送给前端
4. **用户响应** → 确认/取消
5. **恢复执行** → `interrupt()` 返回 resume 值
6. **继续处理** → 执行确认操作

这是一个**异步、持久化、可恢复**的执行流程，符合 LangGraph 1.x 的设计理念。

