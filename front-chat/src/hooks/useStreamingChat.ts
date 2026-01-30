import { useState, useCallback, useRef } from "react"
import {
  ChatMessage,
  StreamEvent,
  ConfirmationResolveResponse,
  InterruptPayload,
} from "@/types"

export function useStreamingChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isProcessingConfirmation, setIsProcessingConfirmation] = useState(false)
  const [pendingInterrupt, setPendingInterrupt] = useState<InterruptPayload | null>(null)
  const [isProcessingInterrupt, setIsProcessingInterrupt] = useState(false)
  const abortControllerRef = useRef<AbortController | null>(null)
  const currentMessageIdRef = useRef<string | null>(null)

  /**
   * 通用的消息更新函数
   * 统一处理 state_update 事件的消息更新逻辑
   * 
   * 核心原则：state_update 中的 content 直接替换现有内容，不追加
   * 这样确保临时状态消息（如 confirmation_resolved）可以被最终结果替换
   */
  const updateMessageWithStateUpdate = useCallback((
    messageId: string,
    stateUpdateData: StreamEvent["data"]
  ) => {
    setMessages((prev) => {
      const updated = [...prev]
      const index = updated.findIndex((msg) => msg.id === messageId)
      if (index === -1) {
        return updated
      }

      const existing = updated[index]
      const existingMetadata = existing.metadata || {}

      // content 直接替换，不追加
      const newContent = stateUpdateData?.content ?? existing.content

      const shouldAttachInterrupt = stateUpdateData?.response_type === "interrupt"
      const newInterrupt: InterruptPayload | undefined = shouldAttachInterrupt
        ? {
            response_type: "interrupt",
            role: (stateUpdateData?.role as any) ?? "assistant",
            content: (stateUpdateData?.content as any) ?? newContent ?? "",
            interrupt_type: (stateUpdateData?.interrupt_type as any),
            action_type: (stateUpdateData?.action_type as any) ?? "",
            action_data: (stateUpdateData?.action_data as any) ?? {},
            display_message: (stateUpdateData?.display_message as any) ?? (stateUpdateData?.content as any) ?? "",
            display_data: (stateUpdateData?.display_data as any),
            metadata: (stateUpdateData?.metadata as any),
            response_data: stateUpdateData?.response_data,
          }
        : existing.interrupt

      // 合并 metadata
      const newMetadata = {
        current_agent: stateUpdateData?.current_agent ?? existingMetadata.current_agent,
        tools_used: stateUpdateData?.tools_used ?? existingMetadata.tools_used,
        execution_steps: stateUpdateData?.execution_steps ?? existingMetadata.execution_steps,
        step_details: stateUpdateData?.step_details ?? existingMetadata.step_details,
      }

      updated[index] = {
        ...existing,
        content: newContent,
        responseType: stateUpdateData?.response_type ?? existing.responseType ?? "text",
        responseData: stateUpdateData?.response_data ?? existing.responseData,
        confirmationPending: stateUpdateData?.confirmation_pending ?? existing.confirmationPending,
        interrupt: newInterrupt,
        metadata: newMetadata,
        isStreaming: true,
      }

      return updated
    })
  }, [])

  const clearInterrupts = useCallback(() => {
    setMessages((prev) =>
      prev.map((m) => ({
        ...m,
        interrupt: undefined,
        // keep responseType/content as-is; history remains visible
      }))
    )
  }, [])

  const resumeInterrupt = useCallback(async (
    resumeData: Record<string, any>,
    sessionId: string = "default"
  ) => {
    setIsProcessingInterrupt(true)
    setError(null)

    // 新建助手消息占位符接收后续流式响应
    const assistantMessageId = `assistant-${Date.now()}`
    currentMessageIdRef.current = assistantMessageId
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      responseType: "text",
      timestamp: new Date(),
      isStreaming: true,
    }
    setMessages((prev) => [...prev, assistantMessage])

    try {
      const response = await fetch("/api/interrupt/resume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, resume_data: resumeData }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      if (!reader) throw new Error("No response body")

      let buffer = ""
      let isReading = true

      while (isReading) {
        const { done, value } = await reader.read()
        if (done) {
          isReading = false
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          try {
            const evt = JSON.parse(line.slice(6)) as StreamEvent

            if (evt.type === "error") {
              setError(evt.error || "未知错误")
              break
            }

            if (evt.type === "interrupt_resumed") {
              setMessages((prev) => {
                const updated = [...prev]
                const index = updated.findIndex((msg) => msg.id === assistantMessageId)
                if (index !== -1) {
                  updated[index] = {
                    ...updated[index],
                    content: evt.message || "已收到输入，继续处理...",
                  }
                }
                return updated
              })
              continue
            }

            if (evt.type === "state_update" && evt.data) {
              if (evt.data.response_type === "interrupt") {
                // 同一轮恢复过程中再次触发 interrupt，前端需要进入新的 pending 状态
                const payload: InterruptPayload = {
                  response_type: "interrupt",
                  role: (evt.data.role as any) ?? "assistant",
                  content: (evt.data.content as any) ?? "",
                  interrupt_type: (evt.data.interrupt_type as any),
                  action_type: (evt.data.action_type as any) ?? "",
                  action_data: (evt.data.action_data as any) ?? {},
                  display_message: (evt.data.display_message as any) ?? (evt.data.content as any) ?? "",
                  display_data: (evt.data.display_data as any),
                  metadata: (evt.data.metadata as any),
                  response_data: evt.data.response_data,
                }
                setPendingInterrupt(payload)
              }
              updateMessageWithStateUpdate(assistantMessageId, evt.data)
            }

            if (evt.type === "done") {
              setMessages((prev) => {
                const updated = [...prev]
                const index = updated.findIndex((msg) => msg.id === assistantMessageId)
                if (index !== -1) {
                  updated[index] = {
                    ...updated[index],
                    isStreaming: false,
                  }
                }
                return updated
              })
              break
            }
          } catch (e) {
            console.error("Failed to parse SSE data:", e)
          }
        }
      }
    } catch (err: any) {
      setError(err.message || "恢复失败")
      setMessages((prev) => {
        const updated = [...prev]
        const index = updated.findIndex((msg) => msg.id === assistantMessageId)
        if (index !== -1) {
          updated[index] = {
            ...updated[index],
            content: `错误: ${err.message || "恢复失败"}`,
            isStreaming: false,
          }
        }
        return updated
      })
    } finally {
      setIsProcessingInterrupt(false)
      currentMessageIdRef.current = null
    }
  }, [updateMessageWithStateUpdate])

  const sendMessage = useCallback(async (content: string, sessionId: string = "default") => {
    // 若当前有待处理 interrupt：input 类型允许用底部输入框直接 resume
    if (pendingInterrupt) {
      if (pendingInterrupt.interrupt_type === "input") {
        const userMessage: ChatMessage = {
          id: `user-${Date.now()}`,
          role: "user",
          content,
          responseType: "text",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, userMessage])

        const resumeData = {
          input_value: content,
          action_type: pendingInterrupt.action_type,
          action_data: pendingInterrupt.action_data,
          metadata: pendingInterrupt.metadata ?? {},
        }

        setPendingInterrupt(null)
        clearInterrupts()
        await resumeInterrupt(resumeData, sessionId)
        return
      }

      // selection 类型必须通过卡片交互，避免 resume_data 不合法
      setError("当前需要选择，请在卡片中完成选择后提交。")
      return
    }

    // 添加用户消息
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
      responseType: "text",
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMessage])

    // 创建助手消息占位符
    const assistantMessageId = `assistant-${Date.now()}`
    currentMessageIdRef.current = assistantMessageId
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      responseType: "text",
      timestamp: new Date(),
      isStreaming: true,
    }
    setMessages((prev) => [...prev, assistantMessage])

    setIsLoading(true)
    setError(null)

    // 创建 AbortController
    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: content,
          session_id: sessionId,
          stream: true,
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error("No response body")
      }

      let buffer = ""
      let isReading = true

      while (isReading) {
        const { done, value } = await reader.read()
        if (done) {
          isReading = false
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n\n")
        buffer = lines.pop() || ""
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6)) as StreamEvent

              if (data.type === "error") {
                setError(data.error || "未知错误")
                break
              }
              if (data.type === "state_update" && data.data) {
                // 调试：打印确认数据
                if (data.data.confirmation_pending) {
                  console.log("🔔 收到确认请求:", data.data.confirmation_pending)
                }

                // 如果收到 interrupt，需要设置 pendingInterrupt 状态
                // 这样用户在对话框输入时才能正确触发 resume
                if (data.data.response_type === "interrupt") {
                  const payload: InterruptPayload = {
                    response_type: "interrupt",
                    role: (data.data.role as any) ?? "assistant",
                    content: (data.data.content as any) ?? "",
                    interrupt_type: (data.data.interrupt_type as any),
                    action_type: (data.data.action_type as any) ?? "",
                    action_data: (data.data.action_data as any) ?? {},
                    display_message: (data.data.display_message as any) ?? (data.data.content as any) ?? "",
                    display_data: (data.data.display_data as any),
                    metadata: (data.data.metadata as any),
                    response_data: data.data.response_data,
                  }
                  setPendingInterrupt(payload)
                }

                updateMessageWithStateUpdate(assistantMessageId, data.data)
              }

              if (data.type === "done") {
                setMessages((prev) => {
                  const updated = [...prev]
                  const index = updated.findIndex(
                    (msg) => msg.id === assistantMessageId
                  )
                  if (index !== -1) {
                    updated[index] = {
                      ...updated[index],
                      isStreaming: false,
                    }
                  }
                  return updated
                })
                break
              }
            } catch (e) {
              console.error("Failed to parse SSE data:", e)
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === "AbortError") {
        // 用户取消，不做处理
        return
      }
      setError(err.message || "发送消息失败")
      setMessages((prev) => {
        const updated = [...prev]
        const index = updated.findIndex(
          (msg) => msg.id === assistantMessageId
        )
        if (index !== -1) {
          updated[index] = {
            ...updated[index],
            content: `错误: ${err.message || "发送消息失败"}`,
            isStreaming: false,
          }
        }
        return updated
      })
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
      currentMessageIdRef.current = null
    }
  }, [pendingInterrupt, clearInterrupts, resumeInterrupt, updateMessageWithStateUpdate])

  const stop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setIsLoading(false)
      setMessages((prev) => {
        const updated = [...prev]
        const index = updated.findIndex(
          (msg) => msg.id === currentMessageIdRef.current
        )
        if (index !== -1) {
          updated[index] = {
            ...updated[index],
            isStreaming: false,
          }
        }
        return updated
      })
    }
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    setError(null)
    setPendingInterrupt(null)
  }, [])

  // 确认操作
  const confirmAction = useCallback(async (confirmationId: string) => {
    setIsProcessingConfirmation(true)

    // 先清除确认状态
    setMessages((prev) =>
      prev.map((msg) => ({
        ...msg,
        confirmationPending: undefined,
      }))
    )

    // 创建新的助手消息来接收后续流式响应
    const assistantMessageId = `assistant-${Date.now()}`
    currentMessageIdRef.current = assistantMessageId
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      responseType: "text",
      timestamp: new Date(),
      isStreaming: true,
    }
    setMessages((prev) => [...prev, assistantMessage])

    try {
      const response = await fetch("/api/confirmation/resolve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmation_id: confirmationId, confirmed: true }),
      })

      if (!response.ok) {
        // 尝试解析错误响应（可能是 JSON）
        try {
          const errorData = await response.json()
          throw new Error(errorData.detail || "确认操作失败")
        } catch {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
      }

      // 检查 Content-Type 判断是流式响应还是 JSON 响应
      const contentType = response.headers.get("content-type") || ""
      if (contentType.includes("text/event-stream")) {
        // 处理流式响应（任务链模式）
        const reader = response.body?.getReader()
        const decoder = new TextDecoder()

        if (!reader) {
          throw new Error("No response body")
        }

        let buffer = ""
        let isReading = true

        while (isReading) {
          const { done, value } = await reader.read()
          if (done) {
            isReading = false
            break
          }

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n\n")
          buffer = lines.pop() || ""

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6)) as StreamEvent

                if (data.type === "error") {
                  setError(data.error || "未知错误")
                  break
                }

                if (data.type === "confirmation_resolved") {
                  // 确认已解析，更新消息内容
                  console.log("✅ 确认已解析:", data.message)
                  setMessages((prev) => {
                    const updated = [...prev]
                    const index = updated.findIndex((msg) => msg.id === assistantMessageId)
                    if (index !== -1) {
                      updated[index] = {
                        ...updated[index],
                        content: data.message || "操作已完成",
                      }
                    }
                    return updated
                  })
                  continue
                }

                if (data.type === "state_update" && data.data) {
                  updateMessageWithStateUpdate(assistantMessageId, data.data)
                }

                if (data.type === "done") {
                  setMessages((prev) => {
                    const updated = [...prev]
                    const index = updated.findIndex((msg) => msg.id === assistantMessageId)
                    if (index !== -1) {
                      updated[index] = {
                        ...updated[index],
                        isStreaming: false,
                      }
                    }
                    return updated
                  })
                  break
                }
              } catch (e) {
                console.error("Failed to parse SSE data:", e)
              }
            }
          }
        }
      } else {
        // 处理 JSON 响应（非任务链模式）
        const result: ConfirmationResolveResponse = await response.json()

        // 更新消息内容
        setMessages((prev) => {
          const updated = [...prev]
          const index = updated.findIndex((msg) => msg.id === assistantMessageId)
          if (index !== -1) {
            updated[index] = {
              ...updated[index],
              content: result.message || "操作已完成",
              isStreaming: false,
            }
          }
          return updated
        })
      }
    } catch (err: any) {
      setError(err.message || "确认操作失败")
      setMessages((prev) => {
        const updated = [...prev]
        const index = updated.findIndex((msg) => msg.id === assistantMessageId)
        if (index !== -1) {
          updated[index] = {
            ...updated[index],
            content: `错误: ${err.message || "确认操作失败"}`,
            isStreaming: false,
          }
        }
        return updated
      })
    } finally {
      setIsProcessingConfirmation(false)
      currentMessageIdRef.current = null
    }
  }, [updateMessageWithStateUpdate])

  // 取消确认
  const cancelConfirmation = useCallback(async (confirmationId: string) => {
    setIsProcessingConfirmation(true)
    try {
      const response = await fetch("/api/confirmation/resolve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmation_id: confirmationId, confirmed: false }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || "取消操作失败")
      }

      // 清除消息中的确认状态，并添加取消消息
      setMessages((prev) => {
        const updated: ChatMessage[] = prev.map((msg) => ({
          ...msg,
          confirmationPending: undefined,
        }))

        updated.push({
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: "操作已取消",
          responseType: "text",
          timestamp: new Date(),
        })

        return updated
      })
    } catch (err: any) {
      setError(err.message || "取消操作失败")
    } finally {
      setIsProcessingConfirmation(false)
    }
  }, [])

  const submitInterrupt = useCallback(async (
    resumeData: Record<string, any>,
    sessionId: string = "default"
  ) => {
    setPendingInterrupt(null)
    clearInterrupts()
    await resumeInterrupt(resumeData, sessionId)
  }, [clearInterrupts, resumeInterrupt])

  // 购买产品
  const buyProduct = useCallback(async (productId: number) => {
    // 发送购买消息，让后端处理购买流程
    await sendMessage(`购买产品 ID: ${productId}`)
  }, [sendMessage])

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    stop,
    clearMessages,
    confirmAction,
    cancelConfirmation,
    isProcessingConfirmation,
    pendingInterrupt,
    submitInterrupt,
    isProcessingInterrupt,
    buyProduct,
  }
}

