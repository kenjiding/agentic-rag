import { useState, useCallback, useRef } from "react"
import { ChatMessage, StreamEvent, ConfirmationResolveResponse } from "@/types"

export function useStreamingChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isProcessingConfirmation, setIsProcessingConfirmation] = useState(false)
  const [isProcessingSelection, setIsProcessingSelection] = useState(false)
  const abortControllerRef = useRef<AbortController | null>(null)
  const currentMessageIdRef = useRef<string | null>(null)

  const sendMessage = useCallback(async (content: string, sessionId: string = "default") => {
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
                // 调试：打印选择数据
                if (data.data.pending_selection) {
                  console.log("🛍️ 收到选择请求:", data.data.pending_selection)
                }

                setMessages((prev) => {
                  const updated = [...prev]
                  const index = updated.findIndex(
                    (msg) => msg.id === assistantMessageId
                  )
                  if (index !== -1) {
                    const existing = updated[index]

                    // 处理内容
                    let newContent = existing.content
                    if (data.data?.content) {
                      const incomingContent = data.data.content
                      if (incomingContent.includes(existing.content) && existing.content) {
                        newContent = incomingContent
                      } else if (incomingContent && !existing.content) {
                        newContent = incomingContent
                      } else if (incomingContent !== existing.content) {
                        newContent = existing.content + incomingContent
                      }
                    }

                    // 合并 metadata
                    const existingMetadata = existing.metadata || {}
                    const newMetadata = {
                      current_agent: data.data?.current_agent ?? existingMetadata.current_agent,
                      tools_used: data.data?.tools_used ?? existingMetadata.tools_used,
                      execution_steps: data.data?.execution_steps ?? existingMetadata.execution_steps,
                      step_details: data.data?.step_details ?? existingMetadata.step_details,
                    }

                    updated[index] = {
                      ...existing,
                      content: newContent,
                      responseType: data.data?.response_type ?? existing.responseType ?? "text",
                      responseData: data.data?.response_data ?? existing.responseData,
                      confirmationPending: data.data?.confirmation_pending ?? existing.confirmationPending,
                      pendingSelection: data.data?.pending_selection ?? existing.pendingSelection,
                      metadata: newMetadata,
                      isStreaming: true,
                    }
                  }
                  return updated
                })
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
  }, [])

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
      content: "已确认，正在处理...",
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
                  // 更新消息内容（与sendMessage中的逻辑相同）
                  setMessages((prev) => {
                    const updated = [...prev]
                    const index = updated.findIndex((msg) => msg.id === assistantMessageId)
                    if (index !== -1) {
                      const existing = updated[index]

                      let newContent = existing.content
                      if (data.data?.content) {
                        const incomingContent = data.data.content
                        if (incomingContent.includes(existing.content) && existing.content) {
                          newContent = incomingContent
                        } else if (incomingContent && !existing.content) {
                          newContent = incomingContent
                        } else if (incomingContent !== existing.content) {
                          newContent = existing.content + incomingContent
                        }
                      }

                      const existingMetadata = existing.metadata || {}
                      const newMetadata = {
                        current_agent: data.data?.current_agent ?? existingMetadata.current_agent,
                        tools_used: data.data?.tools_used ?? existingMetadata.tools_used,
                        execution_steps: data.data?.execution_steps ?? existingMetadata.execution_steps,
                        step_details: data.data?.step_details ?? existingMetadata.step_details,
                      }

                      updated[index] = {
                        ...existing,
                        content: newContent,
                        responseType: data.data?.response_type ?? existing.responseType ?? "text",
                        responseData: data.data?.response_data ?? existing.responseData,
                        confirmationPending: data.data?.confirmation_pending ?? existing.confirmationPending,
                        pendingSelection: data.data?.pending_selection ?? existing.pendingSelection,
                        metadata: newMetadata,
                        isStreaming: true,
                      }
                    }
                    return updated
                  })
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
  }, [])

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

  // 选择产品
  const selectProduct = useCallback(async (selectionId: string, productId: string) => {
    setIsProcessingSelection(true)

    // 先清除选择状态
    setMessages((prev) =>
      prev.map((msg) => ({
        ...msg,
        pendingSelection: undefined,
      }))
    )

    // 创建新的助手消息来接收后续流式响应
    const assistantMessageId = `assistant-${Date.now()}`
    currentMessageIdRef.current = assistantMessageId
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "已选择商品，正在为您创建订单...",
      responseType: "text",
      timestamp: new Date(),
      isStreaming: true,
    }
    setMessages((prev) => [...prev, assistantMessage])

    try {
      const response = await fetch("/api/selection/resolve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selection_id: selectionId, selected_option_id: productId }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || "选择产品失败")
      }

      // 处理流式响应（与sendMessage类似）
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

              if (data.type === "selection_resolved") {
                // 选择已解析，继续等待后续状态更新
                console.log("✅ 选择已解析:", data.message)
                continue
              }

              if (data.type === "state_update" && data.data) {
                // 更新消息内容（与sendMessage中的逻辑相同）
                setMessages((prev) => {
                  const updated = [...prev]
                  const index = updated.findIndex((msg) => msg.id === assistantMessageId)
                  if (index !== -1) {
                    const existing = updated[index]

                    let newContent = existing.content
                    if (data.data?.content) {
                      const incomingContent = data.data.content
                      if (incomingContent.includes(existing.content) && existing.content) {
                        newContent = incomingContent
                      } else if (incomingContent && !existing.content) {
                        newContent = incomingContent
                      } else if (incomingContent !== existing.content) {
                        newContent = existing.content + incomingContent
                      }
                    }

                    const existingMetadata = existing.metadata || {}
                    const newMetadata = {
                      current_agent: data.data?.current_agent ?? existingMetadata.current_agent,
                      tools_used: data.data?.tools_used ?? existingMetadata.tools_used,
                      execution_steps: data.data?.execution_steps ?? existingMetadata.execution_steps,
                      step_details: data.data?.step_details ?? existingMetadata.step_details,
                    }

                    updated[index] = {
                      ...existing,
                      content: newContent,
                      responseType: data.data?.response_type ?? existing.responseType ?? "text",
                      responseData: data.data?.response_data ?? existing.responseData,
                      confirmationPending: data.data?.confirmation_pending ?? existing.confirmationPending,
                      pendingSelection: data.data?.pending_selection ?? existing.pendingSelection,
                      metadata: newMetadata,
                      isStreaming: true,
                    }
                  }
                  return updated
                })
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
    } catch (err: any) {
      setError(err.message || "选择产品失败")
      setMessages((prev) => {
        const updated = [...prev]
        const index = updated.findIndex((msg) => msg.id === assistantMessageId)
        if (index !== -1) {
          updated[index] = {
            ...updated[index],
            content: `错误: ${err.message || "选择产品失败"}`,
            isStreaming: false,
          }
        }
        return updated
      })
    } finally {
      setIsProcessingSelection(false)
      currentMessageIdRef.current = null
    }
  }, [])

  // 取消选择
  const cancelSelection = useCallback(async (selectionId: string) => {
    setIsProcessingSelection(true)
    try {
      const response = await fetch("/api/selection/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selection_id: selectionId }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || "取消选择失败")
      }

      // 清除消息中的选择状态，并添加取消消息
      setMessages((prev) => {
        const updated: ChatMessage[] = prev.map((msg) => ({
          ...msg,
          pendingSelection: undefined,
        }))

        updated.push({
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: "已取消选择",
          responseType: "text",
          timestamp: new Date(),
        })

        return updated
      })
    } catch (err: any) {
      setError(err.message || "取消选择失败")
    } finally {
      setIsProcessingSelection(false)
    }
  }, [])

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
    selectProduct,
    cancelSelection,
    isProcessingSelection,
  }
}

