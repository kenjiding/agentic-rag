import { useState, useCallback, useRef } from "react"
import { ChatMessage, StreamEvent } from "@/types"

export function useStreamingChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const currentMessageIdRef = useRef<string | null>(null)

  const sendMessage = useCallback(async (content: string, sessionId: string = "default") => {
    // 添加用户消息
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
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

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

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

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    stop,
    clearMessages,
  }
}

