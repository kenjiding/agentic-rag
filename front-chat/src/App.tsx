import { useCallback, useEffect, useState } from "react"
import { ChatLayout } from "@/components/chat/ChatLayout"
import { useStreamingChat } from "@/hooks/useStreamingChat"

// 生成唯一的 session ID
function generateSessionId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substring(2)
}

// 从 URL 路径获取 session ID
// 支持 /chat/:sessionId 格式
function getSessionIdFromPath(): string | null {
  const pathname = window.location.pathname

  // 匹配 /chat/:sessionId 格式
  const match = pathname.match(/^\/chat\/([a-zA-Z0-9_-]+)$/)
  if (match && match[1]) {
    return match[1]
  }

  return null
}

function App() {
  const [sessionId, setSessionId] = useState<string>("")
  const [ready, setReady] = useState(false)

  // 初始化 session ID
  useEffect(() => {
    const pathSessionId = getSessionIdFromPath()

    if (pathSessionId) {
      // URL 中已有 session ID，恢复会话
      console.log("🔄 恢复会话:", pathSessionId)
      setSessionId(pathSessionId)
      setReady(true)
    } else {
      // URL 中没有 session ID，生成新的并重定向
      const newSessionId = generateSessionId()
      console.log("🆕 创建新会话:", newSessionId)

      // 使用 replace 重定向到 /chat/:sessionId
      window.location.replace(`/chat/${newSessionId}`)
    }
  }, [])

  const {
    messages,
    isLoading,
    error,
    sendMessage,
    stop,
    clearMessages,
    confirmAction,
    cancelConfirmation,
    isProcessingConfirmation,
  } = useStreamingChat()

  // 包装 sendMessage，自动传入 sessionId
  const handleSend = useCallback(
    (content: string) => {
      if (sessionId) {
        sendMessage(content, sessionId)
      }
    },
    [sendMessage, sessionId]
  )

  // 包装 buyProduct，自动传入 sessionId
  const handleBuyProduct = useCallback(
    (productId: number) => {
      if (sessionId) {
        sendMessage(`购买产品 ID: ${productId}`, sessionId)
      }
    },
    [sendMessage, sessionId]
  )

  if (!ready) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-lg mb-2">正在创建新会话...</div>
          <div className="text-sm text-muted-foreground">Session ID: {sessionId}</div>
        </div>
      </div>
    )
  }

  return (
    <ChatLayout
      messages={messages}
      onSend={handleSend}
      onStop={stop}
      onClear={clearMessages}
      isLoading={isLoading}
      error={error}
      onConfirm={confirmAction}
      onCancel={cancelConfirmation}
      isProcessingConfirmation={isProcessingConfirmation}
      onBuyProduct={handleBuyProduct}
    />
  )
}

export default App
