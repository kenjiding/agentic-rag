import { MessageList } from "./MessageList"
import { ChatInput } from "./ChatInput"
import { ChatMessage } from "@/types"
import { Button } from "@/components/ui/button"
import { Trash2 } from "lucide-react"

interface ChatLayoutProps {
  messages: ChatMessage[]
  onSend: (message: string) => void
  onStop?: () => void
  onClear?: () => void
  isLoading?: boolean
  error?: string | null
}

export function ChatLayout({
  messages,
  onSend,
  onStop,
  onClear,
  isLoading,
  error,
}: ChatLayoutProps) {
  return (
    <div className="flex flex-col h-screen bg-background">
      {/* 头部 */}
      <header className="border-b bg-card px-4 py-3 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold">AI 智能客服</h1>
          <p className="text-sm text-muted-foreground">
            电商智能助手，支持商品搜索、订单查询等功能
          </p>
        </div>
        {onClear && messages.length > 0 && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onClear}
            title="清空对话"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </header>

      {/* 错误提示 */}
      {error && (
        <div className="bg-destructive/10 text-destructive px-4 py-2 text-sm border-b">
          {error}
        </div>
      )}

      {/* 消息列表 */}
      <MessageList messages={messages} />

      {/* 输入框 */}
      <ChatInput
        onSend={onSend}
        onStop={onStop}
        isLoading={isLoading}
      />
    </div>
  )
}

