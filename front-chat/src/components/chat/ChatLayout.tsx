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
  onConfirm?: (confirmationId: string) => void
  onCancel?: (confirmationId: string) => void
  isProcessingConfirmation?: boolean
  onSelectProduct?: (selectionId: string, productId: string) => void
  onCancelSelection?: (selectionId: string) => void
  isProcessingSelection?: boolean
}

export function ChatLayout({
  messages,
  onSend,
  onStop,
  onClear,
  isLoading,
  error,
  onConfirm,
  onCancel,
  isProcessingConfirmation,
  onSelectProduct,
  onCancelSelection,
  isProcessingSelection,
}: ChatLayoutProps) {
  return (
    <div className="flex flex-col h-screen bg-background">
      {/* 头部 */}
      <header className="border-b bg-card px-3 sm:px-4 py-2 sm:py-3 flex items-center justify-between">
        <div className="flex-1 min-w-0">
          <h1 className="text-base sm:text-lg font-semibold truncate">AI 智能客服</h1>
          <p className="text-xs sm:text-sm text-muted-foreground hidden sm:block">
            电商智能助手，支持商品搜索、订单查询等功能
          </p>
        </div>
        {onClear && messages.length > 0 && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onClear}
            title="清空对话"
            className="shrink-0 ml-2"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </header>

      {/* 错误提示 */}
      {error && (
        <div className="bg-destructive/10 text-destructive px-3 sm:px-4 py-2 text-xs sm:text-sm border-b">
          {error}
        </div>
      )}

      {/* 消息列表 */}
      <MessageList
        messages={messages}
        onConfirm={onConfirm}
        onCancel={onCancel}
        isProcessingConfirmation={isProcessingConfirmation}
        onSelectProduct={onSelectProduct}
        onCancelSelection={onCancelSelection}
        isProcessingSelection={isProcessingSelection}
      />

      {/* 输入框 */}
      <ChatInput
        onSend={onSend}
        onStop={onStop}
        isLoading={isLoading}
      />
    </div>
  )
}

