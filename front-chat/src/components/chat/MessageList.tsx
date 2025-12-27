import { ChatMessage } from "@/types"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ProductGrid } from "@/components/business/ProductCard"
import { OrderList } from "@/components/business/OrderTracker"
import { ExecutionSteps } from "./ExecutionSteps"
import { ConfirmationDialog } from "./ConfirmationDialog"
import { ProductSelectionDialog } from "./ProductSelectionDialog"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { motion } from "framer-motion"
import { User, Bot } from "lucide-react"
import { cn } from "@/lib/utils"
import { useEffect, useRef } from "react"

interface MessageListProps {
  messages: ChatMessage[]
  onConfirm?: (confirmationId: string) => void
  onCancel?: (confirmationId: string) => void
  isProcessingConfirmation?: boolean
  onSelectProduct?: (selectionId: string, productId: string) => void
  onCancelSelection?: (selectionId: string) => void
  isProcessingSelection?: boolean
}

export function MessageList({
  messages,
  onConfirm,
  onCancel,
  isProcessingConfirmation = false,
  onSelectProduct,
  onCancelSelection,
  isProcessingSelection = false,
}: MessageListProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const lastMessageRef = useRef<ChatMessage | null>(null)

  // 自动滚动到底部 - 当有新消息或消息内容更新时
  useEffect(() => {
    const scrollToBottom = () => {
      // 查找ScrollArea的viewport元素
      const viewport = document.querySelector('[data-radix-scroll-area-viewport]') as HTMLElement
      if (viewport) {
        // 检查是否接近底部（在100px内），如果是则自动滚动
        const isNearBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 100

        // 如果有新消息或正在流式传输，自动滚动
        const lastMessage = messages[messages.length - 1]
        const hasNewMessage = lastMessage && lastMessage.id !== lastMessageRef.current?.id
        const isStreaming = lastMessage?.isStreaming

        if (hasNewMessage || isStreaming || isNearBottom) {
          requestAnimationFrame(() => {
            if (viewport) {
              viewport.scrollTo({
                top: viewport.scrollHeight,
                behavior: isStreaming ? 'auto' : 'smooth'
              })
            }
          })
        }

        if (lastMessage) {
          lastMessageRef.current = lastMessage
        }
      }
    }

    const timer = setTimeout(scrollToBottom, 100)
    return () => clearTimeout(timer)
  }, [messages])

  return (
    <ScrollArea className="flex-1">
      <div className="p-3 sm:p-4">
        <div ref={scrollRef} className="space-y-3 sm:space-y-4">
          {messages.map((message, index) => (
            <MessageItem
              key={message.id}
              message={message}
              isLast={index === messages.length - 1}
              onConfirm={onConfirm}
              onCancel={onCancel}
              isProcessingConfirmation={isProcessingConfirmation}
              onSelectProduct={onSelectProduct}
              onCancelSelection={onCancelSelection}
              isProcessingSelection={isProcessingSelection}
            />
          ))}
        </div>
      </div>
    </ScrollArea>
  )
}

interface MessageItemProps {
  message: ChatMessage
  isLast: boolean
  onConfirm?: (confirmationId: string) => void
  onCancel?: (confirmationId: string) => void
  isProcessingConfirmation?: boolean
  onSelectProduct?: (selectionId: string, productId: string) => void
  onCancelSelection?: (selectionId: string) => void
  isProcessingSelection?: boolean
}

function MessageItem({
  message,
  isLast,
  onConfirm,
  onCancel,
  isProcessingConfirmation = false,
  onSelectProduct,
  onCancelSelection,
  isProcessingSelection = false,
}: MessageItemProps) {
  const isUser = message.role === "user"

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={cn(
        "flex gap-2 sm:gap-3",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="flex h-7 w-7 sm:h-8 sm:w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Bot className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
        </div>
      )}

      <div
        className={cn(
          "flex max-w-[85%] sm:max-w-[80%] flex-col gap-1.5 sm:gap-2 rounded-lg px-3 py-2 sm:px-4 sm:py-3",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted"
        )}
      >
        {/* 产品列表 - 根据 responseType 渲染 */}
        {/* 注意：如果有 pendingSelection，产品列表会在选择对话框中显示，不需要单独渲染 */}
        {message.responseType === "product_list" && message.responseData?.products && !message.pendingSelection && (
          <ProductGrid products={message.responseData.products} />
        )}

        {/* 订单列表 */}
        {message.responseType === "order_list" && message.responseData?.orders && (
          <OrderList orders={message.responseData.orders} />
        )}

        {/* 混合类型 - 同时包含产品和订单 */}
        {message.responseType === "mixed" && message.responseData && (
          <>
            {message.responseData.products && (
              <ProductGrid products={message.responseData.products} />
            )}
            {message.responseData.orders && (
              <OrderList orders={message.responseData.orders} />
            )}
          </>
        )}

        {/* 文本内容 - 始终显示 */}
        {message.content && (
          <div className="prose prose-sm dark:prose-invert max-w-none text-sm sm:text-base">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code: ({ inline, className, children, ...props }: any) => {
                  const match = /language-(\w+)/.exec(className || "")
                  return !inline && match ? (
                    <pre className="bg-muted p-2 sm:p-3 rounded-md overflow-x-auto text-xs sm:text-sm">
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </pre>
                  ) : (
                    <code className="bg-muted px-1.5 py-0.5 rounded text-xs sm:text-sm" {...props}>
                      {children}
                    </code>
                  )
                },
                p: ({ children }) => <p className="mb-1.5 sm:mb-2 last:mb-0 text-sm sm:text-base leading-relaxed">{children}</p>,
                ul: ({ children }) => <ul className="list-disc pl-3 sm:pl-4 mb-1.5 sm:mb-2 text-sm sm:text-base">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-3 sm:pl-4 mb-1.5 sm:mb-2 text-sm sm:text-base">{children}</ol>,
                li: ({ children }) => <li className="mb-0.5 sm:mb-1">{children}</li>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-primary pl-3 sm:pl-4 italic my-1.5 sm:my-2 text-sm sm:text-base">
                    {children}
                  </blockquote>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* 执行步骤 - 只在流式传输时显示 */}
        {message.isStreaming && message.metadata?.execution_steps && message.metadata.execution_steps.length > 0 && (
          <ExecutionSteps
            steps={message.metadata.execution_steps}
            stepDetails={message.metadata.step_details}
            isActive={true}
          />
        )}

        {/* 确认对话框 */}
        {(() => {
          // 调试：打印确认状态
          if (message.confirmationPending) {
            console.log("✅ 消息包含 confirmationPending:", message.confirmationPending)
            console.log("  onConfirm:", !!onConfirm, "onCancel:", !!onCancel)
          }
          return null
        })()}
        {message.confirmationPending && onConfirm && onCancel && (
          <ConfirmationDialog
            confirmation={message.confirmationPending}
            onConfirm={onConfirm}
            onCancel={onCancel}
            isProcessing={isProcessingConfirmation}
          />
        )}

        {/* 产品选择对话框 */}
        {(() => {
          // 调试：打印选择状态
          if (message.pendingSelection) {
            console.log("🛍️ 消息包含 pendingSelection:", message.pendingSelection)
            console.log("  onSelectProduct:", !!onSelectProduct, "onCancelSelection:", !!onCancelSelection)
          }
          return null
        })()}
        {message.pendingSelection && onSelectProduct && onCancelSelection && (
          <ProductSelectionDialog
            selection={message.pendingSelection}
            onSelect={onSelectProduct}
            onCancel={onCancelSelection}
            isProcessing={isProcessingSelection}
          />
        )}

        {/* 流式输入指示器 */}
        {message.isStreaming && isLast && (
          <div className="flex items-center gap-1 text-muted-foreground text-xs mt-1.5 sm:mt-2">
            <span className="animate-pulse">●</span>
            <span>正在输入...</span>
          </div>
        )}

        {/* 时间戳 */}
        <div
          className={cn(
            "text-[10px] sm:text-xs opacity-70 mt-0.5",
            isUser ? "text-right" : "text-left"
          )}
        >
          {message.timestamp.toLocaleTimeString("zh-CN", {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </div>
      </div>

      {isUser && (
        <div className="flex h-7 w-7 sm:h-8 sm:w-8 shrink-0 items-center justify-center rounded-full bg-secondary">
          <User className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
        </div>
      )}
    </motion.div>
  )
}

