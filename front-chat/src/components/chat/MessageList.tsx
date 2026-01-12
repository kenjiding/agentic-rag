import { ChatMessage } from "@/types"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ProductGrid } from "@/components/business/ProductCard"
import { OrderList } from "@/components/business/OrderTracker"
import { ProductComparison } from "@/components/business/ProductComparison"
import { ExecutionSteps } from "./ExecutionSteps"
import { ConfirmationDialog } from "./ConfirmationDialog"
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
  onBuyProduct?: (productId: number) => void
}

export function MessageList({
  messages,
  onConfirm,
  onCancel,
  isProcessingConfirmation = false,
  onBuyProduct,
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

  // Empty state
  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-4 sm:p-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="text-center max-w-md space-y-4"
        >
          <motion.div
            animate={{ 
              scale: [1, 1.05, 1],
              rotate: [0, 5, -5, 0]
            }}
            transition={{ 
              duration: 3,
              repeat: Infinity,
              repeatType: "reverse"
            }}
            className="mx-auto w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center"
          >
            <Bot className="w-8 h-8 sm:w-10 sm:h-10 text-primary" />
          </motion.div>
          <div className="space-y-2">
            <h3 className="text-lg sm:text-xl font-semibold text-foreground">
              开始对话
            </h3>
            <p className="text-sm sm:text-base text-muted-foreground">
              我是您的AI智能助手，可以帮您搜索商品、查询订单、回答问题等
            </p>
          </div>
          <div className="flex flex-wrap gap-2 justify-center pt-2">
            {["搜索商品", "查询订单", "产品推荐"].map((suggestion, idx) => (
              <motion.div
                key={suggestion}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 + idx * 0.1 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-3 py-1.5 text-xs sm:text-sm rounded-full bg-muted hover:bg-muted/80 text-foreground border border-border/50 transition-colors cursor-pointer"
              >
                {suggestion}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    )
  }

  return (
    <ScrollArea className="flex-1">
      <div className="mx-auto max-w-6xl p-3 sm:p-4 lg:p-6">
        <div ref={scrollRef} className="space-y-4 sm:space-y-5">
          {messages.map((message, index) => (
            <MessageItem
              key={message.id}
              message={message}
              isLast={index === messages.length - 1}
              onConfirm={onConfirm}
              onCancel={onCancel}
              isProcessingConfirmation={isProcessingConfirmation}
              onBuyProduct={onBuyProduct}
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
  onBuyProduct?: (productId: number) => void
}

function MessageItem({
  message,
  isLast,
  onConfirm,
  onCancel,
  isProcessingConfirmation = false,
  onBuyProduct,
}: MessageItemProps) {
  const isUser = message.role === "user"

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
      className={cn(
        "flex gap-3 sm:gap-4 items-end group",
        isUser ? "justify-end" : "justify-start",
        "px-2 sm:px-0" // Add horizontal padding on mobile
      )}
    >
      {!isUser && (
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.2, delay: 0.1 }}
          className="flex h-8 w-8 sm:h-9 sm:w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary to-primary/80 text-primary-foreground shadow-sm ring-2 ring-primary/20"
        >
          <Bot className="h-4 w-4 sm:h-5 sm:w-5" />
        </motion.div>
      )}

      <div
        className={cn(
          "flex max-w-[85%] sm:max-w-[75%] lg:max-w-[85%] flex-col gap-2 sm:gap-2.5 rounded-2xl px-4 py-3 sm:px-5 sm:py-3.5",
          "shadow-sm transition-all duration-200",
          isUser
            ? "bg-primary text-primary-foreground rounded-br-md"
            : "bg-muted text-foreground rounded-bl-md border border-border/50"
        )}
      >
        {/* 产品列表 - 根据 responseType 渲染 */}
        {message.responseType === "product_list" && message.responseData?.products && (
          <ProductGrid products={message.responseData.products} onBuy={onBuyProduct} />
        )}

        {/* 订单列表 */}
        {message.responseType === "order_list" && message.responseData?.orders && (
          <OrderList orders={message.responseData.orders} />
        )}

        {/* 混合类型 - 同时包含产品和订单 */}
        {message.responseType === "mixed" && message.responseData && (
          <>
            {message.responseData.products && (
              <ProductGrid products={message.responseData.products} onBuy={onBuyProduct} />
            )}
            {message.responseData.orders && (
              <OrderList orders={message.responseData.orders} />
            )}
          </>
        )}

        {/* 产品对比 - 专门的对比UI */}
        {message.responseType === "product_comparison" && message.responseData && (
          <ProductComparison
            data={{
              comparison_aspects: message.responseData.comparison_aspects || [],
              comparison_details: message.responseData.comparison_details || {},
              scenario_analysis: message.responseData.scenario_analysis,
              recommendation: message.responseData.recommendation,
              products: message.responseData.products || [],
            }}
          />
        )}

        {/* 文本内容 - 对于product_comparison类型，文本是可选的；其他类型始终显示 */}
        {message.content && message.responseType !== "product_comparison" && (
          <div className={cn(
            "prose prose-sm dark:prose-invert max-w-none",
            "prose-headings:font-semibold prose-headings:text-foreground",
            "prose-p:text-foreground prose-p:leading-7",
            "prose-a:text-primary prose-a:no-underline hover:prose-a:underline",
            "prose-strong:text-foreground prose-strong:font-semibold",
            "prose-code:text-foreground",
            isUser && "prose-invert"
          )}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code: ({ inline, className, children, ...props }: any) => {
                  const match = /language-(\w+)/.exec(className || "")
                  return !inline && match ? (
                    <pre className={cn(
                      "bg-black/10 dark:bg-white/10 p-3 sm:p-4 rounded-lg overflow-x-auto",
                      "border border-border/50",
                      "text-xs sm:text-sm font-mono",
                      "my-2 sm:my-3",
                      isUser && "bg-white/20 dark:bg-white/10"
                    )}>
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </pre>
                  ) : (
                    <code className={cn(
                      "px-1.5 py-0.5 rounded text-xs sm:text-sm font-mono",
                      isUser 
                        ? "bg-white/20 text-primary-foreground" 
                        : "bg-background/80 text-foreground border border-border/50"
                    )} {...props}>
                      {children}
                    </code>
                  )
                },
                p: ({ children }) => (
                  <p className="mb-2 sm:mb-3 last:mb-0 text-sm sm:text-base leading-7 text-balance">
                    {children}
                  </p>
                ),
                ul: ({ children }) => (
                  <ul className="list-disc pl-4 sm:pl-6 mb-2 sm:mb-3 space-y-1 text-sm sm:text-base">
                    {children}
                  </ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal pl-4 sm:pl-6 mb-2 sm:mb-3 space-y-1 text-sm sm:text-base">
                    {children}
                  </ol>
                ),
                li: ({ children }) => (
                  <li className="leading-7">{children}</li>
                ),
                blockquote: ({ children }) => (
                  <blockquote className={cn(
                    "border-l-4 pl-4 sm:pl-5 my-3 sm:my-4 italic",
                    "bg-black/5 dark:bg-white/5 py-2 rounded-r-lg",
                    isUser 
                      ? "border-white/30 text-primary-foreground/90"
                      : "border-primary/50 text-muted-foreground"
                  )}>
                    {children}
                  </blockquote>
                ),
                h1: ({ children }) => (
                  <h1 className="text-xl sm:text-2xl font-semibold mt-4 mb-2">{children}</h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-lg sm:text-xl font-semibold mt-3 mb-2">{children}</h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-base sm:text-lg font-semibold mt-3 mb-1.5">{children}</h3>
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

        {/* 流式输入指示器 */}
        {message.isStreaming && isLast && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 text-muted-foreground text-xs mt-1"
          >
            <div className="flex gap-1">
              <motion.span
                animate={{ y: [0, -4, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
                className="w-1.5 h-1.5 rounded-full bg-current"
              />
              <motion.span
                animate={{ y: [0, -4, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                className="w-1.5 h-1.5 rounded-full bg-current"
              />
              <motion.span
                animate={{ y: [0, -4, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                className="w-1.5 h-1.5 rounded-full bg-current"
              />
            </div>
            <span className="italic">正在输入...</span>
          </motion.div>
        )}

        {/* 时间戳 */}
        <div
          className={cn(
            "text-[10px] sm:text-xs opacity-60 mt-1.5 sm:mt-2 -mb-1",
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
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.2, delay: 0.1 }}
          className="flex h-8 w-8 sm:h-9 sm:w-9 shrink-0 items-center justify-center rounded-full bg-secondary border-2 border-secondary-foreground/10 shadow-sm"
        >
          <User className="h-4 w-4 sm:h-5 sm:w-5" />
        </motion.div>
      )}
    </motion.div>
  )
}

