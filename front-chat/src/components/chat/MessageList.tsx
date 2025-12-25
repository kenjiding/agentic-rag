import { ChatMessage } from "@/types"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ProductGrid } from "@/components/business/ProductCard"
import { OrderList } from "@/components/business/OrderTracker"
import { ExecutionSteps } from "./ExecutionSteps"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { motion } from "framer-motion"
import { User, Bot } from "lucide-react"
import { cn } from "@/lib/utils"

interface MessageListProps {
  messages: ChatMessage[]
}

export function MessageList({ messages }: MessageListProps) {
  return (
    <ScrollArea className="flex-1 p-4">
      <div className="space-y-4">
        {messages.map((message, index) => (
          <MessageItem
            key={message.id}
            message={message}
            isLast={index === messages.length - 1}
          />
        ))}
      </div>
    </ScrollArea>
  )
}

interface MessageItemProps {
  message: ChatMessage
  isLast: boolean
}

function MessageItem({ message, isLast }: MessageItemProps) {
  const isUser = message.role === "user"

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={cn(
        "flex gap-3",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Bot className="h-4 w-4" />
        </div>
      )}

      <div
        className={cn(
          "flex max-w-[80%] flex-col gap-2 rounded-lg px-4 py-3",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted"
        )}
      >
        {/* 产品列表 - 根据 responseType 渲染 */}
        {message.responseType === "product_list" && message.responseData?.products && (
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
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code: ({ inline, className, children, ...props }: any) => {
                  const match = /language-(\w+)/.exec(className || "")
                  return !inline && match ? (
                    <pre className="bg-muted p-3 rounded-md overflow-x-auto">
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </pre>
                  ) : (
                    <code className="bg-muted px-1.5 py-0.5 rounded text-sm" {...props}>
                      {children}
                    </code>
                  )
                },
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="list-disc pl-4 mb-2">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-4 mb-2">{children}</ol>,
                li: ({ children }) => <li className="mb-1">{children}</li>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-primary pl-4 italic my-2">
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

        {/* 流式输入指示器 */}
        {message.isStreaming && isLast && (
          <div className="flex items-center gap-1 text-muted-foreground text-xs mt-2">
            <span className="animate-pulse">●</span>
            <span>正在输入...</span>
          </div>
        )}

        {/* 时间戳 */}
        <div
          className={cn(
            "text-xs opacity-70",
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
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary">
          <User className="h-4 w-4" />
        </div>
      )}
    </motion.div>
  )
}

