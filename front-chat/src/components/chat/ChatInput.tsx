import { useState, KeyboardEvent, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Send, StopCircle } from "lucide-react"
import { cn } from "@/lib/utils"

interface ChatInputProps {
  onSend: (message: string) => void
  onStop?: () => void
  isLoading?: boolean
  disabled?: boolean
}

const MIN_TEXTAREA_HEIGHT = 44 // Minimum height in pixels (matches button height)
const MAX_TEXTAREA_HEIGHT = 200 // Maximum height in pixels

export function ChatInput({ onSend, onStop, isLoading, disabled }: ChatInputProps) {
  const [input, setInput] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return

    // Reset height to calculate scrollHeight
    textarea.style.height = "auto"
    const scrollHeight = textarea.scrollHeight
    
    // Set height within min/max bounds
    const newHeight = Math.min(
      Math.max(scrollHeight, MIN_TEXTAREA_HEIGHT),
      MAX_TEXTAREA_HEIGHT
    )
    textarea.style.height = `${newHeight}px`
  }, [input])

  const handleSend = () => {
    if (input.trim() && !isLoading && !disabled) {
      onSend(input.trim())
      setInput("")
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto"
      }
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends, Shift+Enter creates new line
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const isInputEmpty = !input.trim()

  return (
    <div className="border-t border-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto max-w-4xl px-3 sm:px-4 py-3 sm:py-4">
        <div className="relative flex items-end gap-2">
          {/* Input area */}
          <div className="relative flex-1 min-w-0">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入消息..."
              disabled={disabled || isLoading}
              rows={1}
              className={cn(
                "flex w-full resize-none rounded-xl border border-input bg-background px-4 py-3 text-sm",
                "placeholder:text-muted-foreground",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0",
                "disabled:cursor-not-allowed disabled:opacity-50",
                "transition-all duration-200",
                "min-h-[44px] max-h-[200px] overflow-y-auto",
                // Custom scrollbar styling
                "scrollbar-thin scrollbar-thumb-border scrollbar-track-transparent"
              )}
              style={{
                height: `${MIN_TEXTAREA_HEIGHT}px`,
              }}
            />
          </div>

          {/* Send/Stop button */}
          {isLoading && onStop ? (
            <Button
              variant="outline"
              size="icon"
              onClick={onStop}
              className="shrink-0 h-11 w-11 rounded-xl border-destructive/20 hover:bg-destructive/10 hover:border-destructive/30 transition-all"
              title="停止生成"
            >
              <StopCircle className="h-5 w-5 text-destructive" />
            </Button>
          ) : (
            <Button
              onClick={handleSend}
              disabled={disabled || isLoading || isInputEmpty}
              size="icon"
              className={cn(
                "shrink-0 h-11 w-11 rounded-xl transition-all",
                "disabled:opacity-50 disabled:cursor-not-allowed",
                "hover:scale-105 active:scale-95",
                "shadow-sm hover:shadow-md"
              )}
              title="发送消息"
            >
              <Send className="h-5 w-5" />
            </Button>
          )}
        </div>
        
        {/* Helper text */}
        <p className="text-xs text-muted-foreground mt-2 text-center hidden sm:block">
          按 <kbd className="px-1.5 py-0.5 text-xs font-semibold text-foreground bg-muted border border-border rounded">Enter</kbd> 发送，<kbd className="px-1.5 py-0.5 text-xs font-semibold text-foreground bg-muted border border-border rounded">Shift + Enter</kbd> 换行
        </p>
      </div>
    </div>
  )
}

