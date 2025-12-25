import { useState, KeyboardEvent } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send, StopCircle } from "lucide-react"
import { cn } from "@/lib/utils"

interface ChatInputProps {
  onSend: (message: string) => void
  onStop?: () => void
  isLoading?: boolean
  disabled?: boolean
}

export function ChatInput({ onSend, onStop, isLoading, disabled }: ChatInputProps) {
  const [input, setInput] = useState("")

  const handleSend = () => {
    if (input.trim() && !isLoading && !disabled) {
      onSend(input.trim())
      setInput("")
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t bg-background p-4">
      <div className="flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入消息..."
          disabled={disabled || isLoading}
          className="flex-1"
        />
        {isLoading && onStop ? (
          <Button
            variant="outline"
            size="icon"
            onClick={onStop}
            className="shrink-0"
          >
            <StopCircle className="h-4 w-4" />
          </Button>
        ) : (
          <Button
            onClick={handleSend}
            disabled={disabled || isLoading || !input.trim()}
            size="icon"
            className="shrink-0"
          >
            <Send className="h-4 w-4" />
          </Button>
        )}
      </div>
      <p className="text-xs text-muted-foreground mt-2 text-center">
        按 Enter 发送，Shift + Enter 换行
      </p>
    </div>
  )
}

