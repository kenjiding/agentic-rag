import { ChatLayout } from "@/components/chat/ChatLayout"
import { useStreamingChat } from "@/hooks/useStreamingChat"

function App() {
  const { messages, isLoading, error, sendMessage, stop, clearMessages } =
    useStreamingChat()

  return (
    <ChatLayout
      messages={messages}
      onSend={sendMessage}
      onStop={stop}
      onClear={clearMessages}
      isLoading={isLoading}
      error={error}
    />
  )
}

export default App

