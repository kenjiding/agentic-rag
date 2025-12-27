import { ChatLayout } from "@/components/chat/ChatLayout"
import { useStreamingChat } from "@/hooks/useStreamingChat"

function App() {
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
    selectProduct,
    cancelSelection,
    isProcessingSelection,
  } = useStreamingChat()

  return (
    <ChatLayout
      messages={messages}
      onSend={sendMessage}
      onStop={stop}
      onClear={clearMessages}
      isLoading={isLoading}
      error={error}
      onConfirm={confirmAction}
      onCancel={cancelConfirmation}
      isProcessingConfirmation={isProcessingConfirmation}
      onSelectProduct={selectProduct}
      onCancelSelection={cancelSelection}
      isProcessingSelection={isProcessingSelection}
    />
  )
}

export default App

