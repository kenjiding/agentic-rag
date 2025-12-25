import { motion, AnimatePresence } from "framer-motion"
import { CheckCircle2, Loader2, AlertCircle } from "lucide-react"
import { cn } from "@/lib/utils"
import { ExecutionStepDetail } from "@/types"

interface ExecutionStepsProps {
  steps: string[]
  stepDetails?: ExecutionStepDetail[]
  isActive: boolean
}

export function ExecutionSteps({ steps, stepDetails, isActive }: ExecutionStepsProps) {
  // 如果没有步骤，不显示
  if (!steps || steps.length === 0) {
    return null
  }

  // 使用 stepDetails 如果有，否则从 steps 创建基本结构
  const details: ExecutionStepDetail[] = stepDetails || steps.map((step, index) => ({
    name: step,
    status: (index === steps.length - 1 && isActive) ? "running" : "completed" as const
  }))

  return (
    <div className="mt-3 space-y-2 border-t pt-3">
      <div className="text-xs font-semibold text-muted-foreground mb-2 flex items-center gap-2">
        <span>AI Agent 执行流程</span>
        {isActive && (
          <span className="flex items-center gap-1 text-primary">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
            </span>
            <span className="text-[10px]">执行中</span>
          </span>
        )}
      </div>
      <div className="space-y-1.5">
        <AnimatePresence>
          {details.map((step, index) => {
            const isRunning = step.status === "running"
            const isCompleted = step.status === "completed"
            const isError = step.status === "error"

            return (
              <motion.div
                key={`${step.name}-${index}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2, delay: Math.min(index * 0.05, 0.3) }}
                className={cn(
                  "flex items-start gap-2 text-xs p-1.5 rounded-md transition-colors",
                  isRunning && "bg-primary/5",
                  isError && "bg-destructive/5"
                )}
              >
                {/* 状态图标 */}
                <div className="shrink-0 mt-0.5">
                  {isCompleted ? (
                    <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                  ) : isRunning ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                  ) : (
                    <AlertCircle className="h-3.5 w-3.5 text-destructive" />
                  )}
                </div>

                {/* 步骤内容 */}
                <div className="flex-1 min-w-0">
                  <div className={cn(
                    "font-medium",
                    isRunning ? "text-primary" : isError ? "text-destructive" : "text-foreground"
                  )}>
                    {step.name}
                  </div>
                  {step.detail && (
                    <div className={cn(
                      "text-[10px] mt-0.5 leading-tight",
                      isRunning ? "text-primary/70" : "text-muted-foreground"
                    )}>
                      {step.detail}
                    </div>
                  )}
                </div>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>

      {/* 执行中的提示 */}
      {isActive && details.length > 0 && details[details.length - 1].status === "running" && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-[10px] text-muted-foreground italic pl-6"
        >
          正在处理，请稍候...
        </motion.div>
      )}
    </div>
  )
}

