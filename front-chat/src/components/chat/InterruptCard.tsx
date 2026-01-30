import { useMemo, useState } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { AlertCircle, Loader2 } from "lucide-react"
import type { InterruptPayload, InterruptSelectionOption, SelectionMode } from "@/types"
import { cn } from "@/lib/utils"

interface InterruptCardProps {
  interrupt: InterruptPayload
  onSubmit: (resumeData: Record<string, any>) => void
  isProcessing?: boolean
}

function buildBaseResumeData(interrupt: InterruptPayload) {
  return {
    action_type: interrupt.action_type,
    action_data: interrupt.action_data ?? {},
    metadata: interrupt.metadata ?? {},
  }
}

function getSubmitLabel(interrupt: InterruptPayload, fallback: string) {
  const dd: any = interrupt.display_data
  return (dd && typeof dd.submit_label === "string" && dd.submit_label) || fallback
}

export function InterruptCard({ interrupt, onSubmit, isProcessing = false }: InterruptCardProps) {
  const [error, setError] = useState<string | null>(null)

  const type = interrupt.interrupt_type

  const title = useMemo(() => {
    if (type === "input") return "需要补充信息"
    if (type === "selection") return "需要选择"
    return "需要操作"
  }, [type])

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98, y: 6 }}
      transition={{ duration: 0.2 }}
      className="my-3"
    >
      <Card className="border-blue-200 bg-blue-50/50 dark:border-blue-900/60 dark:bg-blue-950/20">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-full bg-blue-100 dark:bg-blue-900/30">
              <AlertCircle className="h-5 w-5 text-blue-700 dark:text-blue-300" />
            </div>
            <CardTitle className="text-base sm:text-lg text-blue-900 dark:text-blue-100">
              {title}
            </CardTitle>
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          <p className="text-sm whitespace-pre-wrap text-gray-700 dark:text-gray-300">
            {interrupt.display_message || interrupt.content}
          </p>

          {type === "input" && (
            <InputBlock
              interrupt={interrupt}
              onSubmit={(resumeData) => {
                setError(null)
                onSubmit(resumeData)
              }}
              onError={setError}
              isProcessing={isProcessing}
            />
          )}

          {type === "selection" && (
            <SelectionBlock
              interrupt={interrupt}
              onSubmit={(resumeData) => {
                setError(null)
                onSubmit(resumeData)
              }}
              onError={setError}
              isProcessing={isProcessing}
            />
          )}

          {error && (
            <div className="text-xs text-red-600 dark:text-red-400">
              {error}
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}

function InputBlock({
  interrupt,
  onSubmit,
  onError,
  isProcessing,
}: {
  interrupt: InterruptPayload
  onSubmit: (resumeData: Record<string, any>) => void
  onError: (msg: string | null) => void
  isProcessing: boolean
}) {
  const dd: any = interrupt.display_data
  const spec = dd?.input

  const [value, setValue] = useState<string>("")

  if (!spec) {
    return (
      <div className="text-xs text-muted-foreground">
        缺少 input 规格（display_data.input），无法渲染。
      </div>
    )
  }

  const submitLabel = getSubmitLabel(interrupt, "提交并继续")

  const validate = () => {
    const v = value ?? ""
    if (spec.required && !v.trim()) return "请输入内容后再提交。"
    if (typeof spec.min_length === "number" && v.length < spec.min_length) {
      return `输入长度不能少于 ${spec.min_length}。`
    }
    if (typeof spec.max_length === "number" && v.length > spec.max_length) {
      return `输入长度不能超过 ${spec.max_length}。`
    }
    if (typeof spec.pattern === "string" && spec.pattern) {
      try {
        const re = new RegExp(spec.pattern)
        if (!re.test(v)) return "输入格式不符合要求。"
      } catch {
        // ignore invalid pattern from backend; do not block user
      }
    }
    return null
  }

  const handleSubmit = () => {
    const err = validate()
    if (err) {
      onError(err)
      return
    }
    const base = buildBaseResumeData(interrupt)
    onSubmit({ ...base, input_value: value })
  }

  return (
    <div className="space-y-2">
      <div className="text-xs font-medium text-gray-700 dark:text-gray-300">
        {spec.label}
      </div>
      {spec.multiline ? (
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={spec.placeholder}
          disabled={isProcessing}
          rows={3}
          className={cn(
            "w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            "disabled:cursor-not-allowed disabled:opacity-50"
          )}
        />
      ) : (
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={spec.placeholder}
          disabled={isProcessing}
        />
      )}

      <Button
        onClick={handleSubmit}
        disabled={isProcessing}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white"
      >
        {isProcessing ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            处理中...
          </>
        ) : (
          submitLabel
        )}
      </Button>
    </div>
  )
}

function SelectionBlock({
  interrupt,
  onSubmit,
  onError,
  isProcessing,
}: {
  interrupt: InterruptPayload
  onSubmit: (resumeData: Record<string, any>) => void
  onError: (msg: string | null) => void
  isProcessing: boolean
}) {
  const dd: any = interrupt.display_data
  const selection = dd?.selection

  const [selectedSingle, setSelectedSingle] = useState<string>("")
  const [selectedMulti, setSelectedMulti] = useState<Set<string>>(() => new Set())

  if (!selection) {
    return (
      <div className="text-xs text-muted-foreground">
        缺少 selection 规格（display_data.selection），无法渲染。
      </div>
    )
  }

  const mode: SelectionMode = selection.mode
  const options: InterruptSelectionOption[] = selection.options || []
  const minSelected: number | undefined = selection.min_selected
  const maxSelected: number | undefined = selection.max_selected

  const submitLabel = getSubmitLabel(interrupt, "确认选择并继续")

  const selectedCount = mode === "single" ? (selectedSingle ? 1 : 0) : selectedMulti.size

  const validate = () => {
    if (mode === "single") {
      if (!selectedSingle) return "请选择一个选项后再提交。"
      return null
    }
    // multi
    if (typeof minSelected === "number" && selectedCount < minSelected) {
      return `至少需要选择 ${minSelected} 个选项。`
    }
    if (typeof maxSelected === "number" && selectedCount > maxSelected) {
      return `最多只能选择 ${maxSelected} 个选项。`
    }
    if (selectedCount === 0) return "请选择至少一个选项后再提交。"
    return null
  }

  const handleSubmit = () => {
    const err = validate()
    if (err) {
      onError(err)
      return
    }
    const base = buildBaseResumeData(interrupt)
    const selected =
      mode === "single" ? selectedSingle : Array.from(selectedMulti.values())
    onSubmit({ ...base, selected })
  }

  return (
    <div className="space-y-2">
      <div className="space-y-2">
        {options.map((opt) => (
          <label
            key={opt.option_id}
            className={cn(
              "flex items-start gap-2 rounded-md border border-border/60 bg-background/60 px-3 py-2",
              "hover:bg-background/80 transition-colors",
              isProcessing && "opacity-60 cursor-not-allowed"
            )}
          >
            {mode === "single" ? (
              <input
                type="radio"
                name={`sel-${selection.selection_id}`}
                value={opt.option_id}
                checked={selectedSingle === opt.option_id}
                onChange={() => setSelectedSingle(opt.option_id)}
                disabled={isProcessing}
                className="mt-1"
              />
            ) : (
              <input
                type="checkbox"
                value={opt.option_id}
                checked={selectedMulti.has(opt.option_id)}
                onChange={(e) => {
                  setSelectedMulti((prev) => {
                    const next = new Set(prev)
                    if (e.target.checked) next.add(opt.option_id)
                    else next.delete(opt.option_id)
                    return next
                  })
                }}
                disabled={isProcessing}
                className="mt-1"
              />
            )}
            <div className="min-w-0">
              <div className="text-sm font-medium text-foreground truncate">
                {opt.label}
              </div>
              {opt.description && (
                <div className="text-xs text-muted-foreground whitespace-pre-wrap">
                  {opt.description}
                </div>
              )}
            </div>
          </label>
        ))}
      </div>

      <Button
        onClick={handleSubmit}
        disabled={isProcessing}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white"
      >
        {isProcessing ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            处理中...
          </>
        ) : (
          submitLabel
        )}
      </Button>
    </div>
  )
}

