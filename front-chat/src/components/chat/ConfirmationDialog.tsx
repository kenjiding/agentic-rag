import { motion } from "framer-motion"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AlertCircle, CheckCircle, XCircle, ShoppingCart, Loader2 } from "lucide-react"
import { ConfirmationPending } from "@/types"
import { Separator } from "@/components/ui/separator"

interface ConfirmationDialogProps {
  confirmation: ConfirmationPending
  onConfirm: (confirmationId: string) => void
  onCancel: (confirmationId: string) => void
  isProcessing?: boolean
}

const actionIcons: Record<string, typeof AlertCircle> = {
  cancel_order: XCircle,
  create_order: ShoppingCart,
}

const actionLabels: Record<string, string> = {
  cancel_order: "取消订单",
  create_order: "创建订单",
}

export function ConfirmationDialog({
  confirmation,
  onConfirm,
  onCancel,
  isProcessing = false,
}: ConfirmationDialogProps) {
  const Icon = actionIcons[confirmation.action_type] || AlertCircle
  const actionLabel = actionLabels[confirmation.action_type] || confirmation.action_type

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95, y: 10 }}
      transition={{ duration: 0.2 }}
      className="my-4"
    >
      <Card className="border-amber-200 bg-amber-50/50 dark:border-amber-800 dark:bg-amber-950/20">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-full bg-amber-100 dark:bg-amber-900/30">
              <Icon className="h-5 w-5 text-amber-600 dark:text-amber-400" />
            </div>
            <CardTitle className="text-lg text-amber-800 dark:text-amber-200">
              确认{actionLabel}
            </CardTitle>
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          {/* 主消息 */}
          <p className="text-sm whitespace-pre-wrap text-gray-700 dark:text-gray-300">
            {confirmation.display_message}
          </p>

          {/* 订单详情（如果有） */}
          {confirmation.display_data?.items && confirmation.display_data.items.length > 0 && (
            <>
              <Separator className="bg-amber-200 dark:bg-amber-800" />
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                  商品明细
                </h4>
                {confirmation.display_data.items.map((item, index) => (
                  <div
                    key={index}
                    className="flex justify-between text-sm py-1"
                  >
                    <span className="text-gray-600 dark:text-gray-400">
                      {item.name} x {item.quantity}
                    </span>
                    <span className="font-medium text-gray-800 dark:text-gray-200">
                      ¥{item.subtotal.toFixed(2)}
                    </span>
                  </div>
                ))}
                {confirmation.display_data.total_amount && (
                  <div className="flex justify-between font-semibold pt-2 border-t border-amber-200 dark:border-amber-800">
                    <span className="text-gray-700 dark:text-gray-300">总金额</span>
                    <span className="text-amber-700 dark:text-amber-400">
                      ¥{confirmation.display_data.total_amount.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
            </>
          )}
        </CardContent>

        <CardFooter className="gap-3 pt-3">
          <Button
            variant="outline"
            className="flex-1 border-gray-300 hover:bg-gray-100 dark:border-gray-600 dark:hover:bg-gray-800"
            onClick={() => onCancel(confirmation.confirmation_id)}
            disabled={isProcessing}
          >
            <XCircle className="mr-2 h-4 w-4" />
            取消
          </Button>
          <Button
            className="flex-1 bg-amber-600 hover:bg-amber-700 text-white"
            onClick={() => onConfirm(confirmation.confirmation_id)}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                处理中...
              </>
            ) : (
              <>
                <CheckCircle className="mr-2 h-4 w-4" />
                确认
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
    </motion.div>
  )
}
