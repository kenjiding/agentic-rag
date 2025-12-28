import { Order } from "@/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { CheckCircle2, Clock, Truck, XCircle, Package } from "lucide-react"
import { motion } from "framer-motion"

interface OrderTrackerProps {
  order: Order
}

const statusConfig = {
  pending: {
    label: "待支付",
    icon: Clock,
    variant: "warning" as const,
    color: "text-yellow-600",
  },
  paid: {
    label: "已支付",
    icon: CheckCircle2,
    variant: "success" as const,
    color: "text-green-600",
  },
  shipped: {
    label: "已发货",
    icon: Truck,
    variant: "default" as const,
    color: "text-orange-600",
  },
  delivered: {
    label: "已收货",
    icon: CheckCircle2,
    variant: "success" as const,
    color: "text-green-600",
  },
  cancelled: {
    label: "已取消",
    icon: XCircle,
    variant: "destructive" as const,
    color: "text-red-600",
  },
}

export function OrderTracker({ order }: OrderTrackerProps) {
  const config = statusConfig[order.status] || statusConfig.pending
  const StatusIcon = config.icon

  // 计算进度步骤
  const steps = [
    { key: "pending", label: "待支付", completed: order.status !== "pending" },
    { key: "paid", label: "已支付", completed: ["paid", "shipped", "delivered"].includes(order.status) },
    { key: "shipped", label: "已发货", completed: ["shipped", "delivered"].includes(order.status) },
    { key: "delivered", label: "已收货", completed: order.status === "delivered" },
  ]

  const currentStepIndex = steps.findIndex(
    (step) => step.key === order.status
  )

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">订单 {order.order_number}</CardTitle>
            <Badge variant={config.variant} className="flex items-center gap-1">
              <StatusIcon className="w-3 h-3" />
              {config.label}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            创建时间: {order.created_at}
          </p>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* 时间轴进度条 */}
          <div className="relative">
            {steps.map((step, index) => {
              const isCompleted = step.completed
              const isCurrent = index === currentStepIndex && !isCompleted
              const isPast = index < currentStepIndex

              return (
                <div key={step.key} className="relative flex items-start pb-6 last:pb-0">
                  {/* 连接线 */}
                  {index < steps.length - 1 && (
                    <div
                      className={`absolute left-4 top-8 w-0.5 h-full ${
                        isPast || isCompleted
                          ? "bg-primary"
                          : "bg-border"
                      }`}
                    />
                  )}

                  {/* 图标 */}
                  <div
                    className={`relative z-10 flex h-8 w-8 items-center justify-center rounded-full border-2 ${
                      isCompleted
                        ? "bg-primary border-primary text-primary-foreground"
                        : isCurrent
                        ? "bg-primary/10 border-primary text-primary"
                        : "bg-background border-border text-muted-foreground"
                    }`}
                  >
                    {isCompleted ? (
                      <CheckCircle2 className="w-4 h-4" />
                    ) : (
                      <div className="h-2 w-2 rounded-full bg-current" />
                    )}
                  </div>

                  {/* 标签 */}
                  <div className="ml-4 flex-1 pt-1">
                    <p
                      className={`text-sm font-medium ${
                        isCompleted || isCurrent
                          ? "text-foreground"
                          : "text-muted-foreground"
                      }`}
                    >
                      {step.label}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>

          <Separator />

          {/* 订单商品 */}
          <div className="space-y-2">
            <h4 className="text-sm font-semibold">商品明细</h4>
            {order.items.map((item, index) => (
              <div
                key={index}
                className="flex items-center gap-3 py-2 border-b last:border-0"
              >
                {/* 产品图片 - 小而美的设计 */}
                {item.product_images && item.product_images.length > 0 ? (
                  <div className="relative w-12 h-12 shrink-0 rounded-md overflow-hidden bg-gradient-to-br from-muted/40 to-muted/20 border border-border/50 shadow-sm">
                    <img
                      src={item.product_images[0]}
                      alt={item.product_name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.currentTarget.style.display = 'none'
                      }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/5 to-transparent pointer-events-none" />
                  </div>
                ) : (
                  <div className="w-12 h-12 shrink-0 rounded-md bg-gradient-to-br from-muted/40 to-muted/20 border border-border/50 flex items-center justify-center shadow-sm">
                    <Package className="w-5 h-5 text-muted-foreground/50" />
                  </div>
                )}
                
                {/* 产品信息 */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">
                    {item.product_name}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    数量: {item.quantity}
                  </p>
                </div>
                
                {/* 价格 */}
                <div className="text-right shrink-0">
                  <span className="text-sm font-semibold text-foreground">¥{item.subtotal.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>

          <Separator />

          {/* 总金额 */}
          <div className="flex justify-between items-center">
            <span className="text-sm font-semibold">总金额</span>
            <span className="text-xl font-bold text-primary">
              ¥{order.total_amount.toFixed(2)}
            </span>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}

interface OrderListProps {
  orders: Order[]
}

export function OrderList({ orders }: OrderListProps) {
  return (
    <div className="space-y-4 my-4">
      {orders.map((order) => (
        <OrderTracker key={order.id} order={order} />
      ))}
    </div>
  )
}

