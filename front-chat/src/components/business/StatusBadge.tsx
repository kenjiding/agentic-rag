import { Badge } from "@/components/ui/badge"
import { CheckCircle2, Clock, Package, Truck, XCircle } from "lucide-react"

export type OrderStatus = "pending" | "paid" | "shipped" | "delivered" | "cancelled"

interface StatusBadgeProps {
  status: OrderStatus
  showIcon?: boolean
}

const statusConfig: Record<
  OrderStatus,
  { label: string; variant: "default" | "destructive" | "success" | "warning"; icon: typeof CheckCircle2 }
> = {
  pending: {
    label: "待支付",
    variant: "warning",
    icon: Clock,
  },
  paid: {
    label: "已支付",
    variant: "success",
    icon: CheckCircle2,
  },
  shipped: {
    label: "已发货",
    variant: "default",
    icon: Truck,
  },
  delivered: {
    label: "已收货",
    variant: "success",
    icon: Package,
  },
  cancelled: {
    label: "已取消",
    variant: "destructive",
    icon: XCircle,
  },
}

export function StatusBadge({ status, showIcon = true }: StatusBadgeProps) {
  const config = statusConfig[status]
  const Icon = config.icon

  return (
    <Badge variant={config.variant} className="flex items-center gap-1">
      {showIcon && <Icon className="w-3 h-3" />}
      {config.label}
    </Badge>
  )
}

