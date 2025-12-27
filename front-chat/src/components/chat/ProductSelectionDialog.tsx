import { motion } from "framer-motion"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { XCircle, ShoppingBag, Star, Package, TrendingUp, Loader2 } from "lucide-react"
import { PendingSelection, Product } from "@/types"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"

interface ProductSelectionDialogProps {
  selection: PendingSelection
  onSelect: (selectionId: string, productId: string) => void
  onCancel: (selectionId: string) => void
  isProcessing?: boolean
}

export function ProductSelectionDialog({
  selection,
  onSelect,
  onCancel,
  isProcessing = false,
}: ProductSelectionDialogProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95, y: 10 }}
      transition={{ duration: 0.2 }}
      className="my-4"
    >
      <Card className="border-blue-200 bg-blue-50/50 dark:border-blue-800 dark:bg-blue-950/20">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-full bg-blue-100 dark:bg-blue-900/30">
              <ShoppingBag className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            </div>
            <CardTitle className="text-lg text-blue-800 dark:text-blue-200">
              {selection.display_message}
            </CardTitle>
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          <Separator className="bg-blue-200 dark:bg-blue-800" />

          {/* 产品网格 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {selection.options.map((product) => (
              <SelectableProductCard
                key={product.id}
                product={product}
                onSelect={() => onSelect(selection.selection_id, String(product.id))}
                disabled={isProcessing}
              />
            ))}
          </div>
        </CardContent>

        <CardFooter className="pt-3">
          <Button
            variant="outline"
            className="w-full border-gray-300 hover:bg-gray-100 dark:border-gray-600 dark:hover:bg-gray-800"
            onClick={() => onCancel(selection.selection_id)}
            disabled={isProcessing}
          >
            <XCircle className="mr-2 h-4 w-4" />
            取消
          </Button>
        </CardFooter>
      </Card>
    </motion.div>
  )
}

interface SelectableProductCardProps {
  product: Product
  onSelect: () => void
  disabled?: boolean
}

function SelectableProductCard({ product, onSelect, disabled }: SelectableProductCardProps) {
  const isInStock = product.stock > 0
  const priceDisplay = product.price ? `¥${product.price.toFixed(2)}` : "价格面议"

  return (
    <motion.div
      whileHover={!disabled ? { scale: 1.02 } : {}}
      whileTap={!disabled ? { scale: 0.98 } : {}}
      className="h-full"
    >
      <Card
        className={cn(
          "h-full flex flex-col cursor-pointer transition-all duration-200",
          "hover:border-blue-500 hover:shadow-lg",
          disabled && "opacity-60 cursor-not-allowed"
        )}
        onClick={() => !disabled && onSelect()}
      >
        <CardHeader className="pb-2 pt-3 px-3">
          <div className="flex items-start justify-between gap-2">
            <h4 className="text-sm font-semibold leading-tight line-clamp-2 flex-1">
              {product.name}
            </h4>
            {product.special && (
              <Badge variant="destructive" className="shrink-0 text-[10px] px-1.5 py-0.5">
                <TrendingUp className="w-2.5 h-2.5 mr-0.5" />
                特价
              </Badge>
            )}
          </div>
        </CardHeader>

        <CardContent className="flex-1 space-y-2 px-3 pb-2">
          {/* 品牌和分类 */}
          {(product.brand || product.main_category) && (
            <div className="flex flex-wrap gap-1">
              {product.brand && (
                <Badge variant="secondary" className="text-[10px] px-1.5 py-0.5">
                  {product.brand}
                </Badge>
              )}
              {product.main_category && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0.5">
                  {product.main_category}
                </Badge>
              )}
            </div>
          )}

          {/* 价格和评分 */}
          <div className="flex items-center justify-between pt-1 border-t border-border/50">
            <div className="flex flex-col">
              <span className="text-[10px] text-muted-foreground">价格</span>
              <div className="text-lg font-bold text-primary">
                {priceDisplay}
              </div>
            </div>
            {product.rating > 0 && (
              <div className="flex items-center gap-1 bg-yellow-50 dark:bg-yellow-950/20 px-1.5 py-0.5 rounded-full">
                <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                <span className="font-semibold text-xs text-yellow-600 dark:text-yellow-400">
                  {product.rating.toFixed(1)}
                </span>
              </div>
            )}
          </div>

          {/* 库存 */}
          <div className="flex items-center gap-1">
            <Package
              className={cn(
                "w-3 h-3",
                isInStock ? "text-green-500" : "text-red-500"
              )}
            />
            <span
              className={cn(
                "text-xs font-medium",
                isInStock ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
              )}
            >
              {isInStock ? `库存: ${product.stock}件` : "暂时缺货"}
            </span>
          </div>

          {/* 描述 */}
          {product.description && (
            <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
              {product.description}
            </p>
          )}
        </CardContent>

        <CardFooter className="pt-2 px-3 pb-3 bg-blue-50/50 dark:bg-blue-950/30 border-t">
          <Button
            variant="default"
            size="sm"
            className="w-full text-xs h-7 bg-blue-600 hover:bg-blue-700"
            disabled={!isInStock || disabled}
          >
            {disabled ? (
              <>
                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                处理中...
              </>
            ) : (
              <>选择此商品</>
            )}
          </Button>
        </CardFooter>
      </Card>
    </motion.div>
  )
}
