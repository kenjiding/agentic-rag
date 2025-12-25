import { Product } from "@/types"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ShoppingCart, Star, Package, Eye, TrendingUp, Award } from "lucide-react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface ProductCardProps {
  product: Product
  onViewDetails?: (productId: number) => void
}

export function ProductCard({ product, onViewDetails }: ProductCardProps) {
  const isInStock = product.stock > 0
  const priceDisplay = product.price ? `¥${product.price.toFixed(2)}` : "价格面议"
  const stockPercentage = product.stock > 0 ? Math.min((product.stock / 100) * 100, 100) : 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      whileHover={{ y: -6, transition: { duration: 0.2 } }}
      className="h-full"
    >
      <Card className="h-full flex flex-col group overflow-hidden border-2 hover:border-primary/50 transition-all duration-300 bg-gradient-to-br from-card to-card/50">
        {/* 头部区域 - 带渐变背景 */}
        <CardHeader className="pb-4 relative overflow-hidden bg-gradient-to-br from-primary/5 via-primary/3 to-transparent">
          <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
          
          <div className="relative z-10">
            <div className="flex items-start justify-between gap-2 mb-2">
              <h3 className="text-lg font-bold leading-tight line-clamp-2 flex-1 group-hover:text-primary transition-colors">
                {product.name}
              </h3>
              {product.special && (
                <Badge 
                  variant="destructive" 
                  className="shrink-0 ml-2 animate-pulse shadow-lg"
                >
                  <TrendingUp className="w-3 h-3 mr-1" />
                  特价
                </Badge>
              )}
            </div>
            
            {product.model_number && (
              <div className="flex items-center gap-1.5 mt-2">
                <Award className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-xs text-muted-foreground font-mono">
                  {product.model_number}
                </span>
              </div>
            )}
          </div>
        </CardHeader>
        
        <CardContent className="flex-1 space-y-4 px-6">
          {/* 品牌和分类标签 */}
          {(product.brand || product.main_category) && (
            <div className="flex flex-wrap gap-2">
              {product.brand && (
                <Badge variant="secondary" className="text-xs px-2.5 py-1">
                  {product.brand}
                </Badge>
              )}
              {product.main_category && (
                <Badge variant="outline" className="text-xs px-2.5 py-1">
                  {product.main_category}
                </Badge>
              )}
              {product.sub_category && (
                <Badge variant="outline" className="text-xs px-2.5 py-1 opacity-70">
                  {product.sub_category}
                </Badge>
              )}
            </div>
          )}

          {/* 价格和评分区域 */}
          <div className="flex items-center justify-between pt-2 pb-1 border-t border-b border-border/50">
            <div className="flex flex-col">
              <span className="text-xs text-muted-foreground mb-0.5">价格</span>
              <div className="text-3xl font-bold text-primary tracking-tight">
                {priceDisplay}
              </div>
            </div>
            {product.rating > 0 && (
              <div className="flex flex-col items-end">
                <span className="text-xs text-muted-foreground mb-1">评分</span>
                <div className="flex items-center gap-1.5 bg-yellow-50 dark:bg-yellow-950/20 px-3 py-1.5 rounded-full">
                  <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                  <span className="font-bold text-yellow-600 dark:text-yellow-400">
                    {product.rating.toFixed(1)}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* 库存状态 - 带进度条 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Package className={cn(
                  "w-4 h-4",
                  isInStock ? 'text-green-500' : 'text-red-500'
                )} />
                <span className={cn(
                  "text-sm font-medium",
                  isInStock ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                )}>
                  {isInStock ? `库存充足` : "暂时缺货"}
                </span>
              </div>
              {isInStock && (
                <span className="text-xs text-muted-foreground">
                  {product.stock} 件
                </span>
              )}
            </div>
            {isInStock && (
              <div className="w-full h-1.5 bg-secondary rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${stockPercentage}%` }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                  className={cn(
                    "h-full rounded-full",
                    stockPercentage > 50 ? "bg-green-500" : 
                    stockPercentage > 20 ? "bg-yellow-500" : "bg-orange-500"
                  )}
                />
              </div>
            )}
          </div>

          {/* 描述 */}
          {product.description && (
            <div className="pt-2">
              <p className="text-sm text-muted-foreground line-clamp-3 leading-relaxed">
                {product.description}
              </p>
            </div>
          )}
        </CardContent>

        <CardFooter className="pt-4 px-6 pb-6 gap-3 border-t bg-muted/30">
          <Button
            variant="outline"
            className="flex-1 group/btn"
            onClick={() => onViewDetails?.(product.id)}
          >
            <Eye className="w-4 h-4 mr-2 group-hover/btn:scale-110 transition-transform" />
            查看详情
          </Button>
          <Button
            variant="default"
            className="flex-1 group/btn shadow-lg hover:shadow-xl transition-all"
            disabled={!isInStock}
          >
            <ShoppingCart className="w-4 h-4 mr-2 group-hover/btn:scale-110 transition-transform" />
            {isInStock ? "加入购物车" : "缺货"}
          </Button>
        </CardFooter>
      </Card>
    </motion.div>
  )
}

interface ProductGridProps {
  products: Product[]
  onViewDetails?: (productId: number) => void
}

export function ProductGrid({ products, onViewDetails }: ProductGridProps) {
  if (!products || products.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <Package className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>暂无商品信息</p>
      </div>
    )
  }

  return (
    <div className="my-6">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-primary" />
          找到 {products.length} 个商品
        </h3>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {products.map((product, index) => (
          <ProductCard
            key={product.id}
            product={product}
            onViewDetails={onViewDetails}
          />
        ))}
      </div>
    </div>
  )
}

