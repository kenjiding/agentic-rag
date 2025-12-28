import { Product } from "@/types"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ShoppingCart, Star, Package, Eye, TrendingUp, Award, ChevronLeft, ChevronRight } from "lucide-react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"
import { useState, useEffect } from "react"

interface ProductCardProps {
  product: Product
  onViewDetails?: (productId: number) => void
}

export function ProductCard({ product, onViewDetails }: ProductCardProps) {
  const isInStock = product.stock > 0
  const priceDisplay = product.price ? `¥${product.price.toFixed(2)}` : "价格面议"

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      whileHover={{ y: -2, transition: { duration: 0.2 } }}
      className="h-full w-full min-w-0"
    >
      <Card className="h-full w-full flex flex-col group overflow-hidden border hover:border-primary/50 hover:shadow-md transition-all duration-200 bg-card">
        {/* 产品图片区域 - 超紧凑设计 */}
        {product.images && product.images.length > 0 && (
          <div className="relative w-full aspect-[3/2] sm:aspect-[4/3] overflow-hidden bg-gradient-to-br from-muted/40 to-muted/20 border-b border-border/50">
            <img
              src={product.images[0]}
              alt={product.name}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              onError={(e) => {
                e.currentTarget.style.display = 'none'
              }}
            />
            {product.special && (
              <Badge 
                variant="destructive" 
                className="absolute top-1 right-1 text-[8px] px-0.5 py-0 shadow-sm backdrop-blur-sm bg-destructive/90"
              >
                <TrendingUp className="w-1.5 h-1.5 mr-0.5" />
                特
              </Badge>
            )}
            {/* 渐变遮罩 */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/5 to-transparent pointer-events-none" />
          </div>
        )}
        
        {/* 头部区域 - 超紧凑 */}
        <CardHeader className="pb-1 pt-1.5 px-2 sm:px-3">
          <div className="flex items-start justify-between gap-1 mb-0.5">
            <h3 className="text-[10px] sm:text-xs font-semibold leading-tight line-clamp-2 flex-1 group-hover:text-primary transition-colors">
              {product.name}
            </h3>
            {product.special && (!product.images || product.images.length === 0) && (
              <Badge 
                variant="destructive" 
                className="shrink-0 ml-1 text-[8px] px-0.5 py-0"
              >
                <TrendingUp className="w-1.5 h-1.5 mr-0.5" />
                特
              </Badge>
            )}
          </div>
          
          {product.model_number && (
            <div className="hidden sm:flex items-center gap-0.5 mt-0.5">
              <Award className="w-2 h-2 text-muted-foreground" />
              <span className="text-[8px] text-muted-foreground font-mono">
                {product.model_number}
              </span>
            </div>
          )}
        </CardHeader>
        
        <CardContent className="flex-1 space-y-1 px-2 sm:px-3 pb-1.5">
          {/* 品牌和分类标签 - 移动端隐藏 */}
          {(product.brand || product.main_category) && (
            <div className="hidden sm:flex flex-wrap gap-0.5">
              {product.brand && (
                <Badge variant="secondary" className="text-[8px] px-1 py-0">
                  {product.brand}
                </Badge>
              )}
              {product.main_category && (
                <Badge variant="outline" className="text-[8px] px-1 py-0">
                  {product.main_category}
                </Badge>
              )}
            </div>
          )}

          {/* 价格和评分区域 - 超紧凑 */}
          <div className="flex items-center justify-between pt-1 pb-1 border-t border-border/50">
            <div className="flex flex-col">
              <div className="text-sm sm:text-base font-bold text-primary tracking-tight leading-none">
                {priceDisplay}
              </div>
            </div>
            {product.rating > 0 && (
              <div className="hidden sm:flex items-center gap-0.5 bg-yellow-50 dark:bg-yellow-950/20 px-1 py-0.5 rounded-full">
                <Star className="w-2 h-2 fill-yellow-400 text-yellow-400" />
                <span className="font-semibold text-[9px] text-yellow-600 dark:text-yellow-400">
                  {product.rating.toFixed(1)}
                </span>
              </div>
            )}
          </div>

          {/* 库存状态 - 超紧凑 */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-0.5">
              <Package className={cn(
                "w-2.5 h-2.5",
                isInStock ? 'text-green-500' : 'text-red-500'
              )} />
              <span className={cn(
                "text-[9px] font-medium",
                isInStock ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
              )}>
                {isInStock ? `${product.stock}` : "缺货"}
              </span>
            </div>
          </div>

          {/* 描述 - 仅在大屏显示 */}
          {product.description && (
            <div className="pt-0.5 hidden lg:block">
              <p className="text-[9px] text-muted-foreground line-clamp-1 leading-tight">
                {product.description}
              </p>
            </div>
          )}
        </CardContent>

        <CardFooter className="pt-1.5 px-2 sm:px-3 pb-2 sm:pb-2.5 gap-1.5 sm:gap-2 border-t bg-muted/20">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 text-[10px] sm:text-xs h-7 sm:h-8 px-2 sm:px-3 min-w-0 overflow-hidden"
            onClick={() => onViewDetails?.(product.id)}
          >
            <Eye className="w-3 h-3 sm:w-3.5 sm:h-3.5 shrink-0" />
            <span className="ml-1 sm:ml-1.5 truncate">详情</span>
          </Button>
          <Button
            variant="default"
            size="sm"
            className="flex-1 text-[10px] sm:text-xs h-7 sm:h-8 px-2 sm:px-3 min-w-0 overflow-hidden"
            disabled={!isInStock}
          >
            <ShoppingCart className="w-3 h-3 sm:w-3.5 sm:h-3.5 shrink-0" />
            <span className="ml-1 sm:ml-1.5 truncate">{isInStock ? "加购" : "缺货"}</span>
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
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage, setItemsPerPage] = useState(() => {
    // 初始化时根据窗口大小设置
    if (typeof window !== 'undefined') {
      const width = window.innerWidth
      if (width >= 1280) return 12  // 大屏：4列 x 3行
      if (width >= 1024) return 9   // PC：3列 x 3行
      if (width >= 768) return 6    // iPad：2-3列 x 2-3行
      return 4                       // 移动端：2列 x 2行
    }
    return 4
  })
  
  // 计算每页显示数量
  useEffect(() => {
    const calculateItemsPerPage = () => {
      const width = window.innerWidth
      let newItemsPerPage = 4
      
      if (width >= 1280) {
        newItemsPerPage = 12  // 大屏：4列
      } else if (width >= 1024) {
        newItemsPerPage = 9   // PC：3列
      } else if (width >= 768) {
        newItemsPerPage = 6   // iPad：2-3列
      } else {
        newItemsPerPage = 4   // 移动端：2列
      }
      
      setItemsPerPage(newItemsPerPage)
      
      // 如果当前页超出范围，重置到第一页
      const newTotalPages = Math.ceil(products.length / newItemsPerPage)
      if (currentPage > newTotalPages && newTotalPages > 0) {
        setCurrentPage(1)
      }
    }
    
    calculateItemsPerPage()
    window.addEventListener('resize', calculateItemsPerPage)
    return () => window.removeEventListener('resize', calculateItemsPerPage)
  }, [products.length, currentPage])

  const totalPages = Math.ceil(products.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const endIndex = startIndex + itemsPerPage
  const currentProducts = products.slice(startIndex, endIndex)

  // 当产品列表变化时，重置到第一页
  useEffect(() => {
    setCurrentPage(1)
  }, [products.length])

  const handlePrevious = () => {
    setCurrentPage((prev) => Math.max(1, prev - 1))
  }

  const handleNext = () => {
    setCurrentPage((prev) => Math.min(totalPages, prev + 1))
  }

  if (!products || products.length === 0) {
    return (
      <div className="text-center py-8 sm:py-12 text-muted-foreground">
        <Package className="w-10 h-10 sm:w-12 sm:h-12 mx-auto mb-3 sm:mb-4 opacity-50" />
        <p className="text-sm sm:text-base">暂无商品信息</p>
      </div>
    )
  }

  return (
    <div className="my-4 sm:my-6">
      <div className="mb-3 sm:mb-4 flex items-center justify-between">
        <h3 className="text-sm sm:text-lg font-semibold flex items-center gap-1.5 sm:gap-2">
          <TrendingUp className="w-4 h-4 sm:w-5 sm:h-5 text-primary" />
          <span>找到 {products.length} 个商品</span>
        </h3>
        {totalPages > 1 && (
          <span className="text-xs sm:text-sm text-muted-foreground">
            第 {currentPage} / {totalPages} 页
          </span>
        )}
      </div>
      
      {/* 响应式网格布局：移动端强制2列，iPad 2-3列，PC 3-4列，大屏4-5列 */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-1.5 sm:gap-2 md:gap-2.5 lg:gap-3">
        {currentProducts.map((product) => (
          <ProductCard
            key={product.id}
            product={product}
            onViewDetails={onViewDetails}
          />
        ))}
      </div>

      {/* 分页控件 */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-4 sm:mt-6">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePrevious}
            disabled={currentPage === 1}
            className="h-8 sm:h-9 px-3 sm:px-4"
          >
            <ChevronLeft className="w-4 h-4 mr-1" />
            <span className="text-xs sm:text-sm">上一页</span>
          </Button>
          
          <div className="flex items-center gap-1 px-2">
            {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => {
              // 只显示当前页附近的页码
              if (
                page === 1 ||
                page === totalPages ||
                (page >= currentPage - 1 && page <= currentPage + 1)
              ) {
                return (
                  <Button
                    key={page}
                    variant={page === currentPage ? "default" : "ghost"}
                    size="sm"
                    onClick={() => setCurrentPage(page)}
                    className="h-8 w-8 sm:h-9 sm:w-9 p-0 text-xs sm:text-sm"
                  >
                    {page}
                  </Button>
                )
              } else if (
                page === currentPage - 2 ||
                page === currentPage + 2
              ) {
                return <span key={page} className="text-muted-foreground">...</span>
              }
              return null
            })}
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleNext}
            disabled={currentPage === totalPages}
            className="h-8 sm:h-9 px-3 sm:px-4"
          >
            <span className="text-xs sm:text-sm">下一页</span>
            <ChevronRight className="w-4 h-4 ml-1" />
          </Button>
        </div>
      )}
    </div>
  )
}

