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
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className="h-full"
    >
      <Card className="h-full flex flex-col group overflow-hidden border hover:border-primary/50 hover:shadow-md transition-all duration-200 bg-card">
        {/* 头部区域 */}
        <CardHeader className="pb-3 pt-4 px-4">
          <div className="flex items-start justify-between gap-2 mb-1">
            <h3 className="text-sm sm:text-base font-semibold leading-tight line-clamp-2 flex-1 group-hover:text-primary transition-colors">
              {product.name}
            </h3>
            {product.special && (
              <Badge 
                variant="destructive" 
                className="shrink-0 ml-2 text-[10px] px-1.5 py-0.5"
              >
                <TrendingUp className="w-2.5 h-2.5 mr-0.5" />
                特价
              </Badge>
            )}
          </div>
          
          {product.model_number && (
            <div className="flex items-center gap-1 mt-1">
              <Award className="w-3 h-3 text-muted-foreground" />
              <span className="text-[10px] text-muted-foreground font-mono">
                {product.model_number}
              </span>
            </div>
          )}
        </CardHeader>
        
        <CardContent className="flex-1 space-y-3 px-4 pb-3">
          {/* 品牌和分类标签 */}
          {(product.brand || product.main_category) && (
            <div className="flex flex-wrap gap-1.5">
              {product.brand && (
                <Badge variant="secondary" className="text-[10px] px-2 py-0.5">
                  {product.brand}
                </Badge>
              )}
              {product.main_category && (
                <Badge variant="outline" className="text-[10px] px-2 py-0.5">
                  {product.main_category}
                </Badge>
              )}
            </div>
          )}

          {/* 价格和评分区域 */}
          <div className="flex items-center justify-between pt-2 pb-2 border-t border-border/50">
            <div className="flex flex-col">
              <span className="text-[10px] text-muted-foreground mb-0.5">价格</span>
              <div className="text-xl sm:text-2xl font-bold text-primary tracking-tight">
                {priceDisplay}
              </div>
            </div>
            {product.rating > 0 && (
              <div className="flex items-center gap-1 bg-yellow-50 dark:bg-yellow-950/20 px-2 py-1 rounded-full">
                <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                <span className="font-semibold text-xs text-yellow-600 dark:text-yellow-400">
                  {product.rating.toFixed(1)}
                </span>
              </div>
            )}
          </div>

          {/* 库存状态 */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Package className={cn(
                "w-3.5 h-3.5",
                isInStock ? 'text-green-500' : 'text-red-500'
              )} />
              <span className={cn(
                "text-xs font-medium",
                isInStock ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
              )}>
                {isInStock ? `库存: ${product.stock}件` : "暂时缺货"}
              </span>
            </div>
          </div>

          {/* 描述 */}
          {product.description && (
            <div className="pt-1">
              <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
                {product.description}
              </p>
            </div>
          )}
        </CardContent>

        <CardFooter className="pt-3 px-4 pb-4 gap-2 border-t bg-muted/20">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 text-xs h-8"
            onClick={() => onViewDetails?.(product.id)}
          >
            <Eye className="w-3 h-3 mr-1" />
            详情
          </Button>
          <Button
            variant="default"
            size="sm"
            className="flex-1 text-xs h-8"
            disabled={!isInStock}
          >
            <ShoppingCart className="w-3 h-3 mr-1" />
            {isInStock ? "加购" : "缺货"}
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
      return window.innerWidth >= 768 ? 6 : 4
    }
    return 4
  })
  
  // 计算每页显示数量
  useEffect(() => {
    const calculateItemsPerPage = () => {
      const isDesktop = window.innerWidth >= 768
      const newItemsPerPage = isDesktop ? 6 : 4
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
      
      <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-3 gap-3 sm:gap-4">
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

