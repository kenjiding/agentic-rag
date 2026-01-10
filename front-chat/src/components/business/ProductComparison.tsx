import { ProductComparisonData } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Award, Lightbulb, Star, Sparkles, BarChart3 } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface ProductComparisonProps {
  data: ProductComparisonData;
}

export function ProductComparison({ data }: ProductComparisonProps) {
  const { comparison_aspects, comparison_details, scenario_analysis, recommendation, products } = data;

  // 获取所有产品名称（用于渲染对比详情）
  const productNames = products.map(p => p.name);
  
  // 获取产品在对比详情中的实际名称（可能和products数组中的名称不同）
  const getProductNameInDetails = (index: number): string => {
    if (comparison_aspects.length > 0) {
      const firstAspect = comparison_aspects[0];
      const details = comparison_details[firstAspect];
      if (details) {
        const names = Object.keys(details);
        return names[index] || productNames[index] || `产品 ${index + 1}`;
      }
    }
    return productNames[index] || `产品 ${index + 1}`;
  };

  return (
    <div className="space-y-6">
      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(5px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>

      {/* 对比维度分析 - 表格化展示 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
      >
        <Card className="overflow-hidden border-2 border-border/50 shadow-lg bg-gradient-to-br from-background to-muted/10">
          <CardHeader className="bg-gradient-to-r from-muted/50 via-muted/30 to-transparent border-b border-border/50">
            <CardTitle className="flex items-center gap-2.5 text-xl font-bold">
              <div className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20">
                <BarChart3 className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <span>详细对比</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 md:p-6">
            {/* 移动端：卡片式堆叠布局 */}
            <div className="block md:hidden space-y-4 p-4">
              {comparison_aspects.map((aspect, aspectIndex) => {
                const details = comparison_details[aspect];
                if (!details) return null;

                return (
                  <motion.div
                    key={aspect}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: aspectIndex * 0.05 }}
                    className="space-y-3 p-4 rounded-lg border border-border/50 bg-card"
                  >
                    <Badge 
                      variant="outline" 
                      className="text-sm font-semibold px-3 py-1 bg-primary/5 border-primary/30 text-primary mb-3"
                    >
                      {aspect}
                    </Badge>
                    
                    <div className="space-y-3">
                      {Object.entries(details).map(([productName, detail], productIndex) => (
                        <motion.div
                          key={productName}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.2, delay: aspectIndex * 0.05 + productIndex * 0.03 }}
                          className={cn(
                            "relative pl-4 pr-3 py-3 rounded-lg border-l-4 transition-all duration-200",
                            "bg-gradient-to-r from-card to-muted/10",
                            productIndex === 0 
                              ? "border-l-blue-500" 
                              : "border-l-purple-500"
                          )}
                        >
                          <div className="font-semibold text-sm text-foreground/90 mb-1.5">
                            {productName}
                          </div>
                          <div className="text-sm text-muted-foreground leading-relaxed">
                            {detail}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {/* 桌面端：表格布局（支持横向滚动） */}
            <div className="hidden md:block w-full">
              <ScrollArea className="w-full">
                <div className="min-w-[800px]">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-b-2 border-primary/20 bg-gradient-to-r from-muted/50 via-muted/40 to-muted/30 hover:bg-gradient-to-r">
                        <TableHead className="sticky left-0 z-20 px-6 py-4 text-left font-bold text-sm bg-gradient-to-r from-muted/50 via-muted/40 to-muted/30 backdrop-blur-sm min-w-[160px] shadow-[2px_0_4px_rgba(0,0,0,0.05)]">
                          <div className="flex items-center gap-2">
                            <BarChart3 className="w-4 h-4 text-primary" />
                            <span>对比维度</span>
                          </div>
                        </TableHead>
                        {products.map((product, index) => {
                          const displayName = getProductNameInDetails(index);
                          return (
                            <TableHead
                              key={product.id}
                              className={cn(
                                "px-6 py-4 text-center font-bold text-sm min-w-[220px] max-w-[280px]",
                                "relative border-l border-border/40 bg-gradient-to-b from-transparent to-muted/20"
                              )}
                            >
                              <div className="flex flex-col items-center gap-2.5">
                                {/* 产品编号徽章 */}
                                <div className={cn(
                                  "w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white shadow-lg",
                                  index === 0 
                                    ? "bg-gradient-to-br from-blue-500 to-blue-600 ring-2 ring-blue-400/30" 
                                    : "bg-gradient-to-br from-purple-500 to-purple-600 ring-2 ring-purple-400/30"
                                )}>
                                  {index + 1}
                                </div>
                                
                                <div className="space-y-1.5">
                                  <div className="font-bold text-base text-foreground leading-tight">
                                    {displayName}
                                  </div>
                                  <div className="flex flex-col gap-1 items-center">
                                    {product.brand && (
                                      <Badge variant="secondary" className="text-xs px-2 py-0.5">
                                        {product.brand}
                                      </Badge>
                                    )}
                                    {product.price && (
                                      <div className="text-primary font-bold text-sm">
                                        ¥{product.price.toLocaleString()}
                                      </div>
                                    )}
                                  </div>
                                </div>
                                
                                {/* 列标识色块 */}
                                <div className={cn(
                                  "absolute bottom-0 left-0 right-0 h-1.5 rounded-t-full",
                                  index === 0 
                                    ? "bg-gradient-to-r from-blue-500 to-blue-600" 
                                    : "bg-gradient-to-r from-purple-500 to-purple-600"
                                )} />
                              </div>
                            </TableHead>
                          );
                        })}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {comparison_aspects.map((aspect, aspectIndex) => {
                        const details = comparison_details[aspect];
                        if (!details) return null;

                        return (
                          <TableRow
                            key={aspect}
                            className={cn(
                              "border-b border-border/20 transition-all duration-200 group",
                              "hover:bg-gradient-to-r hover:from-muted/40 hover:via-muted/20 hover:to-transparent"
                            )}
                            style={{
                              animation: `fadeInUp 0.3s ease-out ${aspectIndex * 0.03}s both`,
                            }}
                          >
                            {/* 维度名称列（固定） */}
                            <TableCell className="sticky left-0 z-10 px-6 py-4 bg-gradient-to-r from-card via-card to-transparent backdrop-blur-sm border-r border-border/30 shadow-[2px_0_4px_rgba(0,0,0,0.05)] group-hover:bg-gradient-to-r group-hover:from-muted/40 group-hover:via-muted/30 group-hover:to-transparent">
                              <Badge 
                                variant="outline" 
                                className="text-sm font-semibold px-3 py-1.5 bg-primary/5 border-primary/30 text-primary whitespace-nowrap"
                              >
                                {aspect}
                              </Badge>
                            </TableCell>
                            
                            {/* 产品对比数据列 */}
                            {products.map((product, productIndex) => {
                              const displayName = getProductNameInDetails(productIndex);
                              const detail = details[displayName] || details[product.name] || "-";
                              
                              return (
                                <TableCell
                                  key={product.id}
                                  className={cn(
                                    "px-6 py-4 text-left border-l border-border/20",
                                    "bg-gradient-to-br from-card/50 to-transparent",
                                    "group-hover:bg-gradient-to-br group-hover:from-muted/30 group-hover:to-transparent",
                                    "transition-colors duration-200"
                                  )}
                                >
                                  <div className={cn(
                                    "text-sm leading-relaxed",
                                    "text-muted-foreground group-hover:text-foreground/80",
                                    "transition-colors duration-200"
                                  )}>
                                    {detail}
                                  </div>
                                </TableCell>
                              );
                            })}
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
                <ScrollBar orientation="horizontal" />
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* 场景化分析 */}
      {scenario_analysis && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
        >
          <Card className="overflow-hidden border-2 border-amber-500/30 bg-gradient-to-br from-amber-50/50 via-background to-purple-50/30 dark:from-amber-950/20 dark:via-background dark:to-purple-950/20 shadow-xl">
            <CardHeader className="bg-gradient-to-r from-amber-500/10 via-amber-400/5 to-transparent border-b border-amber-500/20 p-4 md:p-6">
              <CardTitle className="flex flex-col sm:flex-row items-start sm:items-center gap-2 sm:gap-2.5 text-lg md:text-xl font-bold">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/30">
                    <Award className="w-4 h-4 md:w-5 md:h-5 text-amber-600 dark:text-amber-400" />
                  </div>
                  <span>场景化分析</span>
                </div>
                {scenario_analysis.场景 && (
                  <Badge 
                    variant="secondary" 
                    className="bg-gradient-to-r from-amber-500/20 to-amber-400/10 border-amber-500/30 text-amber-700 dark:text-amber-300 font-medium text-xs md:text-sm"
                  >
                    <Sparkles className="w-3 h-3 mr-1" />
                    {scenario_analysis.场景}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 md:p-6 space-y-4 md:space-y-5">
              {scenario_analysis.评分 && (
                <div>
                  <div className="font-semibold text-sm md:text-base mb-3 md:mb-4 flex items-center gap-2">
                    <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                    场景评分
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 md:gap-4">
                    {Object.entries(scenario_analysis.评分).map(([productName, score], index) => {
                      const scorePercent = (score / 10) * 100;
                      const isWinner = score === Math.max(...Object.values(scenario_analysis.评分 || {}));
                      
                      return (
                        <motion.div
                          key={productName}
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.3, delay: index * 0.1 }}
                          className={cn(
                            "relative p-3 md:p-4 rounded-xl border-2 transition-all duration-300",
                            "bg-gradient-to-br from-card via-card to-muted/20",
                            isWinner 
                              ? "border-amber-500/50 shadow-lg ring-2 ring-amber-500/20" 
                              : "border-border/50 hover:shadow-md"
                          )}
                        >
                          {isWinner && (
                            <div className="absolute -top-2 -right-2">
                              <Badge className="bg-gradient-to-r from-amber-500 to-amber-600 text-white border-0 shadow-lg text-xs">
                                <Award className="w-3 h-3 mr-1" />
                                推荐
                              </Badge>
                            </div>
                          )}
                          
                          <div className="space-y-2 md:space-y-3 pr-10 md:pr-12">
                            <div className="font-semibold text-sm md:text-base">{productName}</div>
                            
                            {/* 评分可视化 */}
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <span className="text-2xl font-bold text-primary">{score.toFixed(1)}</span>
                                <span className="text-sm text-muted-foreground">/ 10</span>
                              </div>
                              
                              {/* 进度条 */}
                              <div className="relative h-3 bg-muted rounded-full overflow-hidden">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${scorePercent}%` }}
                                  transition={{ duration: 0.8, delay: index * 0.1, ease: "easeOut" }}
                                  className={cn(
                                    "absolute top-0 left-0 h-full rounded-full bg-gradient-to-r",
                                    isWinner
                                      ? "from-amber-500 to-amber-600"
                                      : "from-blue-500 to-blue-600"
                                  )}
                                />
                              </div>
                              
                              {/* 星级显示 */}
                              <div className="flex items-center gap-1">
                                {Array.from({ length: 5 }).map((_, i) => (
                                  <Star
                                    key={i}
                                    className={cn(
                                      "w-4 h-4",
                                      i < Math.round(score / 2)
                                        ? "fill-yellow-400 text-yellow-400"
                                        : "fill-muted text-muted"
                                    )}
                                  />
                                ))}
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                </div>
              )}
              
              {scenario_analysis.推荐理由 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.4 }}
                  className="p-3 md:p-4 rounded-xl bg-gradient-to-br from-amber-50/80 to-amber-100/40 dark:from-amber-950/40 dark:to-amber-900/20 border border-amber-500/30"
                >
                  <div className="font-semibold text-sm md:text-base mb-2 flex items-center gap-2 text-amber-900 dark:text-amber-100">
                    <Lightbulb className="w-3.5 h-3.5 md:w-4 md:h-4" />
                    推荐理由
                  </div>
                  <div className="text-xs md:text-sm leading-relaxed text-amber-800 dark:text-amber-200">
                    {scenario_analysis.推荐理由}
                  </div>
                </motion.div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* 综合推荐 */}
      {recommendation && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.4 }}
        >
          <Card className="overflow-hidden border-2 border-primary/50 bg-gradient-to-br from-primary/10 via-primary/5 to-primary/10 shadow-2xl ring-2 ring-primary/20">
            <CardHeader className="bg-gradient-to-r from-primary/20 via-primary/10 to-transparent border-b border-primary/30 p-4 md:p-6">
              <CardTitle className="flex items-center gap-2 sm:gap-2.5 text-lg md:text-xl font-bold text-primary">
                <div className="p-1.5 md:p-2 rounded-lg bg-primary/20 border border-primary/40">
                  <Lightbulb className="w-4 h-4 md:w-5 md:h-5 text-primary" />
                </div>
                <span>综合推荐</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 md:p-6">
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.5 }}
                className="text-sm md:text-base leading-relaxed text-foreground font-medium"
              >
                {recommendation}
              </motion.p>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
}