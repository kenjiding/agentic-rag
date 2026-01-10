/** 消息类型 */
export type MessageRole = "user" | "assistant" | "system";

/** 响应内容类型 */
export type ResponseType =
  | "text"
  | "product_list"
  | "order_list"
  | "mixed"
  | "interrupt"
  | "confirmation"
  | "product_comparison"
  | "error";

/** 产品信息 */
export interface Product {
  id: number;
  name: string;
  model_number?: string;
  brand?: string;
  main_category?: string;
  sub_category?: string;
  price?: number;
  stock: number;
  rating: number;
  special: boolean;
  description?: string;
  images?: string[];
}

/** 订单信息 */
export interface Order {
  id: number;
  order_number: string;
  status: "pending" | "paid" | "shipped" | "delivered" | "cancelled";
  total_amount: number;
  created_at: string;
  items: OrderItem[];
}

/** 订单项 */
export interface OrderItem {
  product_name: string;
  quantity: number;
  subtotal: number;
  product_images?: string[];
}

/** 产品对比数据 */
export interface ProductComparisonData {
  comparison_aspects: string[];
  comparison_details: Record<string, Record<string, string>>;
  scenario_analysis?: {
    场景?: string;
    评分?: Record<string, number>;
    推荐理由?: string;
  };
  recommendation?: string;
  products: Product[];
}

/** 结构化响应数据 */
export interface ResponseData {
  products?: Product[];
  orders?: Order[];
  comparison_aspects?: string[];
  comparison_details?: Record<string, Record<string, string>>;
  scenario_analysis?: ProductComparisonData["scenario_analysis"];
  recommendation?: string;
  [key: string]: any;
}

/** 待确认操作数据（用于 UI 展示） */
export interface ConfirmationDisplayData {
  items?: Array<{
    name: string;
    quantity: number;
    price?: number;
    subtotal: number;
    product_images?: string[];
  }>;
  total_amount?: number;
  order?: Order;
  [key: string]: any;
}

/** 待确认操作 */
export interface ConfirmationPending {
  confirmation_id: string;
  action_type: string;
  display_message: string;
  display_data?: ConfirmationDisplayData;
  expires_at?: string;
}

/** 聊天消息 */
export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;  // AI 生成的文本描述
  responseType: ResponseType;  // 响应类型
  responseData?: ResponseData;  // 结构化数据
  confirmationPending?: ConfirmationPending;  // 待确认操作
  metadata?: {
    current_agent?: string;
    tools_used?: Array<{
      agent?: string;
      tool?: string;
      args?: any;
      result?: any;
    }>;
    execution_steps?: string[];
    step_details?: ExecutionStepDetail[];
  };
  timestamp: Date;
  isStreaming?: boolean;
}

/** 执行步骤详情 */
export interface ExecutionStepDetail {
  name: string;
  detail?: string;
  status: "running" | "completed" | "error";
}

/** 流式响应事件 */
export interface StreamEvent {
  type: "state_update" | "done" | "error" | "confirmation_resolved";
  data?: {
    content?: string;
    role?: string;
    response_type?: ResponseType;
    response_data?: ResponseData;
    current_agent?: string;
    tools_used?: any[];
    execution_steps?: string[];
    step_details?: ExecutionStepDetail[];
    confirmation_pending?: ConfirmationPending;  // 待确认操作
  };
  message?: string;  // 用于 confirmation_resolved 等事件
  error?: string;
}

/** 确认解析请求 */
export interface ConfirmationResolveRequest {
  confirmation_id: string;
  confirmed: boolean;
}

/** 确认解析响应 */
export interface ConfirmationResolveResponse {
  success: boolean;
  status: string;
  action_type: string;
  message: string;
  data?: any;
  error?: string;
}

