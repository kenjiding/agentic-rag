/** 消息类型 */
export type MessageRole = "user" | "assistant" | "system";

/** 响应内容类型 */
export type ResponseType = "text" | "product_list" | "order_list" | "mixed";

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
}

/** 结构化响应数据 */
export interface ResponseData {
  products?: Product[];
  orders?: Order[];
  [key: string]: any;
}

/** 聊天消息 */
export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;  // AI 生成的文本描述
  responseType: ResponseType;  // 响应类型
  responseData?: ResponseData;  // 结构化数据
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
  type: "state_update" | "done" | "error";
  data?: {
    content?: string;
    role?: string;
    response_type?: ResponseType;
    response_data?: ResponseData;
    current_agent?: string;
    tools_used?: any[];
    execution_steps?: string[];
    step_details?: ExecutionStepDetail[];
  };
  error?: string;
}

