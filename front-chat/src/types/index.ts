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

/** 结构化响应数据 */
export interface ResponseData {
  products?: Product[];
  orders?: Order[];
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

/** 待选择操作 */
export interface PendingSelection {
  selection_id: string;
  selection_type: string;  // "product", "address", etc.
  options: Product[];  // 可选项列表（产品选择时为 Product[]）
  display_message: string;
  metadata?: {
    task_chain_id?: string;
    [key: string]: any;
  };
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
  pendingSelection?: PendingSelection;  // 待选择操作（新增）
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
  type: "state_update" | "done" | "error" | "selection_resolved" | "confirmation_resolved";
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
    pending_selection?: PendingSelection;  // 待选择操作（新增）
  };
  message?: string;  // 用于 selection_resolved 和 confirmation_resolved 等事件
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

/** 选择解析请求 */
export interface SelectionResolveRequest {
  selection_id: string;
  selected_option_id: string;
}

/** 选择解析响应 */
export interface SelectionResolveResponse {
  success: boolean;
  status: string;
  selection_type: string;
  selected_option?: Product | any;
  message?: string;
  error?: string;
}

