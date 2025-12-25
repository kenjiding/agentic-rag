# AI 智能客服前端

基于 React 18 + Vite 6 + TypeScript 构建的现代化电商智能客服前端系统。

## 技术栈

- **React 18**: UI 框架
- **Vite 6**: 构建工具
- **TypeScript**: 类型安全
- **Tailwind CSS 3.4.1**: 样式框架（稳定版本）
- **Shadcn/UI**: 基于 Radix UI 的组件库
- **Framer Motion**: 动画库
- **React Markdown**: Markdown 渲染

## 快速开始

### 1. 安装依赖

```bash
npm install
```

**重要**：确保安装的是 Tailwind CSS 3.4.1，而不是 4.x 版本。

### 2. 启动开发服务器

```bash
npm run dev
```

访问 http://localhost:5173

### 3. 构建生产版本

```bash
npm run build
```

## 项目结构

```
front-chat/
├── src/
│   ├── components/
│   │   ├── ui/          # Shadcn/UI 基础组件
│   │   ├── business/    # 业务组件（ProductCard, OrderTracker）
│   │   └── chat/        # 聊天核心组件
│   ├── hooks/           # 自定义 Hooks
│   ├── lib/             # 工具函数
│   ├── types/           # TypeScript 类型定义
│   ├── App.tsx          # 主应用组件
│   └── main.tsx         # 入口文件
├── public/              # 静态资源
└── vite.config.ts       # Vite 配置
```

## 功能特性

- ✅ 流式消息渲染（打字机效果）
- ✅ Markdown 支持（代码块、表格、列表等）
- ✅ 结构化消息渲染
  - 产品卡片展示（ProductCard）
  - 订单追踪器（OrderTracker）
  - 状态标签（StatusBadge）
- ✅ 响应式设计
- ✅ 错误处理和重连机制

## API 集成

前端通过 `/api/chat` 端点与后端通信，支持 Server-Sent Events (SSE) 流式响应。

### 请求格式

```json
{
  "message": "用户消息",
  "session_id": "default",
  "stream": true
}
```

### 响应格式（SSE）

```
data: {"type": "state_update", "data": {"content": "..."}}
data: {"type": "done"}
```

## 后端要求

确保后端 API 服务器运行在 `http://localhost:8000`，并支持：

1. CORS 跨域
2. SSE 流式响应
3. `/api/chat` 端点

## 开发说明

### 添加新的业务组件

1. 在 `src/components/business/` 创建组件
2. 在 `MessageList.tsx` 中集成渲染逻辑
3. 更新 `types/index.ts` 中的类型定义

### 自定义样式

修改 `src/index.css` 中的 CSS 变量来调整主题。

## 故障排查

如果遇到 Tailwind CSS 相关错误，请查看 [INSTALL.md](./INSTALL.md) 获取详细说明。

## License

MIT

