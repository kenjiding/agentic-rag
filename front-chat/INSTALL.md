# 安装说明

## 重要提示

项目已重新创建，使用 **Tailwind CSS 3.4.1**（稳定版本），配置已正确设置。

## 安装步骤

### 1. 安装依赖

```bash
cd front-chat
npm install
```

**注意**：如果遇到权限问题，请确保：
- 使用正确的 Node.js 版本（推荐 18+）
- 检查 npm 配置
- 或者使用 `pnpm` 或 `yarn` 替代

### 2. 验证 Tailwind CSS 版本

安装后，确认安装的是 Tailwind CSS 3.x：

```bash
npm list tailwindcss
```

应该显示 `tailwindcss@3.4.1`，而不是 4.x 版本。

### 3. 启动开发服务器

```bash
npm run dev
```

服务器将在 `http://localhost:5173` 启动。

## 配置说明

### ✅ 已正确配置

- **package.json**: `"tailwindcss": "3.4.1"` (精确版本，无 ^ 符号)
- **postcss.config.js**: 使用标准的 `tailwindcss` 插件
- **tailwind.config.js**: 完整的 Shadcn/UI 配置
- **src/index.css**: 使用 `@tailwind` 指令

### 如果仍然遇到 Tailwind CSS 4.x 错误

如果安装后仍然出现 Tailwind CSS 4.x 的错误，请：

1. 删除 `node_modules` 和锁文件：
```bash
rm -rf node_modules package-lock.json pnpm-lock.yaml yarn.lock
```

2. 重新安装：
```bash
npm install
```

3. 强制使用 3.4.1 版本：
```bash
npm install tailwindcss@3.4.1 --save-dev --save-exact
```

## 验证

启动服务器后，访问 `http://localhost:5173`，应该能看到：
- ✅ 页面正常加载
- ✅ 样式正常显示
- ✅ 没有 PostCSS 错误

## 故障排查

如果遇到问题：

1. **检查 Tailwind 版本**：
   ```bash
   npm list tailwindcss
   ```

2. **检查 PostCSS 配置**：
   ```bash
   cat postcss.config.js
   ```
   应该显示 `tailwindcss: {}`

3. **检查 CSS 文件**：
   ```bash
   head -3 src/index.css
   ```
   应该显示 `@tailwind base;` 等指令

4. **清除缓存**：
   ```bash
   rm -rf node_modules/.vite
   ```

