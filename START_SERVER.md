# 启动服务器说明

## 问题诊断

如果遇到 `ECONNREFUSED` 错误，说明后端服务器没有运行。

## 启动步骤

### 1. 启动后端服务器

在项目根目录（`shipin`）下运行：

```bash
cd server
python app.py
```

或者：

```bash
python server/app.py
```

后端服务器默认运行在 `http://localhost:5000`

### 2. 启动前端开发服务器

在另一个终端窗口中，进入前端目录：

```bash
cd dish-nutrition-vue
npm run dev
```

前端开发服务器默认运行在 `http://localhost:5173`

## 检查服务器状态

### 检查后端是否运行

在浏览器中访问：`http://localhost:5000/api/ping`

如果返回 `{"ok": true, "version": "1.0.0"}` 或类似内容，说明后端正常运行。

### 检查端口占用

Windows:
```bash
netstat -ano | findstr :5000
```

Linux/Mac:
```bash
lsof -i :5000
```

## 常见问题

### 1. 端口被占用

如果 5000 端口被占用，可以修改端口：

**方法1：使用环境变量**
```bash
# Windows PowerShell
$env:API_PORT="5001"
python server/app.py

# Windows CMD
set API_PORT=5001
python server/app.py

# Linux/Mac
export API_PORT=5001
python server/app.py
```

**方法2：修改 vite.config.js**
将 `proxyTarget` 改为其他端口，例如：
```javascript
const proxyTarget = env.VITE_DEV_PROXY_TARGET || env.VITE_API_BASE_URL || 'http://localhost:5001'
```

### 2. 后端服务器启动失败

检查：
- Python 环境是否正确
- 依赖是否安装：`pip install -r requirements.txt`
- 数据库配置是否正确（检查 `.env` 文件）
- MySQL 服务是否运行（如果使用 MySQL）

### 3. 代理配置

确保 `vite.config.js` 中的 `enableProxy` 为 `true`，或者设置环境变量：
```bash
VITE_DEV_PROXY=1 npm run dev
```

## 快速启动脚本

### Windows (start_server.bat)
```batch
@echo off
echo Starting backend server...
start cmd /k "cd server && python app.py"
timeout /t 3
echo Starting frontend server...
cd dish-nutrition-vue
npm run dev
```

### Linux/Mac (start_server.sh)
```bash
#!/bin/bash
echo "Starting backend server..."
cd server && python app.py &
sleep 3
echo "Starting frontend server..."
cd ../dish-nutrition-vue
npm run dev
```

