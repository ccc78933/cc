#!/bin/bash
echo "正在启动后端服务器..."
cd "$(dirname "$0")"
python server/app.py

