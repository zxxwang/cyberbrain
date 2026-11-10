#!/bin/bash
# CyberBrain 启动脚本

set -e

cd "$(dirname "$0")"

echo "🧠 CyberBrain"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. 检查虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ 虚拟环境已激活"
else
    echo "⚠️  未找到 .venv，使用系统 Python"
fi

# 2. 检查环境配置
if [ ! -f ".env" ]; then
    echo "❌ 未找到 .env 文件"
    echo "请先复制 .env.example 并配置必要的环境变量"
    exit 1
fi
echo "✓ 环境配置已加载"

# 3. 检查依赖服务
echo ""
echo "检查依赖服务..."

if ! curl -s http://localhost:6333/ > /dev/null 2>&1; then
    echo "⚠️  Qdrant 未运行，尝试启动..."
    docker-compose up -d qdrant redis
    echo "等待服务启动..."
    sleep 3
fi
echo "✓ Qdrant 可用"

if ! redis-cli ping > /dev/null 2>&1; then
    echo "⚠️  Redis 未运行"
else
    echo "✓ Redis 可用"
fi

# 4. 启动服务
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 API: http://localhost:8000"
echo "📖 文档: http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 main.py
