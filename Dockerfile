# 基础镜像
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（仅编译期需要）
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc g++ && \
    apt-get autoremove -y

# 复制应用代码
COPY app/ ./app/
COPY main.py .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

# 启动命令
CMD ["python", "main.py"]
