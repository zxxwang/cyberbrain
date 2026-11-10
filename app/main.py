"""
Main Application: FastAPI应用入口

职责：
- 初始化所有服务（repository、workers）
- 注册路由
- 启动/关闭生命周期管理
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.client.redis_client import redis_client
from app.client.mongodb_client import mongodb_client
from app.repository.agent_life_event_repository import agent_life_event_repository
from app.common.httpx_patch import apply_httpx_patch
from app.common.middleware import LoggingMiddleware
from app.config.settings import settings
from app.controller import (
    admin_controller,
    cyberbrain_controller,
    health_controller,
)
from app.repository.memory_repository import memory_repository
from app.service.knowledge_memory_service import knowledge_memory_service
from app.service.subjectivity_service import subjectivity_service
from app.worker.decay_worker import decay_worker
from app.worker.prefrontal_lobe_memory_worker import prefrontal_lobe_memory_worker
from app.worker.thought_generator_worker import thought_generator_worker
from app.worker.task_reminder_worker import task_reminder_worker

# 配置日志
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 应用httpx补丁（必须在任何httpx使用之前）
apply_httpx_patch()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==========================
    # Startup
    # ==========================
    """应用启动时初始化"""
    logger.info("Starting CyberBrain...")

    # 初始化memory repository
    success = await memory_repository.initialize()
    if not success:
        logger.error("Failed to initialize memory repository, service will not function")
        return

    # 初始化knowledge service
    success = await knowledge_memory_service.initialize()
    if not success:
        logger.error("Failed to initialize knowledge memory service")
        return

    # 初始化 MongoDB
    success = await mongodb_client.connect()
    if not success:
        logger.warning("Failed to connect to MongoDB, life event recording disabled")
    else:
        logger.info("MongoDB connected successfully")
        # 创建索引
        await agent_life_event_repository.ensure_indexes()

    # 旧数据迁移：把旧版本留在 pending_tasks 里的已完成/已取消任务移出（幂等）
    try:
        await subjectivity_service.migrate_finished_tasks()
    except Exception as e:
        logger.error(f"Failed to migrate finished tasks: {e}", exc_info=True)

    # 启动workers
    asyncio.create_task(decay_worker.start())
    asyncio.create_task(prefrontal_lobe_memory_worker.start())
    asyncio.create_task(thought_generator_worker.start())
    asyncio.create_task(task_reminder_worker.start())

    logger.info(f"Service started on {settings.HOST}:{settings.PORT}")

    yield

    # ==========================
    # Shutdown
    # ==========================
    """应用关闭时清理"""
    logger.info("Shutting down CyberBrain...")

    # 停止workers
    decay_worker.stop()
    prefrontal_lobe_memory_worker.stop()
    thought_generator_worker.stop()
    task_reminder_worker.stop()

    # 关闭连接
    await redis_client.close()
    await mongodb_client.disconnect()

    logger.info("Service shutdown complete")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    description="CyberBrain: AI Agent Memory System with Brief and Episodic memory support"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 日志中间件
app.add_middleware(LoggingMiddleware)

# 注册路由
app.include_router(health_controller.router)
app.include_router(admin_controller.router)
app.include_router(cyberbrain_controller.router)

# 挂载静态文件（使用项目根目录的 static 文件夹）
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # 首页重定向到初始化页面
    @app.get("/")
    async def root():
        return FileResponse(str(static_dir / "index.html"))
else:
    logger.warning(f"Static directory not found: {static_dir}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
