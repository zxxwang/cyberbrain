"""Settings: 所有配置项"""
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 固定解析到项目根目录（本文件在 app/config/ 下，往上三层即是根），
# 不随进程的工作目录漂移：无论从哪里启动服务，读写都指向同一个 .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # 服务配置
    APP_NAME: str = "CyberBrain"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"  # DEBUG / INFO / WARNING / ERROR

    # Qdrant配置
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_DIMENSION: int = 1024
    QDRANT_BRIEF_COLLECTION: str = "brief_memories"
    QDRANT_EPISODIC_COLLECTION: str = "episodic_memories"

    # Mem0配置
    MEM0_BRIEF_HISTORY_DB: str = "./data/brief_history.db"
    MEM0_EPISODIC_HISTORY_DB: str = "./data/episodic_history.db"

    # Redis配置（前额叶工作记忆）
    REDIS_URL: str = "redis://localhost:6379/0"

    # MongoDB配置（Agent生命事件记录）
    MONGODB_URI: str = "replace-with-yours"
    MONGODB_DB_NAME: str = "cyberbrain"
    MONGODB_COLLECTION_LIFE_EVENT: str = "agent_life_event"

    # Embedding配置（Qwen via OpenAI-compatible API）
    EMBEDDING_MODEL: str = "text-embedding-v3"
    EMBEDDING_API_KEY: str = "replace-with-yours"
    EMBEDDING_BASE_URL: str = "replace-with-yours"
    EMBEDDING_DIMENSION: int = 1024

    # LLM配置（Doubao via Ark API, you can substitute any other LLM here）
    ARK_BASE_URL: str = "replace-with-yours"
    ARK_SYSTEM_ONE_URL: str = "replace-with-yours"
    ARK_API_KEY: str = "replace-with-yours"
    DOUBAO_MODEL_PRO: str = "doubao-seed-2-0-pro-260215"
    DOUBAO_MODEL_LITE: str = "doubao-seed-2-1-lite-260915"
    DOUBAO_MODEL_CHARACTER: str = "doubao-seed-character-260628"

    # Brief稠密检索阈值
    BRIEF_DENSE_SCORE_THRESHOLD: float = 0.3

    # Brief检索的相关性过滤（qwen3-rerank交叉编码器）
    BRIEF_RERANK_ENABLED: bool = True
    BRIEF_RERANK_MIN_SCORE: float = 0.42
    BRIEF_RERANK_CANDIDATES: int = 10

    # 新旧记忆冲突排序
    BRIEF_RECENCY_TIEBREAK_MARGIN: float = 0.05  # 分数差小于此值时按updated_at排序（设0关闭）
    BRIEF_CONFLICT_SIMILARITY: float = 0.85  # 记忆间余弦≥此值才判定为同一事物的新旧版本

    # Query Rewriting & Reranking
    RERANK_MODEL: str = "qwen3-rerank"
    QUERY_REWRITE_NUM: int = 1  # 关闭改写：实验显示4改写仅多召回0.8条(+16%)但耗时增加2575ms，ROI极低
    RERANK_CANDIDATE_SIZE: int = 10  # 优化：从30降至10，减少rerank耗时
    RERANK_SKIP_THRESHOLD: float = 0.8  # 向量搜索Top1分数 > 此值时跳过rerank
    EPISODIC_RERANK_MIN_SCORE: float = 0.35  # episodic rerank 分数低于此值丢弃（实测 <0.35 基本是噪声）

    # 记忆衰减配置
    DECAY_FACTOR: float = 0.96  # 每次衰减乘以此因子
    DECAY_MIN_WEIGHT: float = 0.2  # 低于此值的记忆将被清理
    REACTIVATION_THRESHOLD: float = 0.55  # 对方的话与已有episodic记忆的余弦≥此值视为再次提及，权重恢复为1.0

    # 前额叶工作记忆（短时记忆队列 wm:{agent_id}:stm + 压缩队列 wm:{agent_id}:stm:compress）
    PFC_STM_MAX_LEN: int = 100  # stm长度超过此值触发压缩
    PFC_STM_COMPRESS_COUNT: int = 50  # 每次取最早的N条进行压缩
    PFC_COMPRESS_MAX_BLOCKS: int = 16  # 压缩块超过此值时合并最早的两块
    PFC_LOCK_TTL: int = 300  # 压缩/总结锁的过期时间（秒），防止进程崩溃后死锁
    PFC_DAILY_HOUR: int = 4  # 每日总结执行时刻（服务器本地时间，0-23）

    # 赛博大脑 react
    BRAIN_BRIEF_TOP_K: int = 10  # System1参考的信条/事实条数
    BRAIN_EPISODIC_TOP_K: int = 5  # System2参考的情景记忆条数
    BRAIN_KNOWLEDGE_SCORE_THRESHOLD: float = 0.5  # 预取知识相似度≥此值时直接使用，不依赖System1判断
    BRAIN_KNOWLEDGE_INJECT_TOP_K: int = 2  # 注入System2的知识条数（知识内容长，多了稀释注意力）
    BRAIN_KNOWLEDGE_CONTENT_CHARS: int = 1200  # 每条知识注入的最大字符数
    BRAIN_SYSTEM2_TEMPERATURE: float = 0.3  # 实测0.7时编造率显著更高，编造内容会写入PFC并被巩固成长期记忆
    BRAIN_SYSTEM2_MAX_TOKENS: int = 400  # 优化后：心里想<20字+事实来源<20字+分寸<12字，预计总输出<150字≈180 tokens
    BRAIN_SYSTEM3_TEMPERATURE: float = 0.5  # System3只负责措辞不加内容，可以比System2放开一点
    BRAIN_SYSTEM3_MAX_TOKENS: int = 1000
    # 路由：System1的noul/choice概率 -> 走哪条认知路径（阈值需用探针标定）
    BRAIN_ROUTE_REASONING_THRESHOLD: float = 0.6  # needs_reasoning ≥ 此值 -> System2
    BRAIN_ROUTE_MEMORY_THRESHOLD: float = 0.6  # needs_memory ≥ 此值 -> 带回忆
    BRAIN_ROUTE_THINK_THRESHOLD: float = 0.5  # 直觉行为里"得想一想"的概率 ≥ 此值 -> System2
    BRAIN_ROUTE_MIN_ACTION_CONFIDENCE: float = 0.55  # 直觉行为置信度低于此值时不走fast（拿不准就交给System2）
    BRAIN_ROUTE_HOSTILE_MAX_CONFLICT: float = 0.4  # 对熟人选了拒绝/顶回去，但冲突程度低于此值时，交给System2复核
    BRAIN_SYSTEM1_RECENT_TURNS: int = 4  # System1看到的最近对话条数（原文，不含本句），用于理解"需要""好的"这类省略回答

    # 待办
    TODO_DEDUP_SIMILARITY: float = 0.6  # 去重粗筛：余弦≥此值的已有任务作为候选，再由 System One 判断是否同一件事
    TODO_DEDUP_WINDOW_HOURS: int = 24  # 完成/取消的任务在这段时间内仍参与去重

    # Worker配置
    DECAY_WORKER_INTERVAL: int = 86400  # 24小时
    THOUGHT_GENERATOR_INTERVAL: int = 3600  # 念头生成间隔（秒），默认1小时
    TASK_REMINDER_INTERVAL: int = 3600  # 待办任务检查间隔（秒），默认1小时

    @field_validator("MEM0_BRIEF_HISTORY_DB", "MEM0_EPISODIC_HISTORY_DB")
    @classmethod
    def _resolve_from_project_root(cls, v: str) -> str:
        """相对路径按项目根目录解析，不随启动目录漂移（否则从 app/ 下启动会多出一个 app/data）"""
        p = Path(v)
        return str(p if p.is_absolute() else (_PROJECT_ROOT / p).resolve())

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        # .env 里出现 settings 未定义的 key 时启动直接报错，避免改了 .env 却不生效
        extra="forbid",
    )

settings = Settings()
