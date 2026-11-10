"""
运行时可调配置：可通过 /admin/config 接口查询和修改的配置项（白名单）

这些配置在代码里都是"调用时读 settings.X"，修改 settings 对象后，下一次读取即为新值。
修改时先写回 .env（重启后延续），再改内存中的 settings 对象（立即生效）。
注意：操作系统环境变量优先级高于 .env，若部署时用环境变量设置了同名项，重启后仍以环境变量为准；
多 worker 进程部署时，其他进程内存中的值不会同步，需重启后从 .env 读取。

新增可调项时：先确认代码里是在函数调用时读取（不是 __init__ 或模块顶层缓存），再加到 RUNTIME_CONFIGS。
"""
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config.settings import Settings, settings, _PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfigSpec:
    description: str
    min: Optional[float] = None  # 数值下限（含），bool 项不填
    max: Optional[float] = None  # 数值上限（含）


# 顺序与 settings.py 一致
RUNTIME_CONFIGS: Dict[str, RuntimeConfigSpec] = {
    # Brief 检索
    "BRIEF_DENSE_SCORE_THRESHOLD": RuntimeConfigSpec("Brief稠密检索的相似度阈值，低于此值的召回结果丢弃", 0, 1),
    "BRIEF_RERANK_ENABLED": RuntimeConfigSpec("Brief检索是否启用qwen3-rerank交叉编码器做相关性过滤"),
    "BRIEF_RERANK_MIN_SCORE": RuntimeConfigSpec("Brief rerank分数低于此值的记忆丢弃", 0, 1),
    "BRIEF_RERANK_CANDIDATES": RuntimeConfigSpec("Brief检索送入rerank的候选条数（稠密和BM25各召回这么多）", 1, 100),
    "BRIEF_RECENCY_TIEBREAK_MARGIN": RuntimeConfigSpec("新旧记忆冲突排序：分数差小于此值时按updated_at排序（设0关闭）", 0, 1),
    "BRIEF_CONFLICT_SIMILARITY": RuntimeConfigSpec("记忆间余弦≥此值才判定为同一事物的新旧版本", 0, 1),
    # Episodic 检索
    "QUERY_REWRITE_NUM": RuntimeConfigSpec("Episodic检索的查询改写数量（含原查询），1表示不改写", 1, 10),
    "RERANK_CANDIDATE_SIZE": RuntimeConfigSpec("Episodic检索召回后送入rerank的候选总数", 1, 100),
    "RERANK_SKIP_THRESHOLD": RuntimeConfigSpec("向量搜索Top1分数大于此值时跳过rerank", 0, 1),
    "EPISODIC_RERANK_MIN_SCORE": RuntimeConfigSpec("Episodic rerank分数低于此值的记忆丢弃", 0, 1),
    # 记忆衰减
    "DECAY_FACTOR": RuntimeConfigSpec("记忆衰减：每次衰减时权重乘以此因子", 0, 1),
    "DECAY_MIN_WEIGHT": RuntimeConfigSpec("记忆衰减：权重低于此值的记忆将被清理", 0, 1),
    "REACTIVATION_THRESHOLD": RuntimeConfigSpec("记忆重激活：对方的话与已有情景记忆的余弦相似度≥此值时视为再次提及，权重恢复为1.0", 0, 1),
    # 前额叶工作记忆
    "PFC_STM_MAX_LEN": RuntimeConfigSpec("PFC短时记忆队列长度超过此值时触发压缩", 1, 10000),
    "PFC_STM_COMPRESS_COUNT": RuntimeConfigSpec("PFC每次压缩时取最早的N条记录（不能大于PFC_STM_MAX_LEN）", 1, 10000),
    "PFC_COMPRESS_MAX_BLOCKS": RuntimeConfigSpec("PFC压缩块数量超过此值时合并最早的两块", 1, 1000),
    "PFC_LOCK_TTL": RuntimeConfigSpec("PFC压缩/总结锁的过期时间（秒），防止进程崩溃后死锁", 1, 86400),
    "PFC_DAILY_HOUR": RuntimeConfigSpec("PFC每日总结执行时刻（服务器本地时间，0-23）；当前这一轮等待结束后才按新时刻计算", 0, 23),
    # 赛博大脑 react
    "BRAIN_BRIEF_TOP_K": RuntimeConfigSpec("System1参考的信条/事实条数", 1, 50),
    "BRAIN_EPISODIC_TOP_K": RuntimeConfigSpec("System2参考的情景记忆条数", 1, 50),
    "BRAIN_KNOWLEDGE_SCORE_THRESHOLD": RuntimeConfigSpec("预取知识相似度≥此值时直接使用，不依赖System1判断", 0, 1),
    "BRAIN_KNOWLEDGE_INJECT_TOP_K": RuntimeConfigSpec("注入System2的知识条数（知识内容长，多了稀释注意力）", 0, 20),
    "BRAIN_KNOWLEDGE_CONTENT_CHARS": RuntimeConfigSpec("每条知识注入System2的最大字符数", 1, 20000),
    "BRAIN_SYSTEM2_TEMPERATURE": RuntimeConfigSpec("System2温度（实测0.7时编造率显著更高）", 0, 2),
    "BRAIN_SYSTEM2_MAX_TOKENS": RuntimeConfigSpec("System2最大输出token数", 1, 32000),
    "BRAIN_SYSTEM3_TEMPERATURE": RuntimeConfigSpec("System3温度（只负责措辞不加内容，可比System2放开一点）", 0, 2),
    "BRAIN_SYSTEM3_MAX_TOKENS": RuntimeConfigSpec("System3最大输出token数", 1, 32000),
    "BRAIN_ROUTE_REASONING_THRESHOLD": RuntimeConfigSpec("路由：System1判断needs_reasoning≥此值时走System2", 0, 1),
    "BRAIN_ROUTE_MEMORY_THRESHOLD": RuntimeConfigSpec("路由：needs_memory≥此值时带回忆", 0, 1),
    "BRAIN_ROUTE_THINK_THRESHOLD": RuntimeConfigSpec("路由：直觉行为里\"得想一想\"的概率≥此值时走System2", 0, 1),
    "BRAIN_ROUTE_MIN_ACTION_CONFIDENCE": RuntimeConfigSpec("路由：System1直觉行为的置信度低于此值时不走fast，交给System2", 0, 1),
    "BRAIN_ROUTE_HOSTILE_MAX_CONFLICT": RuntimeConfigSpec("路由：对关系表里的人选了拒绝/顶回去，但话题冲突程度低于此值时，交给System2复核", 0, 1),
    "BRAIN_SYSTEM1_RECENT_TURNS": RuntimeConfigSpec("System1看到的最近对话条数（不含本句），用于理解省略回答；0表示不给", 0, 20),
    # 待办
    "TODO_DEDUP_SIMILARITY": RuntimeConfigSpec("待办去重粗筛：与新待办余弦相似度≥此值的已有任务作为候选，再由System One判断是否同一件事", 0, 1),
    "TODO_DEDUP_WINDOW_HOURS": RuntimeConfigSpec("待办去重：完成或取消的任务在这段时间（小时）内仍参与去重", 0, 720),
}

assert set(RUNTIME_CONFIGS) <= set(Settings.model_fields), \
    f"RUNTIME_CONFIGS 中有 settings 未定义的项: {set(RUNTIME_CONFIGS) - set(Settings.model_fields)}"


def _type_name(name: str) -> str:
    return Settings.model_fields[name].annotation.__name__


def list_runtime_configs() -> List[Dict[str, Any]]:
    """返回所有可调配置的名字、当前值、默认值、类型、描述和取值范围"""
    result = []
    for name, spec in RUNTIME_CONFIGS.items():
        result.append({
            "name": name,
            "value": getattr(settings, name),
            "default": Settings.model_fields[name].default,
            "type": _type_name(name),
            "description": spec.description,
            "min": spec.min,
            "max": spec.max,
        })
    return result


def _coerce(name: str, raw: Any) -> Any:
    """按 settings 字段类型做严格校验：不接受字符串数字，bool 和数值不混用；float 项允许传整数。不合法返回 None"""
    annotation = Settings.model_fields[name].annotation
    if annotation is bool:
        return raw if isinstance(raw, bool) else None
    if isinstance(raw, bool):
        return None
    if annotation is int:
        return raw if isinstance(raw, int) else None
    if annotation is float:
        return float(raw) if isinstance(raw, (int, float)) else None
    return None


class RuntimeConfigError(ValueError):
    def __init__(self, errors: Dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


def update_runtime_configs(updates: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    校验并更新配置：全部通过才写入，任意一项不合法则一项都不改

    Returns:
        {name: {"old": 旧值, "new": 新值}}，只包含值确实变化的项
    """
    errors: Dict[str, str] = {}
    validated: Dict[str, Any] = {}

    for name, raw in updates.items():
        spec = RUNTIME_CONFIGS.get(name)
        if spec is None:
            errors[name] = "不是可运行时修改的配置项"
            continue
        value = _coerce(name, raw)
        if value is None:
            errors[name] = f"类型应为 {_type_name(name)}，收到 {raw!r}"
            continue
        if (spec.min is not None and value < spec.min) or (spec.max is not None and value > spec.max):
            errors[name] = f"取值范围 [{spec.min}, {spec.max}]，收到 {value}"
            continue
        validated[name] = value

    # 跨项约束：压缩条数不能超过触发压缩的长度
    if not errors:
        max_len = validated.get("PFC_STM_MAX_LEN", settings.PFC_STM_MAX_LEN)
        count = validated.get("PFC_STM_COMPRESS_COUNT", settings.PFC_STM_COMPRESS_COUNT)
        if count > max_len:
            errors["PFC_STM_COMPRESS_COUNT"] = f"不能大于 PFC_STM_MAX_LEN（{max_len}），收到 {count}"

    if errors:
        raise RuntimeConfigError(errors)

    changed = {name: {"old": getattr(settings, name), "new": value}
               for name, value in validated.items() if getattr(settings, name) != value}
    if not changed:
        return {}

    # 先写文件再改内存：写文件失败时内存不变，避免"内存已改、重启后又回退"的不一致
    _write_env({name: c["new"] for name, c in changed.items()})
    for name, c in changed.items():
        setattr(settings, name, c["new"])
    logger.warning(f"Runtime config updated (memory + {ENV_FILE}): {changed}")
    return changed


# 与 Settings 读取的是同一个 .env（路径由 _PROJECT_ROOT 固定为项目根目录，不随工作目录漂移）
ENV_FILE = _PROJECT_ROOT / ".env"
_env_lock = threading.Lock()


def _format_env_value(value: Any) -> str:
    return str(value)  # bool -> True/False，float -> 0.45，pydantic 读取时都能解析


def _write_env(values: Dict[str, Any]) -> None:
    """
    把 values 写回 .env：已有的 KEY= 行原地替换（保留其他行和注释），没有的追加到末尾。
    先写临时文件再 os.replace 原子替换，进程中途崩溃也不会留下写了一半的 .env。
    """
    with _env_lock:
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
        pending = dict(values)
        for i, line in enumerate(lines):
            m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            if m and m.group(1) in pending:
                lines[i] = f"{m.group(1)}={_format_env_value(pending.pop(m.group(1)))}"
        lines += [f"{k}={_format_env_value(v)}" for k, v in pending.items()]

        tmp = ENV_FILE.with_name(ENV_FILE.name + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if ENV_FILE.exists():
            os.chmod(tmp, ENV_FILE.stat().st_mode)  # 保留原文件权限
        os.replace(tmp, ENV_FILE)
