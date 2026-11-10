"""Doubao LLM客户端（使用Ark API，OpenAI-compatible）"""
import contextvars
import logging
from typing import Dict, List, Any, Optional, Union

import httpx
from arkruntime import AsyncArk

from app.config.settings import settings

logger = logging.getLogger(__name__)

# 当前请求内的 LLM token 用量收集器：调用方 start_usage_collection() 后，
# 同一上下文（含其派生的 asyncio 任务）里的 chat 调用都会把用量追加进来
_usage_collector: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar(
    "llm_usage_collector", default=None
)


def start_usage_collection() -> List[Dict[str, Any]]:
    """开始收集当前上下文的 LLM 用量，返回收集用的列表"""
    usage: List[Dict[str, Any]] = []
    _usage_collector.set(usage)
    return usage


class DoubaoClient:
    """
    必须使用AsyncArk：同步的Ark客户端会阻塞事件循环，
    一次查询改写(~2s)或LLM重排(~0.9s)期间整个服务无法处理任何其它请求。
    """

    def __init__(self):
        self.client = AsyncArk(
            base_url=settings.ARK_BASE_URL,
            api_key=settings.ARK_API_KEY
        )
        # System One决策模型的HTTP客户端
        # 注意：System One需要特殊的beta header
        self.http_client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {settings.ARK_API_KEY}",
                "Content-Type": "application/json",
                "ark-beta-systemone-api": "true"  # System One必需的header
            },
            timeout=30.0
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 10000,
        model: str = settings.DOUBAO_MODEL_LITE,
    ) -> str:
        """标准的chat completion（thinking强制关闭）"""
        completion = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            thinking={"type": "disabled"},
            max_tokens=max_tokens
        )
        usage = getattr(completion, "usage", None)
        if usage:
            logger.info(f"[LLMUsage] model={model} prompt_tokens={usage.prompt_tokens} "
                        f"completion_tokens={usage.completion_tokens}")
            collector = _usage_collector.get()
            if collector is not None:
                collector.append({"model": model, "prompt_tokens": usage.prompt_tokens,
                                  "completion_tokens": usage.completion_tokens})
        return completion.choices[0].message.content if completion.choices else ""

    async def generate(self, prompt: str, max_tokens: int = 10000) -> str:
        """简化版：只传prompt"""
        return await self.chat([{"role": "system", "content": prompt}], max_tokens=max_tokens)

    async def decide(
        self,
        state: str,
        questions: Dict[str, Dict[str, Any]],
        model: str = None
    ) -> Dict[str, Any]:
        """
        System One决策模型调用

        Args:
            state: 上下文文本（类似Ollama的state）
            questions: 问题配置字典，格式：
                {
                    "question_key": {
                        "type": "choice" | "noul" | "score",
                        "instructions": "问题描述",
                        "criteria": {"option1": "描述1", ...}  # choice/score需要
                    }
                }
            model: 模型名称，默认使用DOUBAO_MODEL_LITE

        Returns:
            {
                "answers": {
                    "question_key": {
                        "choice": "selected_option",  # choice类型
                        "noul": 0.85,                 # noul类型（概率）
                        "score": 7                    # score类型
                    }
                }
            }

        Example:
            >>> result = await client.decide(
            ...     state="用户说：我要退货",
            ...     questions={
            ...         "intent": {
            ...             "type": "choice",
            ...             "instructions": "用户的意图是什么？",
            ...             "criteria": {
            ...                 "return": "退货",
            ...                 "exchange": "换货",
            ...                 "inquiry": "咨询"
            ...             }
            ...         }
            ...     }
            ... )
            >>> print(result["answers"]["intent"]["choice"])  # "return"
        """
        try:
            model = model or settings.DOUBAO_MODEL_LITE

            payload = {
                "model": model,
                "state": state,
                "questions": questions
            }

            # 调用System One API
            response = await self.http_client.post(
                settings.ARK_SYSTEM_ONE_URL,
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            logger.debug(f"System One response: {result}")
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling System One: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to call System One: {e}", exc_info=True)
            raise


doubao_client = DoubaoClient()
