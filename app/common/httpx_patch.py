"""
让httpx默认 trust_env=False（应用启动时调用一次）

macOS上httpx不读取系统代理的ExceptionsList，配置了系统代理时连localhost（Qdrant）会超时。
mem0、openai等第三方库内部自己创建httpx客户端，只能通过patch统一处理。
"""
from typing import Any

import httpx

_original_client_init = httpx.Client.__init__
_original_async_client_init = httpx.AsyncClient.__init__


def _patched_client_init(self, *args: Any, trust_env: bool = False, **kwargs: Any) -> None:
    _original_client_init(self, *args, trust_env=trust_env, **kwargs)


def _patched_async_client_init(self, *args: Any, trust_env: bool = False, **kwargs: Any) -> None:
    _original_async_client_init(self, *args, trust_env=trust_env, **kwargs)


def apply_httpx_patch() -> None:
    httpx.Client.__init__ = _patched_client_init
    httpx.AsyncClient.__init__ = _patched_async_client_init


# 自动应用patch（导入时执行）
apply_httpx_patch()
