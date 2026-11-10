"""
自定义豆包LLM实现，支持关闭深度思考
"""
from typing import List, Dict, Optional

from mem0.configs.llms.openai import OpenAIConfig
from mem0.llms.openai import OpenAILLM


class DoubaoLLM(OpenAILLM):
    """
    豆包LLM的自定义实现，自动在所有API调用中添加 thinking: {type: disabled}
    """

    def __init__(self, config: Optional[OpenAIConfig] = None):
        super().__init__(config)
        self.disable_thinking = getattr(config, 'disable_thinking', True)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        覆写generate_response，自动添加thinking参数
        """
        # 如果配置了disable_thinking，自动添加extra_body参数
        if self.disable_thinking and 'extra_body' not in kwargs:
            kwargs['extra_body'] = {"thinking": {"type": "disabled"}}

        # 调用父类方法
        return super().generate_response(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )
