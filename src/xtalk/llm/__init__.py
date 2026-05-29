from .dummy import DummyChatModel
from langchain_openai import ChatOpenAI

__all__ = ["DummyChatModel", "ChatOpenAI"]

try:
    from .qwen_local import LocalQwenChatModel

    __all__.append("LocalQwenChatModel")
except Exception:
    pass
