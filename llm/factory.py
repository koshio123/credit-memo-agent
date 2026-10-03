from llm.cache import CachedBackend
from llm.claude_code import ClaudeCodeBackend
from llm.local import OllamaBackend
from llm.settings import LLMSettings
from llm.types import LLMBackend


def create_backend(settings: LLMSettings | None = None) -> LLMBackend:
    """設定に従ってバックエンドを作る。キャッシュが有効なら包んで返す。

    Args:
        settings: LLM の設定。None なら環境変数から読む。

    Returns:
        LLM のバックエンド。

    Raises:
        NotImplementedError: 未実装のバックエンドが選ばれたとき。
    """
    settings = settings or LLMSettings()

    backend: LLMBackend
    match settings.llm_backend:
        case "local":
            backend = OllamaBackend(
                base_url=settings.ollama_base_url,
                models={
                    "fast": settings.local_model_fast,
                    "standard": settings.local_model_standard,
                    "strong": settings.local_model_strong,
                },
                num_ctx=settings.local_num_ctx,
                think=settings.local_think,
            )
        case "claude_code":
            backend = ClaudeCodeBackend(
                models={
                    "fast": settings.claude_model_fast,
                    "standard": settings.claude_model_standard,
                    "strong": settings.claude_model_strong,
                },
                think=settings.claude_think,
            )
        case name:
            raise NotImplementedError(f"LLMバックエンド {name!r} は未実装です")

    if settings.llm_cache_enabled:
        return CachedBackend(backend, settings.llm_cache_dir)
    return backend
