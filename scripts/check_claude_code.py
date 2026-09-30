"""Claude Code 経由のバックエンドを実機で確認する（CIでは実行しない。利用枠を少し使う）。

  uv run python -m scripts.check_claude_code

3段階のモデルIDがサブスクリプションで使えること、キャッシュ層を通すと2回目がLLMを呼ばないことを確かめる。
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from llm.cache import CachedBackend
from llm.factory import create_backend
from llm.settings import LLMSettings
from llm.types import LLMBackendError, LLMRequest, Message, Tier

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("check_claude_code")

TIERS: tuple[Tier, ...] = ("fast", "standard", "strong")


async def main() -> None:
    with tempfile.TemporaryDirectory() as cache_dir:
        settings = LLMSettings(
            llm_backend="claude_code", llm_cache_enabled=False, llm_cache_dir=Path(cache_dir)
        )
        raw = create_backend(settings)
        cached = CachedBackend(raw, Path(cache_dir))

        for tier in TIERS:
            request = LLMRequest(
                system="指示に従い、余計な説明なしで答えてください。",
                messages=[Message(role="user", content="「了解」とだけ答えてください。")],
                tier=tier,
                max_tokens=64,
            )
            try:
                res = await cached.complete(request)
            except LLMBackendError as e:
                log.info("[%s] %s → 失敗: %s", tier, raw.model_for(tier), e)
                continue
            log.info(
                "[%s] 設定=%s 実際=%s text=%r usage=%s cached=%s",
                tier,
                raw.model_for(tier),
                res.model,
                res.text,
                res.usage,
                res.cached,
            )

        log.info("== 2回目（キャッシュから返るはず）")
        again = await cached.complete(
            LLMRequest(
                system="指示に従い、余計な説明なしで答えてください。",
                messages=[Message(role="user", content="「了解」とだけ答えてください。")],
                tier="fast",
                max_tokens=64,
            )
        )
        log.info("cached=%s text=%r", again.cached, again.text)


if __name__ == "__main__":
    asyncio.run(main())
