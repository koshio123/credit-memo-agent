"""Ollama の実機との接続を確認する（CIでは実行しない）。

  uv run python scripts/check_ollama.py

llm/ の実装が前提にしている挙動（think の受理、done_reason、切り捨て検知など）を確かめる。
"""

import asyncio
import logging

from llm.factory import create_backend
from llm.settings import LLMSettings
from llm.types import LLMBackendError, LLMRequest, Message

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("check_ollama")


async def main() -> None:
    settings = LLMSettings(llm_backend="local", llm_cache_enabled=False)
    backend = create_backend(settings)

    log.info("== 1. 日本語の基本応答（fast=%s）", backend.model_for("fast"))
    res = await backend.complete(
        LLMRequest(
            system="あなたは与信担当者です。簡潔に日本語で答えてください。",
            messages=[Message(role="user", content="自己資本比率とは何ですか。一文で。")],
            tier="fast",
            max_tokens=200,
        )
    )
    log.info("text=%r", res.text)
    log.info("model=%s usage=%s truncated=%s", res.model, res.usage, res.truncated)
    log.info("<think> が本文に混じっていない: %s", "<think>" not in res.text)

    log.info("== 2. max_tokens で打ち切ったときに truncated が立つか")
    res = await backend.complete(
        LLMRequest(
            messages=[
                Message(role="user", content="有価証券報告書について詳しく説明してください。")
            ],
            tier="fast",
            max_tokens=8,
        )
    )
    log.info("text=%r truncated=%s", res.text, res.truncated)

    log.info("== 3. 長い入力が num_ctx を超えたときにエラーになるか（num_ctx=512）")
    small = create_backend(
        LLMSettings(llm_backend="local", llm_cache_enabled=False, local_num_ctx=512)
    )
    long_text = "売上高は前期比で増加した。" * 400
    try:
        res = await small.complete(
            LLMRequest(
                messages=[Message(role="user", content=long_text + "\n以上を一文で要約せよ。")],
                tier="fast",
                max_tokens=50,
            )
        )
        log.info("エラーにならなかった: usage=%s truncated=%s", res.usage, res.truncated)
    except LLMBackendError as e:
        log.info("期待どおりエラー: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
