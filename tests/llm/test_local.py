import json

import httpx
import pytest

from llm.local import (
    PER_MESSAGE_OVERHEAD_TOKENS,
    PROMPT_OVERHEAD_TOKENS,
    TOKENS_PER_CHAR_ESTIMATE,
    OllamaBackend,
)
from llm.types import LLMBackendError, LLMRequest, Message

pytestmark = pytest.mark.anyio


def _backend(handler: httpx.MockTransport) -> OllamaBackend:
    return OllamaBackend(
        base_url="http://ollama.test",
        models={"fast": "small", "standard": "mid", "strong": "large"},
        num_ctx=10_000,
        client=httpx.AsyncClient(transport=handler),
    )


async def test_リクエストを組み立てて応答を読む() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "large",
                "message": {"role": "assistant", "content": "自己資本比率は45.3%です。"},
                "prompt_eval_count": 12,
                "eval_count": 7,
            },
        )

    backend = _backend(httpx.MockTransport(handler))
    res = await backend.complete(
        LLMRequest(
            system="あなたは与信担当です。",
            messages=[Message(role="user", content="要約して")],
            tier="strong",
            max_tokens=256,
            temperature=0.2,
        )
    )

    assert seen["url"] == "http://ollama.test/api/chat"
    assert seen["body"] == {
        "model": "large",
        "messages": [
            {"role": "system", "content": "あなたは与信担当です。"},
            {"role": "user", "content": "要約して"},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 256, "num_ctx": 10_000},
    }
    assert res.text == "自己資本比率は45.3%です。"
    assert (res.backend, res.model) == ("local", "large")
    assert (res.usage.input_tokens, res.usage.output_tokens) == (12, 7)
    assert res.cached is False


async def test_systemが空ならsystemメッセージを付けない() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    backend = _backend(httpx.MockTransport(handler))
    await backend.complete(LLMRequest(messages=[Message(role="user", content="やあ")]))

    assert [m["role"] for m in bodies[0]["messages"]] == ["user"]  # type: ignore[index]


async def test_HTTPエラーはLLMBackendErrorにする() -> None:
    backend = _backend(httpx.MockTransport(lambda _: httpx.Response(404, text="model not found")))

    with pytest.raises(LLMBackendError, match="404"):
        await backend.complete(LLMRequest(messages=[Message(role="user", content="x")]))


async def test_接続できないときもLLMBackendErrorにする() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    backend = _backend(httpx.MockTransport(handler))

    with pytest.raises(LLMBackendError, match="Ollama"):
        await backend.complete(LLMRequest(messages=[Message(role="user", content="x")]))


async def test_想定外の形式の応答もLLMBackendErrorにする() -> None:
    backend = _backend(httpx.MockTransport(lambda _: httpx.Response(200, json={"unexpected": 1})))

    with pytest.raises(LLMBackendError, match="解釈できません"):
        await backend.complete(LLMRequest(messages=[Message(role="user", content="x")]))


@pytest.mark.parametrize(
    ("done_reason", "truncated"), [("stop", False), ("length", True), (None, False)]
)
async def test_打ち切られた応答にはtruncatedを立てる(
    done_reason: str | None, truncated: bool
) -> None:
    payload: dict[str, object] = {"message": {"role": "assistant", "content": "売上高は"}}
    if done_reason is not None:
        payload["done_reason"] = done_reason
    backend = _backend(httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))

    res = await backend.complete(LLMRequest(messages=[Message(role="user", content="x")]))

    assert res.truncated is truncated


async def test_thinkの設定をそのまま送る() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    backend = OllamaBackend(
        base_url="http://ollama.test",
        models={"fast": "s", "standard": "m", "strong": "l"},
        think=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await backend.complete(LLMRequest(messages=[Message(role="user", content="x")]))

    assert bodies[0]["think"] is True


def test_cache_saltにnum_ctxとthinkが入る() -> None:
    def make(num_ctx: int, think: bool) -> OllamaBackend:
        return OllamaBackend(
            base_url="http://ollama.test",
            models={"fast": "s", "standard": "m", "strong": "l"},
            num_ctx=num_ctx,
            think=think,
        )

    salts = {
        make(1000, False).cache_salt,
        make(2000, False).cache_salt,
        make(1000, True).cache_salt,
    }

    assert len(salts) == 3


def _sized_backend(num_ctx: int, calls: list[int]) -> OllamaBackend:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    return OllamaBackend(
        base_url="http://ollama.test",
        models={"fast": "s", "standard": "m", "strong": "l"},
        num_ctx=num_ctx,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _request_of(chars: int, max_tokens: int) -> LLMRequest:
    return LLMRequest(
        messages=[Message(role="user", content="あ" * chars)], tier="fast", max_tokens=max_tokens
    )


async def test_num_ctxに収まらない入力は送信せずエラーにする() -> None:
    # Ollama は収まらない入力を黙って切り詰め、応答の prompt_eval_count も切り詰め後の値を返す。
    # 実機で確認済み（4426トークンの入力が258トークンとして「成功」した）ので、送る前に止める
    calls: list[int] = []
    backend = _sized_backend(num_ctx=1000, calls=calls)

    with pytest.raises(LLMBackendError, match="num_ctx"):
        await backend.complete(_request_of(chars=2000, max_tokens=100))

    assert calls == []


async def test_収まる限界ちょうどの入力は送り_1文字超えたら止める() -> None:
    num_ctx, max_tokens = 1000, 100
    overhead = PROMPT_OVERHEAD_TOKENS + PER_MESSAGE_OVERHEAD_TOKENS * 1  # user 1通
    limit_chars = int((num_ctx - max_tokens - overhead) / TOKENS_PER_CHAR_ESTIMATE)
    calls: list[int] = []
    backend = _sized_backend(num_ctx, calls)

    res = await backend.complete(_request_of(limit_chars, max_tokens))
    assert res.text == "ok"

    with pytest.raises(LLMBackendError, match="num_ctx"):
        await backend.complete(_request_of(limit_chars + 1, max_tokens))
    assert len(calls) == 1


async def test_見積もりは実測の最大値に余裕を持たせている() -> None:
    # 実測（Qwen3）の最大は 有報風の文章 0.95 トークン/文字。珍しい漢字や記号に備えて余裕を持つ
    assert TOKENS_PER_CHAR_ESTIMATE >= 0.95 * 1.25


async def test_短い発話が多い履歴はメッセージごとの定型分を見積もりに含める() -> None:
    calls: list[int] = []
    backend = _sized_backend(num_ctx=1000, calls=calls)
    request = LLMRequest(
        messages=[Message(role="user", content="a") for _ in range(100)],
        tier="fast",
        max_tokens=100,
    )  # 文字数は100しかないが、100通分の定型分で収まらない

    with pytest.raises(LLMBackendError, match="num_ctx"):
        await backend.complete(request)
    assert calls == []


async def test_systemも見積もりに含める() -> None:
    calls: list[int] = []
    backend = _sized_backend(num_ctx=1000, calls=calls)
    request = LLMRequest(
        system="い" * 1500, messages=[Message(role="user", content="x")], tier="fast", max_tokens=10
    )

    with pytest.raises(LLMBackendError, match="num_ctx"):
        await backend.complete(request)
    assert calls == []
