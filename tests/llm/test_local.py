import json

import httpx
import pytest

from llm.local import OllamaBackend
from llm.types import LLMBackendError, LLMRequest, Message

pytestmark = pytest.mark.anyio


def _backend(handler: httpx.MockTransport) -> OllamaBackend:
    return OllamaBackend(
        base_url="http://ollama.test",
        models={"fast": "small", "standard": "mid", "strong": "large"},
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
        "options": {"temperature": 0.2, "num_predict": 256},
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
