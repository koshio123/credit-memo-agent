import pytest

from llm.fake import ScriptedBackend
from llm.types import LLMRequest, LLMResponse, Message, Usage
from llm.usage import CountingBackend

pytestmark = pytest.mark.anyio


def _request() -> LLMRequest:
    return LLMRequest(messages=[Message(role="user", content="q")])


async def test_呼び出し回数とトークン数を数える() -> None:
    fresh = LLMResponse(
        text="a", backend="b", model="m", usage=Usage(input_tokens=10, output_tokens=2)
    )
    backend = CountingBackend(ScriptedBackend([fresh, fresh]))
    await backend.complete(_request())
    await backend.complete(_request())
    assert (backend.calls, backend.cached_calls) == (2, 0)
    assert (backend.input_tokens, backend.output_tokens) == (20, 4)


async def test_キャッシュから返った呼び出しは_トークンに数えない() -> None:
    hit = LLMResponse(
        text="a", backend="b", model="m", cached=True, usage=Usage(input_tokens=10, output_tokens=2)
    )
    backend = CountingBackend(ScriptedBackend([hit]))
    await backend.complete(_request())
    assert (backend.calls, backend.cached_calls) == (1, 1)
    assert (backend.input_tokens, backend.output_tokens) == (0, 0)


def test_バックエンドの情報はそのまま渡す() -> None:
    inner = ScriptedBackend([], models={"fast": "x"}, salt="s")
    wrapped = CountingBackend(inner)
    assert (wrapped.name, wrapped.cache_salt, wrapped.model_for("fast")) == ("scripted", "s", "x")
