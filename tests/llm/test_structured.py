"""LLM の出力を JSON で受け、Pydantic で検証する。失敗したら理由を添えて 1 回だけ作り直させる。"""

import pytest
from pydantic import BaseModel

from llm.fake import ScriptedBackend
from llm.structured import (
    StructuredOutputError,
    complete_structured,
    extract_json,
    json_schema_hint,
)
from llm.types import LLMBackendError, LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.anyio


class Answer(BaseModel):
    title: str
    score: int


def _request(text: str = "質問") -> LLMRequest:
    return LLMRequest(system="s", messages=[Message(role="user", content=text)])


# ---- JSON の取り出し ----


def test_そのままのJSON() -> None:
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_コードフェンスで囲まれたJSON() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_前後に説明文があっても_最初のJSONを取り出す() -> None:
    assert extract_json('結果は次のとおりです。\n{"a": {"b": [1, 2]}}\n以上です。') == (
        '{"a": {"b": [1, 2]}}'
    )


def test_JSONが無ければエラー() -> None:
    with pytest.raises(ValueError):
        extract_json("JSONはありません")


# ---- 検証と作り直し ----


async def test_正しいJSONはそのままモデルになる() -> None:
    backend = ScriptedBackend(['{"title": "a", "score": 3}'])
    result = await complete_structured(backend, _request(), Answer)
    assert result == Answer(title="a", score=3)
    assert len(backend.requests) == 1


async def test_検証に失敗したら_理由を添えて1回だけ作り直す() -> None:
    backend = ScriptedBackend(['{"title": "a"}', '{"title": "a", "score": 3}'])
    result = await complete_structured(backend, _request("元の質問"), Answer)

    assert result.score == 3
    assert len(backend.requests) == 2
    retry = backend.requests[1].messages[0].content
    assert "元の質問" in retry  # 元の依頼を保つ（履歴は 1 通のまま）
    assert "score" in retry  # 何が足りなかったかを伝える
    assert len(backend.requests[1].messages) == 1


async def test_JSONでない出力も_作り直す() -> None:
    backend = ScriptedBackend(["了解しました", '{"title": "a", "score": 1}'])
    assert (await complete_structured(backend, _request(), Answer)).title == "a"


async def test_作り直しても失敗したら_エラーにする() -> None:
    backend = ScriptedBackend(['{"title": 1}', "{}"])
    with pytest.raises(StructuredOutputError) as e:
        await complete_structured(backend, _request(), Answer)
    assert len(backend.requests) == 2
    assert "title" in str(e.value)


async def test_作り直しの回数は指定できる() -> None:
    backend = ScriptedBackend(["x", "y", '{"title": "a", "score": 1}'])
    result = await complete_structured(backend, _request(), Answer, retries=2)
    assert result.score == 1


async def test_途中で切れた出力は_作り直さずにエラー() -> None:
    cut = LLMResponse(text='{"title": "a", "sc', backend="b", model="m", truncated=True)
    backend = ScriptedBackend([cut])
    with pytest.raises(StructuredOutputError, match="途中で切れ"):
        await complete_structured(backend, _request(), Answer)
    assert len(backend.requests) == 1


async def test_バックエンドの失敗は_そのまま伝える() -> None:
    backend = ScriptedBackend([LLMBackendError("落ちた")])
    with pytest.raises(LLMBackendError):
        await complete_structured(backend, _request(), Answer)


def test_スキーマの説明を作れる() -> None:
    hint = json_schema_hint(Answer)
    assert "title" in hint
    assert "score" in hint
