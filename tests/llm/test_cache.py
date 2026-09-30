from pathlib import Path

import pytest

from llm.cache import CachedBackend
from llm.fake import ScriptedBackend
from llm.types import LLMBackendError, LLMRequest, Message

pytestmark = pytest.mark.anyio


def _req(text: str = "こんにちは") -> LLMRequest:
    return LLMRequest(messages=[Message(role="user", content=text)])


async def test_同じ入力ならLLMを呼ばずキャッシュから返す(tmp_path: Path) -> None:
    inner = ScriptedBackend(["一回目の応答"])
    backend = CachedBackend(inner, tmp_path)

    first = await backend.complete(_req())
    second = await backend.complete(_req())

    assert first.text == second.text == "一回目の応答"
    assert first.cached is False
    assert second.cached is True
    assert len(inner.requests) == 1


async def test_入力が違えば別のキャッシュになる(tmp_path: Path) -> None:
    inner = ScriptedBackend(["A", "B"])
    backend = CachedBackend(inner, tmp_path)

    a = await backend.complete(_req("質問A"))
    b = await backend.complete(_req("質問B"))

    assert (a.text, b.text) == ("A", "B")
    assert len(inner.requests) == 2


async def test_モデル割当が変わればキャッシュを使い回さない(tmp_path: Path) -> None:
    old = CachedBackend(ScriptedBackend(["旧モデル"], models={"standard": "model-a"}), tmp_path)
    new_inner = ScriptedBackend(["新モデル"], models={"standard": "model-b"})
    new = CachedBackend(new_inner, tmp_path)

    await old.complete(_req())
    res = await new.complete(_req())

    assert res.text == "新モデル"
    assert res.cached is False


async def test_失敗した呼び出しはキャッシュしない(tmp_path: Path) -> None:
    inner = ScriptedBackend([LLMBackendError("一時的な失敗"), "リトライ成功"])
    backend = CachedBackend(inner, tmp_path)

    with pytest.raises(LLMBackendError):
        await backend.complete(_req())
    res = await backend.complete(_req())

    assert res.text == "リトライ成功"
    assert res.cached is False


async def test_壊れたキャッシュファイルは読み飛ばして上書きする(tmp_path: Path) -> None:
    inner = ScriptedBackend(["正常な応答"])
    backend = CachedBackend(inner, tmp_path)
    await backend.complete(_req())
    (cache_file,) = tmp_path.glob("*.json")
    cache_file.write_text("{壊れたJSON", encoding="utf-8")

    inner2 = ScriptedBackend(["再生成"])
    res = await CachedBackend(inner2, tmp_path).complete(_req())

    assert res.text == "再生成"
    assert res.cached is False
    assert "再生成" in cache_file.read_text(encoding="utf-8")
