"""メモ生成コマンドの、失敗時の振る舞い。LLM は呼ばない（呼ぶ前に止まる場合だけ）。"""

import logging
import sys

import pytest

from scripts.generate_memo import main

pytestmark = pytest.mark.anyio


def _run(monkeypatch: pytest.MonkeyPatch, *args: str, backend: str = "local") -> None:
    monkeypatch.setattr(sys, "argv", ["generate_memo", *args])
    monkeypatch.setenv("LLM_BACKEND", backend)
    monkeypatch.setenv("LLM_CACHE_ENABLED", "false")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


async def test_対象外の証券コードは_理由だけを出して終わる(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _run(monkeypatch, "--sec-code", "0000")
    with caplog.at_level(logging.ERROR):
        code = await main()
    assert code == 1
    assert "対象外" in caplog.text
    assert "Traceback" not in caplog.text


async def test_1社が失敗しても_残りの会社は処理を続ける(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _run(monkeypatch, "--sec-code", "0000", "0001")
    with caplog.at_level(logging.ERROR):
        code = await main()
    assert code == 1
    assert "0000" in caplog.text and "0001" in caplog.text  # 両方を試した


async def test_claude_codeは_一度に3社までで拒否する(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _run(monkeypatch, "--sec-code", "6744", "1969", "8892", "9024", backend="claude_code")
    with caplog.at_level(logging.ERROR):
        code = await main()
    assert code == 2
    assert "3 社まで" in caplog.text


async def test_APIキーが環境にあれば_LLMを呼ぶ前に作成を拒否する(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _run(monkeypatch, "--sec-code", "6744", backend="claude_code")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    with caplog.at_level(logging.ERROR):
        code = await main()
    assert code == 1
    assert "API 課金" in caplog.text
    assert "dummy" not in caplog.text  # キーの値は出さない
