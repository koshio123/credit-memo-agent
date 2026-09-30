from pathlib import Path

import pytest

from llm.cache import CachedBackend
from llm.claude_code import ClaudeCodeBackend
from llm.factory import create_backend
from llm.local import OllamaBackend
from llm.settings import LLMSettings


def _settings(tmp_path: Path, **kwargs: object) -> LLMSettings:
    # .env を読まず、引数だけで設定を作る
    return LLMSettings(_env_file=None, llm_cache_dir=tmp_path, **kwargs)  # type: ignore[arg-type]


def test_localはキャッシュ付きOllamaを返す(tmp_path: Path) -> None:
    backend = create_backend(_settings(tmp_path, llm_backend="local"))

    assert isinstance(backend, CachedBackend)
    assert isinstance(backend.inner, OllamaBackend)


def test_キャッシュを切れる(tmp_path: Path) -> None:
    backend = create_backend(_settings(tmp_path, llm_backend="local", llm_cache_enabled=False))

    assert isinstance(backend, OllamaBackend)


def test_未実装のバックエンドは分かるエラーにする(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="anthropic_api"):
        create_backend(_settings(tmp_path, llm_backend="anthropic_api"))


def test_claude_codeはキャッシュ付きで作られる(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    backend = create_backend(_settings(tmp_path, llm_backend="claude_code"))

    assert isinstance(backend, CachedBackend)
    assert isinstance(backend.inner, ClaudeCodeBackend)
    assert backend.model_for("strong") == "claude-opus-5-5"


def test_num_ctxとthinkが設定から渡る(tmp_path: Path) -> None:
    backend = create_backend(
        _settings(
            tmp_path,
            llm_backend="local",
            llm_cache_enabled=False,
            local_num_ctx=32768,
            local_think=True,
        )
    )

    assert isinstance(backend, OllamaBackend)
    assert (backend.num_ctx, backend.think) == (32768, True)
