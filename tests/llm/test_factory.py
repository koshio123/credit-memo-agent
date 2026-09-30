from pathlib import Path

import pytest

from llm.cache import CachedBackend
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


@pytest.mark.parametrize("name", ["claude_code", "anthropic_api"])
def test_未実装のバックエンドは分かるエラーにする(tmp_path: Path, name: str) -> None:
    with pytest.raises(NotImplementedError, match=name):
        create_backend(_settings(tmp_path, llm_backend=name))
