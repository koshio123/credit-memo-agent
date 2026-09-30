"""実データへのつなぎ方のテスト。起動時に DB やモデルを要求しないこと。"""

from collections.abc import Sequence

import pytest

from edinet_mcp.service import EdinetMcpError
from edinet_mcp.wiring import LazySearcher
from retrieval.store import Hit


class Recorder:
    def is_indexed(self, doc_id: str) -> bool:
        return True

    def search(self, query: str, k: int, *, doc_ids: Sequence[str] | None = None) -> list[Hit]:
        return []


def test_最初の検索まで_検索器を作らない() -> None:
    made: list[int] = []

    def factory() -> Recorder:
        made.append(1)
        return Recorder()

    lazy = LazySearcher(factory)
    assert made == []  # 起動時には作らない（DB が無くても財務数値などは使える）
    lazy.search("x", 1)
    lazy.search("y", 1)
    assert made == [1]  # 作るのは1回だけ


def test_作れなければ_ツールのエラーにする_次の呼び出しでもう一度試す() -> None:
    attempts: list[int] = []

    def factory() -> Recorder:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("接続できません")
        return Recorder()

    lazy = LazySearcher(factory)
    with pytest.raises(EdinetMcpError, match="検索"):
        lazy.search("x", 1)
    assert lazy.search("x", 1) == []  # DB を起動した後は、そのまま使える
