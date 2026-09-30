"""取り込み（チャンク分け → 保存 → 埋め込み）の統合テスト。実際の DB を使う。"""

from collections.abc import Sequence

import pytest

from retrieval.embedding import HashEmbedder
from retrieval.ingest import ingest_document
from retrieval.store import ChunkStore, DocumentRecord

pytestmark = pytest.mark.db

HEADER = "EDINET提出書類\nサンプル株式会社(E00000)\n有価証券報告書\n"


def _page(body: str, n: int) -> str:
    return f"{HEADER}{body}\n{n}/10"


PAGES = [
    _page("3 【事業等のリスク】\n原材料価格が高騰した場合、収益を圧迫する可能性があります。", 1),
    _page("4 【従業員の状況】\n従業員の平均年齢は四十歳です。", 2),
]
DOC = DocumentRecord("D1", "9999", "サンプル", "2026-03-31", 2)


class CountingEmbedder(HashEmbedder):
    """呼ばれた回数と、上限として扱う文字数を持つ、テスト用の埋め込み。"""

    def __init__(self, key: str = "hash", limit_chars: int | None = None) -> None:
        super().__init__(dim=16)
        self._key = key
        self.document_calls = 0
        self._limit = limit_chars

    @property
    def key(self) -> str:
        return self._key

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return super().embed_documents(texts)

    def over_limit(self, documents: Sequence[str]) -> list[bool]:
        return [self._limit is not None and len(d) > self._limit for d in documents]


def test_取り込むとチャンクと埋め込みが保存される(store: ChunkStore) -> None:
    e = CountingEmbedder()

    result = ingest_document(store, DOC, PAGES, [e])

    chunks = store.get_chunks("D1")
    assert result.n_chunks == len(chunks) == 2
    assert store.count_embeddings("hash", "D1") == 2
    assert result.embedded == {"hash": 2}
    assert [c.heading_path for c in chunks] == [["3 【事業等のリスク】"], ["4 【従業員の状況】"]]


def test_内容が変わっていなければ_再実行で埋め込みをやり直さない(store: ChunkStore) -> None:
    e = CountingEmbedder()
    ingest_document(store, DOC, PAGES, [e])
    assert e.document_calls == 1

    result = ingest_document(store, DOC, PAGES, [e])

    assert e.document_calls == 1  # 2回目は埋め込みを呼ばない
    assert result.embedded == {"hash": 0}
    assert store.count_embeddings("hash", "D1") == 2


def test_内容が変わったら_古い埋め込みは残さず作り直す(store: ChunkStore) -> None:
    e = CountingEmbedder()
    ingest_document(store, DOC, PAGES, [e])

    changed = [PAGES[0], _page("4 【従業員の状況】\n従業員の平均年齢は四十五歳です。", 2)]
    result = ingest_document(store, DOC, changed, [e])

    assert e.document_calls == 2
    assert result.embedded == {"hash": 2}
    assert "四十五歳" in " ".join(c.text for c in store.get_chunks("D1"))
    assert store.count_embeddings("hash", "D1") == 2


def test_新しいモデルを足すと_そのモデルの分だけ埋め込む(store: ChunkStore) -> None:
    a, b = CountingEmbedder("model-a"), CountingEmbedder("model-b")
    ingest_document(store, DOC, PAGES, [a])

    result = ingest_document(store, DOC, PAGES, [a, b])

    assert (a.document_calls, b.document_calls) == (1, 1)  # a は既にあるので呼ばない
    assert result.embedded == {"model-a": 0, "model-b": 2}
    assert store.count_embeddings("model-b", "D1") == 2


def test_入力上限を超えるチャンクの数を報告する(store: ChunkStore) -> None:
    e = CountingEmbedder(limit_chars=20)  # 短い上限。本文の長いチャンクは超える

    result = ingest_document(store, DOC, PAGES, [e])

    assert result.over_limit["hash"] >= 1
    # 超えていても取り込みは止めない（報告して、チャンクの大きさを見直す材料にする）
    assert store.count_embeddings("hash", "D1") == result.n_chunks


def test_本文が空の文書は_失敗させず0件で終える(store: ChunkStore) -> None:
    result = ingest_document(store, DOC, [_page("", 1)], [CountingEmbedder()])
    assert result.n_chunks == 0
    assert store.get_chunks("D1") == []
