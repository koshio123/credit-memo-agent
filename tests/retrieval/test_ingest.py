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


def test_文書の情報だけ変わっても_保存し直す(store: ChunkStore) -> None:
    e = CountingEmbedder()
    ingest_document(store, DOC, PAGES, [e])

    renamed = DocumentRecord("D1", "9999", "改名後", "2026-03-31", 2)
    ingest_document(store, renamed, PAGES, [e])

    assert store.get_document("D1") == renamed
    assert store.count_embeddings("hash", "D1") == 2  # 埋め込みは作り直されている


def test_本文が空でも_文書の行は作る(store: ChunkStore) -> None:
    ingest_document(store, DOC, [_page("", 1)], [CountingEmbedder()])
    assert store.get_document("D1") == DOC


def test_内容が変わる文書を_一部のモデルだけで取り込もうとすると_何も消さずに断る(
    store: ChunkStore,
) -> None:
    # 置き換えると、渡していないモデルの埋め込みも連鎖削除で消え、黙って検索が壊れる
    a, b = CountingEmbedder("model-a"), CountingEmbedder("model-b")
    ingest_document(store, DOC, PAGES, [a, b])
    changed = [PAGES[0], _page("4 【従業員の状況】\n従業員の平均年齢は四十五歳です。", 2)]

    with pytest.raises(ValueError, match="model-b"):
        ingest_document(store, DOC, changed, [a])

    assert "四十歳" in " ".join(c.text for c in store.get_chunks("D1"))  # 元のまま
    assert store.count_embeddings("model-b", "D1") == 2


def test_内容が同じなら_一部のモデルだけでも取り込める(store: ChunkStore) -> None:
    a, b = CountingEmbedder("model-a"), CountingEmbedder("model-b")
    ingest_document(store, DOC, PAGES, [a, b])
    ingest_document(store, DOC, PAGES, [a])
    assert store.count_embeddings("model-b", "D1") == 2


class WrongDimEmbedder(CountingEmbedder):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 3 for _ in texts]  # 宣言（16次元）と違う


def test_ベクトルの次元が宣言と違えば_保存せずに断る(store: ChunkStore) -> None:
    with pytest.raises(ValueError, match="次元"):
        ingest_document(store, DOC, PAGES, [WrongDimEmbedder()])
    assert store.count_embeddings("hash", "D1") == 0


def test_取り込み済みのモデルは_トークンの数え直しもしない(store: ChunkStore) -> None:
    class Spy(CountingEmbedder):
        over_calls = 0

        def over_limit(self, documents: Sequence[str]) -> list[bool]:
            Spy.over_calls += 1
            return super().over_limit(documents)

    e = Spy()
    ingest_document(store, DOC, PAGES, [e])
    result = ingest_document(store, DOC, PAGES, [e])

    assert Spy.over_calls == 1
    assert result.over_limit == {}  # 今回埋め込んだモデルだけ報告する
