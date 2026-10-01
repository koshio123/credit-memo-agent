"""検索（全文・ベクトル・融合）の統合テスト。実際の DB を使う。"""

import pytest

from retrieval.chunker import Chunk
from retrieval.embedding import HashEmbedder
from retrieval.search import LexicalIndex, Retriever
from retrieval.store import ChunkStore, DocumentRecord, Hit

pytestmark = pytest.mark.db

RISK = "原材料価格が高騰した場合、収益を圧迫する可能性があります。"
STAFF = "従業員の平均年齢は四十歳で、平均勤続年数は十五年です。"
DIV = "配当性向は三十パーセントを目安として、安定的な配当を継続します。"
DOC = DocumentRecord("D1", "9999", "サンプル", "2026-03-31", 3)


@pytest.fixture
def loaded(store: ChunkStore) -> tuple[ChunkStore, HashEmbedder]:
    e = HashEmbedder(dim=64)
    chunks = [
        Chunk(f"D1:{i}", "D1", ["第2 【事業の状況】"], i + 1, i + 1, text)
        for i, text in enumerate([STAFF, RISK, DIV])
    ]
    store.upsert_document(DOC, chunks)
    store.add_embeddings(
        "hash", [c.chunk_id for c in chunks], e.embed_documents([c.text for c in chunks])
    )
    return store, e


def ids(hits: list[Hit]) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


def test_ベクトル検索だけ(loaded: tuple[ChunkStore, HashEmbedder]) -> None:
    store, e = loaded
    r = Retriever(store, e)
    assert ids(r.search("原材料価格の高騰が収益に与える影響", k=3, mode="vector"))[0] == "D1:1"


def test_融合は_BM25とベクトルの両方で上位のものを上に出す(
    loaded: tuple[ChunkStore, HashEmbedder],
) -> None:
    store, e = loaded
    r = Retriever(store, e)
    hits = r.search("原材料価格の高騰が収益に与える影響", k=3, mode="hybrid")
    assert ids(hits)[0] == "D1:1"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_融合の順位は_1から順に並ぶ(loaded: tuple[ChunkStore, HashEmbedder]) -> None:
    store, e = loaded
    hits = Retriever(store, e).search("従業員の平均年齢", k=3, mode="hybrid")
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_kで件数を絞る(loaded: tuple[ChunkStore, HashEmbedder]) -> None:
    store, e = loaded
    assert len(Retriever(store, e).search("従業員", k=1, mode="hybrid")) == 1


def test_文書で絞り込める(loaded: tuple[ChunkStore, HashEmbedder]) -> None:
    store, e = loaded
    r = Retriever(store, e)
    assert r.search("原材料", k=3, mode="hybrid", doc_ids=["NONE"]) == []


def test_クエリは全角半角をそろえてから検索する(loaded: tuple[ChunkStore, HashEmbedder]) -> None:
    # 本文は NFKC で正規化されている。全角の数字・英字を含むクエリでも同じ結果になる
    store, e = loaded
    r = Retriever(store, e)
    assert ids(r.search("配当性向は３０％", k=3, mode="bm25")) == ids(
        r.search("配当性向は30%", k=3, mode="bm25")
    )


def test_BM25検索だけ(loaded: tuple[ChunkStore, HashEmbedder]) -> None:
    store, e = loaded
    hits = Retriever(store, e).search("原材料価格の高騰が収益に与える影響", k=3, mode="bm25")
    assert ids(hits)[0] == "D1:1"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert hits[0].chunk.text == RISK


def test_BM25検索は_文書で絞り込める(loaded: tuple[ChunkStore, HashEmbedder]) -> None:
    store, e = loaded
    r = Retriever(store, e)
    assert r.search("原材料価格", k=3, mode="bm25", doc_ids=["NONE"]) == []
    assert ids(r.search("原材料価格", k=3, mode="bm25", doc_ids=["D1"]))[0] == "D1:1"


def test_BM25の統計は_絞り込みに関係なく全文書のもの(store: ChunkStore) -> None:
    # 別の文書に大量にある2文字は、絞り込んだ先でも軽く見る（IDF は索引全体で計算）
    e = HashEmbedder(dim=64)
    a = Chunk("A:0", "A", ["h"], 1, 1, "圧迫する")
    b = Chunk("B:0", "B", ["h"], 1, 1, "検討する")
    store.upsert_document(DocumentRecord("A", "1", "甲", "2026-03-31", 1), [a])
    store.upsert_document(DocumentRecord("B", "2", "乙", "2026-03-31", 1), [b])
    r = Retriever(store, e)
    only_a = r.search("圧迫する", k=3, mode="bm25", doc_ids=["A"])
    both = r.search("圧迫する", k=3, mode="bm25")
    assert only_a[0].score == both[0].score


def test_索引を作った後に取り込んだ文書も_検索できる(store: ChunkStore) -> None:
    e = HashEmbedder(dim=64)
    r = Retriever(store, e)
    assert r.search("原材料価格", k=3, mode="bm25") == []
    store.upsert_document(
        DocumentRecord("D1", "9999", "サンプル", "2026-03-31", 1),
        [Chunk("D1:0", "D1", ["h"], 1, 1, RISK)],
    )
    assert ids(r.search("原材料価格", k=3, mode="bm25")) == ["D1:0"]


def test_モードを省略すると_評価で最良だったBM25とベクトルの融合(
    loaded: tuple[ChunkStore, HashEmbedder],
) -> None:
    store, e = loaded
    r = Retriever(store, e)
    q = "原材料価格の高騰が収益に与える影響"
    assert ids(r.search(q, k=3)) == ids(r.search(q, k=3, mode="hybrid"))


def test_語句側の索引は_複数のRetrieverで共有できる(
    loaded: tuple[ChunkStore, HashEmbedder],
) -> None:
    store, e = loaded
    shared = LexicalIndex(store)
    r1, r2 = Retriever(store, e, lexical_index=shared), Retriever(store, e, lexical_index=shared)
    r1.search("原材料価格", k=1, mode="bm25")
    built = shared.index
    r2.search("原材料価格", k=1, mode="bm25")
    assert shared.index is built  # 作り直していない


def test_知らないモードはエラー(loaded: tuple[ChunkStore, HashEmbedder]) -> None:
    store, e = loaded
    with pytest.raises(ValueError):
        Retriever(store, e).search("x", k=1, mode="magic")  # type: ignore[arg-type]


def test_複数のスレッドから同時に検索しても_索引の作り直しで結果が壊れない(
    loaded: tuple[ChunkStore, HashEmbedder],
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    store, e = loaded
    shared = LexicalIndex(store)

    def run(_: int) -> list[str]:
        return ids(
            Retriever(store, e, shared).search("原材料価格", k=3, mode="bm25", doc_ids=["D1"])
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(32)))
    assert all(r == ["D1:1"] for r in results)


def test_書類が検索できる状態かを調べられる(loaded: tuple[ChunkStore, HashEmbedder]) -> None:
    store, e = loaded
    r = Retriever(store, e)
    assert r.is_indexed("D1")
    assert not r.is_indexed("NONE")
    # 別のモデルの埋め込みしか無い書類は、このモデルでは検索できない
    other = HashEmbedder(dim=64)
    other_key = Retriever(store, other)
    assert other_key.is_indexed("D1")  # 同じ key "hash"
