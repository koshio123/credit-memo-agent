"""PostgreSQL への保存と検索の統合テスト。実際の DB（pgvector と pg_bigm）を使う。"""

import pytest

from retrieval.chunker import Chunk
from retrieval.embedding import HashEmbedder
from retrieval.store import ChunkStore, DocumentRecord, Hit

pytestmark = pytest.mark.db


def _chunk(doc: str, seq: int, text: str, heading: list[str] | None = None, page: int = 1) -> Chunk:
    return Chunk(f"{doc}:{seq}", doc, heading or ["第2 【事業の状況】"], page, page, text)


DOC_A = DocumentRecord("A", "1111", "甲社", "2026-03-31", 10)
DOC_B = DocumentRecord("B", "2222", "乙社", "2026-03-31", 12)

RISK = "原材料価格が高騰した場合、収益を圧迫する可能性があります。"
STAFF = "従業員の平均年齢は四十歳で、平均勤続年数は十五年です。"
DIV = "配当性向は三十パーセントを目安として、安定的な配当を継続します。"


def test_スキーマの作成は何度実行しても壊れない(store: ChunkStore) -> None:
    store.init_schema()
    store.init_schema()


def test_文書とチャンクを保存して_見出しの階層もそのまま読み出せる(store: ChunkStore) -> None:
    store.upsert_document(
        DOC_A, [_chunk("A", 0, RISK, ["第2 【事業の状況】", "3 【事業等のリスク】"], page=23)]
    )

    (chunk,) = store.get_chunks("A")

    assert chunk.chunk_id == "A:0"
    assert chunk.heading_path == ["第2 【事業の状況】", "3 【事業等のリスク】"]
    assert (chunk.page_start, chunk.page_end) == (23, 23)
    assert chunk.text == RISK


def test_同じ文書を保存し直すと_古いチャンクと埋め込みは消える(store: ChunkStore) -> None:
    e = HashEmbedder(dim=8)
    store.upsert_document(DOC_A, [_chunk("A", 0, RISK), _chunk("A", 1, STAFF)])
    store.add_embeddings("hash", ["A:0", "A:1"], e.embed_documents([RISK, STAFF]))

    store.upsert_document(DOC_A, [_chunk("A", 0, DIV)])

    assert [c.text for c in store.get_chunks("A")] == [DIV]
    assert store.count_embeddings("hash", "A") == 0  # 古いチャンクの埋め込みが残らない


# ---- 全文検索（pg_bigm の 2-gram 類似度） ----


def test_全文検索は_2gramの類似度が高い順に返し_順位は1始まり(store: ChunkStore) -> None:
    store.upsert_document(DOC_A, [_chunk("A", 0, STAFF), _chunk("A", 1, RISK), _chunk("A", 2, DIV)])

    hits = store.search_lexical("原材料価格の高騰は収益にどう影響するか", k=3)

    assert hits[0].chunk.chunk_id == "A:1"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert hits[0].score >= hits[-1].score


def test_全文検索は_文書で絞り込める(store: ChunkStore) -> None:
    store.upsert_document(DOC_A, [_chunk("A", 0, RISK)])
    store.upsert_document(DOC_B, [_chunk("B", 0, RISK)])

    hits = store.search_lexical("原材料価格の高騰", k=5, doc_ids=["B"])

    assert [h.chunk.doc_id for h in hits] == ["B"]


def test_全文検索は_共通の2gramが無いチャンクを返さない(store: ChunkStore) -> None:
    store.upsert_document(DOC_A, [_chunk("A", 0, "ABCDEFG")])
    assert store.search_lexical("あいうえお", k=5) == []


# ---- ベクトル検索（pgvector のコサイン距離） ----


def test_ベクトル検索は_近い順に返す(store: ChunkStore) -> None:
    e = HashEmbedder(dim=64)
    store.upsert_document(DOC_A, [_chunk("A", 0, STAFF), _chunk("A", 1, RISK), _chunk("A", 2, DIV)])
    store.add_embeddings("hash", ["A:0", "A:1", "A:2"], e.embed_documents([STAFF, RISK, DIV]))

    hits = store.search_vector("hash", e.embed_queries(["原材料価格の高騰"])[0], k=3)

    assert hits[0].chunk.chunk_id == "A:1"
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score > hits[-1].score


def test_ベクトル検索は_モデルと文書で絞り込む(store: ChunkStore) -> None:
    e = HashEmbedder(dim=16)
    store.upsert_document(DOC_A, [_chunk("A", 0, RISK)])
    store.upsert_document(DOC_B, [_chunk("B", 0, RISK)])
    store.add_embeddings("m1", ["A:0", "B:0"], e.embed_documents([RISK, RISK]))
    store.add_embeddings("m2", ["A:0"], e.embed_documents([RISK]))
    q = e.embed_queries(["原材料"])[0]

    assert {h.chunk.doc_id for h in store.search_vector("m1", q, k=5)} == {"A", "B"}
    assert [h.chunk.doc_id for h in store.search_vector("m1", q, k=5, doc_ids=["B"])] == ["B"]
    assert len(store.search_vector("m2", q, k=5)) == 1  # 別のモデルの埋め込みは混ざらない


def test_埋め込みの追加は上書きできる(store: ChunkStore) -> None:
    e = HashEmbedder(dim=8)
    store.upsert_document(DOC_A, [_chunk("A", 0, RISK)])
    store.add_embeddings("hash", ["A:0"], e.embed_documents([RISK]))
    store.add_embeddings("hash", ["A:0"], e.embed_documents([STAFF]))

    assert store.count_embeddings("hash", "A") == 1
    hits = store.search_vector("hash", e.embed_queries([STAFF])[0], k=1)
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_埋め込みの数を数えられる(store: ChunkStore) -> None:
    e = HashEmbedder(dim=8)
    store.upsert_document(DOC_A, [_chunk("A", 0, RISK), _chunk("A", 1, STAFF)])
    assert store.count_embeddings("hash", "A") == 0
    store.add_embeddings("hash", ["A:0"], e.embed_documents([RISK]))
    assert store.count_embeddings("hash", "A") == 1
    assert isinstance(store.search_lexical("原材料", k=1)[0], Hit)
