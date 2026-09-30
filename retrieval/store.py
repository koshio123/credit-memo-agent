"""チャンクの保存と検索（PostgreSQL + pgvector + pg_bigm）。

- 全文検索は pg_bigm の 2-gram 類似度（形態素解析器が要らず、日本語の部分一致に強い）。
- ベクトル検索は pgvector のコサイン距離。埋め込みはモデルごとに保存し、検索は必ず model で絞る。
- スコアは検索方式ごとに尺度が違う。方式をまたぐ比較には順位（rank）を使う（retrieval/fusion.py）。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector import Vector  # type: ignore[import-untyped]
from pgvector.psycopg import register_vector  # type: ignore[import-untyped]
from psycopg import sql
from psycopg.rows import tuple_row

from retrieval.chunker import Chunk

_SCHEMA_SQL = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")

_CHUNK_COLUMNS = sql.SQL("c.chunk_id, c.doc_id, c.heading_path, c.page_start, c.page_end, c.text")


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    sec_code: str
    company: str
    period_end: str
    n_pages: int


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # 方式ごとに尺度が違う（全文: 2-gram 類似度 / ベクトル: コサイン類似度）
    rank: int  # 1 始まり


def _chunk_from_row(row: Sequence[object]) -> Chunk:
    chunk_id, doc_id, heading_path, page_start, page_end, text = row
    return Chunk(
        chunk_id=str(chunk_id),
        doc_id=str(doc_id),
        heading_path=list(heading_path),  # type: ignore[call-overload]
        page_start=int(page_start),  # type: ignore[call-overload]
        page_end=int(page_end),  # type: ignore[call-overload]
        text=str(text),
    )


class ChunkStore:
    def __init__(self, conn: psycopg.Connection[tuple[object, ...]]) -> None:
        self._conn = conn

    @classmethod
    def connect(cls, url: str, schema: str | None = None) -> ChunkStore:
        """接続する。schema を指定すると、その中に表を作る（テストの分離用）。"""
        conn = psycopg.connect(url, row_factory=tuple_row)
        # vector 型を登録する前に、拡張がなければならない
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_bigm")
        conn.commit()
        if schema is not None:
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        register_vector(conn)
        store = cls(conn)
        store.init_schema()
        return store

    def close(self) -> None:
        self._conn.close()

    def init_schema(self) -> None:
        self._conn.execute(_SCHEMA_SQL)  # type: ignore[arg-type]
        self._conn.commit()

    # ---- 保存 ----

    def upsert_document(self, doc: DocumentRecord, chunks: Sequence[Chunk]) -> None:
        """文書を保存し直す。古いチャンクと、その埋め込みは消える（外部キーの連鎖削除）。"""
        with self._conn.transaction():
            self._conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc.doc_id,))
            self._conn.execute(
                "INSERT INTO documents (doc_id, sec_code, company, period_end, n_pages)"
                " VALUES (%s, %s, %s, %s, %s)",
                (doc.doc_id, doc.sec_code, doc.company, doc.period_end, doc.n_pages),
            )
            with self._conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks"
                    " (chunk_id, doc_id, seq, heading_path, page_start, page_end, text)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [
                        (c.chunk_id, c.doc_id, i, c.heading_path, c.page_start, c.page_end, c.text)
                        for i, c in enumerate(chunks)
                    ],
                )

    def add_embeddings(
        self, model: str, chunk_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids と vectors の数が違います")
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunk_embeddings (chunk_id, model, embedding) VALUES (%s, %s, %s)"
                " ON CONFLICT (chunk_id, model) DO UPDATE SET embedding = EXCLUDED.embedding",
                [(cid, model, Vector(list(v))) for cid, v in zip(chunk_ids, vectors, strict=True)],
            )

    # ---- 読み出し ----

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        query = sql.SQL("SELECT {cols} FROM chunks c WHERE c.doc_id = %s ORDER BY c.seq").format(
            cols=_CHUNK_COLUMNS
        )
        rows = self._conn.execute(query, (doc_id,)).fetchall()
        return [_chunk_from_row(r) for r in rows]

    def count_embeddings(self, model: str, doc_id: str) -> int:
        row = self._conn.execute(
            "SELECT count(*) FROM chunk_embeddings e JOIN chunks c USING (chunk_id)"
            " WHERE e.model = %s AND c.doc_id = %s",
            (model, doc_id),
        ).fetchone()
        return int(row[0]) if row else 0  # type: ignore[call-overload]

    # ---- 検索 ----

    def search_lexical(self, query: str, k: int, doc_ids: Sequence[str] | None = None) -> list[Hit]:
        """pg_bigm の 2-gram 類似度が高い順。共通の 2-gram が無いチャンクは返さない。"""
        statement = sql.SQL(
            "SELECT {cols}, s.score FROM ("
            "  SELECT chunk_id, bigm_similarity(text, %(q)s) AS score FROM chunks"
            "  WHERE %(docs)s::text[] IS NULL OR doc_id = ANY(%(docs)s)"
            ") s JOIN chunks c USING (chunk_id) WHERE s.score > 0"
            " ORDER BY s.score DESC, c.chunk_id LIMIT %(k)s"
        ).format(cols=_CHUNK_COLUMNS)
        rows = self._conn.execute(
            statement,
            {"q": query, "docs": list(doc_ids) if doc_ids is not None else None, "k": k},
        ).fetchall()
        return [Hit(_chunk_from_row(r[:6]), float(r[6]), i) for i, r in enumerate(rows, 1)]  # type: ignore[arg-type]

    def search_vector(
        self,
        model: str,
        query_vector: Sequence[float],
        k: int,
        doc_ids: Sequence[str] | None = None,
    ) -> list[Hit]:
        """コサイン類似度が高い順（1 - コサイン距離）。指定したモデルの埋め込みだけを対象にする。"""
        statement = sql.SQL(
            "SELECT {cols}, 1 - (e.embedding <=> %(v)s) AS score"
            " FROM chunk_embeddings e JOIN chunks c USING (chunk_id)"
            " WHERE e.model = %(m)s AND (%(docs)s::text[] IS NULL OR c.doc_id = ANY(%(docs)s))"
            " ORDER BY e.embedding <=> %(v)s, c.chunk_id LIMIT %(k)s"
        ).format(cols=_CHUNK_COLUMNS)
        rows = self._conn.execute(
            statement,
            {
                "v": Vector(list(query_vector)),
                "m": model,
                "docs": list(doc_ids) if doc_ids is not None else None,
                "k": k,
            },
        ).fetchall()
        return [Hit(_chunk_from_row(r[:6]), float(r[6]), i) for i, r in enumerate(rows, 1)]  # type: ignore[arg-type]
