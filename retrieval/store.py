"""チャンクの保存と、ベクトル検索（PostgreSQL + pgvector）。

- 語句の検索（BM25）は retrieval/search.py の LexicalIndex が、チャンクからメモリ上に作る。
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
    score: float  # 方式ごとに尺度が違う（BM25: 点数 / ベクトル: コサイン類似度）
    rank: int  # 1 始まり


def _chunk_from_row(row: Sequence[object]) -> Chunk:
    """DB の行をチャンクにする。

    Args:
        row: chunk_id, doc_id, heading_path, page_start, page_end, text の順の行。

    Returns:
        チャンク。
    """
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
    def __init__(
        self,
        conn: psycopg.Connection[tuple[object, ...]],
        url: str = "",
        schema: str | None = None,
    ) -> None:
        """接続済みの DB からストアを作る。

        Args:
            conn: PostgreSQL の接続。
            url: 接続先のURL（別の接続を開くときに使う）。
            schema: 表を置くスキーマ。None なら既定。
        """
        self._conn = conn
        self.url = url
        self.schema = schema

    def in_transaction(self) -> bool:
        """未確定のトランザクションが開いたままか。

        Returns:
            開いたままなら True。
        """
        return self._conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE

    @classmethod
    def connect(cls, url: str, schema: str | None = None) -> ChunkStore:
        """接続する。schema を指定すると、その中に表を作る（テストの分離用）。

        Args:
            url: PostgreSQL の接続URL。
            schema: 表を作るスキーマ。None なら既定。

        Returns:
            表の作成まで済んだストア。
        """
        # autocommit にする。そうしないと、読み取りで暗黙のトランザクションが開いたままになり、
        # 続く書き込みの transaction() が入れ子（セーブポイント）になって確定されず、接続を閉じると
        # 書き込みがすべて巻き戻される（取り込みで実際に起きた）。autocommit なら、読み取りは
        # そのつど確定し、書き込みは transaction() の範囲で確実に確定する。
        conn = psycopg.connect(url, row_factory=tuple_row, autocommit=True)
        # vector 型を登録する前に、拡張がなければならない
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if schema is not None:
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        register_vector(conn)
        store = cls(conn, url=url, schema=schema)
        store.init_schema()
        return store

    def close(self) -> None:
        self._conn.close()

    def init_schema(self) -> None:
        self._conn.execute(_SCHEMA_SQL)  # type: ignore[arg-type]

    # ---- 保存 ----

    def upsert_document(self, doc: DocumentRecord, chunks: Sequence[Chunk]) -> None:
        """文書を保存し直す。古いチャンクと、その埋め込みは消える（外部キーの連鎖削除）。

        Args:
            doc: 文書の情報。
            chunks: 保存するチャンク（並び順がそのまま seq になる）。
        """
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
        """チャンクの埋め込みを保存する。同じチャンク・モデルの組があれば上書きする。

        Args:
            model: 埋め込みモデルの識別子。
            chunk_ids: チャンクID。
            vectors: チャンクIDと同じ順のベクトル。

        Raises:
            ValueError: chunk_ids と vectors の数が違うとき。
        """
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
        """文書のチャンクを、並び順に返す。

        Args:
            doc_id: 書類ID。

        Returns:
            チャンクのリスト。文書が無ければ空。
        """
        query = sql.SQL("SELECT {cols} FROM chunks c WHERE c.doc_id = %s ORDER BY c.seq").format(
            cols=_CHUNK_COLUMNS
        )
        rows = self._conn.execute(query, (doc_id,)).fetchall()
        return [_chunk_from_row(r) for r in rows]

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """文書の情報を返す。

        Args:
            doc_id: 書類ID。

        Returns:
            文書の情報。保存されていなければ None。
        """
        row = self._conn.execute(
            "SELECT doc_id, sec_code, company, period_end, n_pages FROM documents"
            " WHERE doc_id = %s",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        return DocumentRecord(str(row[0]), str(row[1]), str(row[2]), str(row[3]), int(row[4]))  # type: ignore[call-overload]

    def embedded_models(self, doc_id: str) -> list[str]:
        """文書のチャンクに埋め込みが保存されているモデル（名前順）。

        Args:
            doc_id: 書類ID。

        Returns:
            モデルの識別子のリスト。
        """
        rows = self._conn.execute(
            "SELECT DISTINCT e.model FROM chunk_embeddings e JOIN chunks c USING (chunk_id)"
            " WHERE c.doc_id = %s ORDER BY e.model",
            (doc_id,),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        """全文書のチャンクを、文書・並び順に返す（BM25 の索引を作るのに使う）。

        Returns:
            全チャンクのリスト。
        """
        query = sql.SQL("SELECT {cols} FROM chunks c ORDER BY c.doc_id, c.seq").format(
            cols=_CHUNK_COLUMNS
        )
        return [_chunk_from_row(r) for r in self._conn.execute(query).fetchall()]

    def corpus_fingerprint(self) -> tuple[int, int]:
        """チャンク全体の目印（件数と本文のハッシュの和）。取り込みで変わったかを安く調べる。

        Returns:
            (チャンク数, ハッシュの和)。
        """
        row = self._conn.execute(
            "SELECT count(*), coalesce(sum(hashtext(chunk_id || text)::bigint), 0) FROM chunks"
        ).fetchone()
        return (int(row[0]), int(row[1])) if row else (0, 0)  # type: ignore[call-overload]

    def chunks_unchanged(self, doc_id: str, chunks: Sequence[Chunk]) -> bool:
        """保存済みのチャンクが、渡されたものと完全に同じか（ID・見出し・ページ・本文）。

        Args:
            doc_id: 書類ID。
            chunks: 比べるチャンク。

        Returns:
            同じなら True。
        """
        return self.get_chunks(doc_id) == list(chunks)

    def count_embeddings(self, model: str, doc_id: str) -> int:
        """文書のチャンクのうち、指定したモデルの埋め込みが保存されている数。

        Args:
            model: 埋め込みモデルの識別子。
            doc_id: 書類ID。

        Returns:
            保存されている埋め込みの数。
        """
        row = self._conn.execute(
            "SELECT count(*) FROM chunk_embeddings e JOIN chunks c USING (chunk_id)"
            " WHERE e.model = %s AND c.doc_id = %s",
            (model, doc_id),
        ).fetchone()
        return int(row[0]) if row else 0  # type: ignore[call-overload]

    # ---- 検索 ----

    def search_vector(
        self,
        model: str,
        query_vector: Sequence[float],
        k: int,
        doc_ids: Sequence[str] | None = None,
    ) -> list[Hit]:
        """コサイン類似度が高い順（1 - コサイン距離）。指定したモデルの埋め込みだけを対象にする。

        Args:
            model: 埋め込みモデルの識別子。
            query_vector: 質問のベクトル。
            k: 返す件数の上限。
            doc_ids: 対象にする書類ID。None なら全書類。

        Returns:
            順位つきの検索結果（類似度の高い順）。
        """
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
