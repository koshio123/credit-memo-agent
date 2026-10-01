"""出典スパン: どの書類の何ページの、どの文字範囲か。

- 引用文は LLM が書かない。検索結果のチャンクや、探し出した位置から、コードが作る。
- スパンは「引用文 == そのページの本文[開始:終了]」をコードで検証できる。
- ページの本文は、PDF から取り出した本文（`ingest/pdf_baseline.read_pages`。NFKC 正規化済み）。
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from retrieval.chunker import Chunk


class SpanNotFoundError(Exception):
    """チャンクの本文が、ページの本文の中に見つからない。"""


class SourceSpan(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    page: int = Field(ge=1)  # 1 始まり
    start: int = Field(ge=0)  # ページの本文での開始位置（文字）
    end: int  # 終了位置（含まない）
    quote: str

    @model_validator(mode="after")
    def _consistent(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("end は start より後にする")
        if len(self.quote) != self.end - self.start:
            raise ValueError("quote の長さが end - start と合わない")
        return self


def verify_span(span: SourceSpan, pages: Sequence[str]) -> bool:
    """引用文が、そのページの本文の該当位置と一致するか。"""
    if not 1 <= span.page <= len(pages):
        return False
    return pages[span.page - 1][span.start : span.end] == span.quote


def make_span(doc_id: str, page: int, pages: Sequence[str], quote: str) -> SourceSpan | None:
    """ページの本文から引用文を探してスパンを作る。見つからなければ None（位置を推測しない）。"""
    if not quote or not 1 <= page <= len(pages):
        return None
    start = pages[page - 1].find(quote)
    if start < 0:
        return None
    return SourceSpan(doc_id=doc_id, page=page, start=start, end=start + len(quote), quote=quote)


def locate_chunk(chunk: Chunk, pages: Sequence[str]) -> list[SourceSpan]:
    """チャンクの本文を、ページごとのスパンに分ける（複数ページにまたがるチャンクは複数になる）。

    チャンクの本文は、ページ本文の連続した行（体裁の行を除く）なので、各ページで先頭から
    できるだけ多くの行を取り、残りを次のページで探す。同じ行がページ内に複数あるときは、
    最初の出現位置になる（引用文とページは正しいが、位置が別の出現になりうる）。
    """
    lines = chunk.text.split("\n")
    spans: list[SourceSpan] = []
    i = 0
    for page in range(chunk.page_start, chunk.page_end + 1):
        if i >= len(lines) or not 1 <= page <= len(pages):
            continue
        text = pages[page - 1]
        taken = 0
        while i + taken < len(lines) and "\n".join(lines[i : i + taken + 1]) in text:
            taken += 1
        if taken == 0:
            continue
        quote = "\n".join(lines[i : i + taken])
        start = text.find(quote)
        spans.append(
            SourceSpan(
                doc_id=chunk.doc_id, page=page, start=start, end=start + len(quote), quote=quote
            )
        )
        i += taken
    if i < len(lines):
        raise SpanNotFoundError(
            f"チャンク {chunk.chunk_id} の本文を、ページ本文から見つけられません"
        )
    return spans
