"""出典スパン（doc_id・ページ・文字位置・引用文）の作成と検証。引用文はコードが作る。"""

import pytest
from pydantic import ValidationError

from agents.citations import SourceSpan, SpanNotFoundError, locate_chunk, make_span, verify_span
from retrieval.chunker import Chunk, chunk_pages

PAGES = [
    "表紙\n会社の概要",
    "3 【事業等のリスク】\n"
    "原材料価格が高騰した場合、収益を圧迫する可能性があります。\n"
    "為替の変動も影響します。",
    "為替の変動は、輸出入の価格に影響します。\n4 【従業員の状況】\n従業員の平均年齢は四十歳です。",
]


# ---- スパンの検証: 引用文 == ページ本文[開始:終了] ----


def test_スパンは_引用文がページ本文の該当位置と一致すれば正しい() -> None:
    span = SourceSpan(doc_id="D1", page=2, start=13, end=25, quote=PAGES[1][13:25])
    assert verify_span(span, PAGES)


def test_引用文が違えば_不正() -> None:
    span = SourceSpan(doc_id="D1", page=2, start=13, end=25, quote="あ" * 12)
    assert not verify_span(span, PAGES)


def test_位置がページの外なら_不正() -> None:
    assert not verify_span(
        SourceSpan(doc_id="D1", page=2, start=0, end=9999, quote="x" * 9999), PAGES
    )
    assert not verify_span(SourceSpan(doc_id="D1", page=99, start=0, end=1, quote="表"), PAGES)


def test_スパンは_位置と引用文の長さが合わないと作れない() -> None:
    with pytest.raises(ValidationError):
        SourceSpan(doc_id="D1", page=1, start=0, end=5, quote="表紙")  # 5文字ではない


def test_ページは1以上_開始は0以上_終了は開始より後() -> None:
    for kwargs in (
        {"page": 0, "start": 0, "end": 1, "quote": "x"},
        {"page": 1, "start": -1, "end": 1, "quote": "xx"},
        {"page": 1, "start": 2, "end": 2, "quote": ""},
    ):
        with pytest.raises(ValidationError):
            SourceSpan(doc_id="D1", **kwargs)  # type: ignore[arg-type]


# ---- 引用文からスパンを作る ----


def test_引用文の位置を探してスパンを作る() -> None:
    span = make_span("D1", 2, PAGES, "収益を圧迫する可能性があります")
    assert span is not None
    assert (span.page, PAGES[1][span.start : span.end]) == (2, "収益を圧迫する可能性があります")
    assert verify_span(span, PAGES)


def test_ページに無い引用文は_Noneを返す_推測で位置を作らない() -> None:
    assert make_span("D1", 2, PAGES, "存在しない文") is None
    assert make_span("D1", 9, PAGES, "表紙") is None
    assert make_span("D1", 2, PAGES, "") is None


# ---- チャンクからスパンを作る ----


def _chunk(text: str, p0: int, p1: int) -> Chunk:
    return Chunk("D1:0", "D1", ["第2 【事業の状況】"], p0, p1, text)


def test_単一ページのチャンクは_1つのスパンになる() -> None:
    chunk = _chunk("原材料価格が高騰した場合、収益を圧迫する可能性があります。", 2, 2)
    (span,) = locate_chunk(chunk, PAGES)
    assert verify_span(span, PAGES)
    assert span.quote == chunk.text


def test_複数行のチャンクも_1つのスパンになる() -> None:
    chunk = _chunk(
        "原材料価格が高騰した場合、収益を圧迫する可能性があります。\n為替の変動も影響します。", 2, 2
    )
    (span,) = locate_chunk(chunk, PAGES)
    assert span.quote == chunk.text


def test_複数ページにまたがるチャンクは_ページごとのスパンに分ける() -> None:
    chunk = _chunk("為替の変動も影響します。\n為替の変動は、輸出入の価格に影響します。", 2, 3)
    spans = locate_chunk(chunk, PAGES)
    assert [s.page for s in spans] == [2, 3]
    assert all(verify_span(s, PAGES) for s in spans)
    assert "\n".join(s.quote for s in spans) == chunk.text


def test_チャンクの本文がページに無ければ_エラーにする() -> None:
    with pytest.raises(SpanNotFoundError):
        locate_chunk(_chunk("どこにも無い文", 2, 2), PAGES)


def test_チャンク分けの結果は_すべてスパンにできる() -> None:
    # チャンクの本文はページ本文の連続した部分（体裁の行を除く）という前提の確認
    header = "EDINET提出書類\nサンプル株式会社(E00000)\n有価証券報告書\n"
    pages = [
        f"{header}第2 【事業の状況】\n3 【事業等のリスク】\n{'あ' * 40}。{'い' * 40}。\n1/3",
        f"{header}{'う' * 30}。\n4 【従業員の状況】\n従業員は百名です。\n2/3",
    ]
    chunks = chunk_pages("D1", pages, max_chars=50)
    assert chunks
    for chunk in chunks:
        spans = locate_chunk(chunk, pages)
        assert all(verify_span(s, pages) for s in spans)
