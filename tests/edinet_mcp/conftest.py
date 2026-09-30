"""edinet_mcp のテストで使う、小さな合成データ。実データ（data/）には依存しない。"""

from collections.abc import Sequence
from decimal import Decimal

import pytest

from edinet_mcp.service import EdinetService
from evals.companies import Company, Filing, Filings, SelectionSnapshot
from finance.ratios import compute_ratios
from ingest.xbrl_facts import Fact, extract
from retrieval.chunker import Chunk
from retrieval.store import Hit

CUR_DOC, PREV_DOC = "S100CUR0", "S100PRV0"


def company(sec_code: str = "9999", name: str = "サンプル工業", doc_id: str = CUR_DOC) -> Company:
    return Company(
        sec_code=sec_code,
        edinet_code="E00000",
        name=name,
        industry="機械",
        fiscal_year_end="3月31日",
        capital_million_yen=1000,
        filings=Filings(
            current=Filing(doc_id=doc_id, period_end="2026-03-31", submitted="2026-06-24"),
            previous=Filing(doc_id=PREV_DOC, period_end="2025-03-31", submitted="2025-06-25"),
        ),
        why_selected="テスト用",
        selection_snapshot=SelectionSnapshot(
            note="テスト", equity_ratio_pct=50, current_ratio_pct=200, operating_margin_pct=10
        ),
    )


def _facts(prefix: str, values: dict[str, int]) -> list[Fact]:
    out: list[Fact] = []
    for element, value in values.items():
        duration = element in {"NetSales", "OperatingIncome"}
        context = f"{prefix}{'Duration' if duration else 'Instant'}"
        out.append(Fact(element, context, "連結", Decimal(value), nil=False))
    return out


CURRENT = {
    "Assets": 1000,
    "NetAssets": 500,
    "CurrentAssets": 600,
    "CurrentLiabilities": 300,
    "NetSales": 2000,
    "OperatingIncome": 200,
}
PREVIOUS = {**CURRENT, "NetSales": 1800, "OperatingIncome": 150}
FACTS = _facts("CurrentYear", CURRENT) + _facts("Prior1Year", PREVIOUS)

PAGES = ["表紙", "3 【事業等のリスク】\n原材料価格が高騰する可能性があります。", "従業員の状況"]


class FakeSearcher:
    """検索の代わり。渡された引数を記録し、決まったヒットを返す。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(
        self, query: str, k: int, mode: str = "hybrid_bm25", doc_ids: Sequence[str] | None = None
    ) -> list[Hit]:
        self.calls.append({"query": query, "k": k, "doc_ids": list(doc_ids or [])})
        chunk = Chunk(
            f"{CUR_DOC}:1", CUR_DOC, ["第2 【事業の状況】", "3 【事業等のリスク】"], 2, 3, PAGES[1]
        )
        return [Hit(chunk, 0.5, 1)]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"  # trio は使わない


@pytest.fixture
def searcher() -> FakeSearcher:
    return FakeSearcher()


@pytest.fixture
def service(searcher: FakeSearcher) -> EdinetService:
    def load_facts(doc_id: str) -> list[Fact]:
        if doc_id == CUR_DOC:
            return FACTS
        raise FileNotFoundError(doc_id)

    def load_pages(doc_id: str) -> list[str]:
        if doc_id == CUR_DOC:
            return PAGES
        raise FileNotFoundError(doc_id)

    return EdinetService(
        searcher=searcher,
        companies=[company(), company("8888", "別の会社", "S100OTH0")],
        load_facts=load_facts,
        load_pages=load_pages,
    )


def expected_ratios():
    current = extract(FACTS, "current").financials
    previous = extract(FACTS, "previous").financials
    return compute_ratios(current, previous)
