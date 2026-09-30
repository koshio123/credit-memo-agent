"""ツールの中身。検索・XBRL・PDF の読み込みは外から渡す（テストで差し替えられる）。

- 財務数値は XBRL から取る（機械可読で正確）。PDF は本文の検索と、出典のページに使う。
- 比率と水準の判定は finance/ratios.py（架空の融資内規に基づく）。融資の可否は返さない。
"""

import unicodedata
from collections.abc import Callable, Sequence
from typing import Protocol

from edinet_mcp.models import (
    CompanyInfo,
    FinancialsResult,
    PageResult,
    Passage,
    Period,
    RatiosResult,
)
from evals.companies import Company
from finance.ratios import compute_ratios
from ingest.xbrl_facts import (
    Fact,
    NoConsolidatedStatements,
    UnsupportedAccountingStandard,
    extract,
)
from retrieval.store import Hit

MAX_K = 20
POLICY_NOTE = (
    "算式と水準の境界は、架空の融資内規（policies/credit_policy.md、第7条ほか）による。"
    "融資の可否や条件の判断は含まない。"
)


class EdinetMcpError(Exception):
    """呼び出し側（エージェント）に、そのまま見せてよい失敗。"""


class Searcher(Protocol):
    def search(self, query: str, k: int, *, doc_ids: Sequence[str] | None = None) -> list[Hit]: ...


class EdinetService:
    def __init__(
        self,
        searcher: Searcher,
        companies: Sequence[Company],
        load_facts: Callable[[str], list[Fact]],
        load_pages: Callable[[str], list[str]],
    ) -> None:
        self._searcher = searcher
        self._companies = {c.sec_code: c for c in companies}
        self._load_facts = load_facts
        self._load_pages = load_pages

    # ---- 会社 ----

    def list_companies(self) -> list[CompanyInfo]:
        return [
            CompanyInfo(
                sec_code=c.sec_code,
                name=c.name,
                industry=c.industry,
                doc_id=c.filings.current.doc_id,
                period_end=c.filings.current.period_end,
            )
            for c in self._companies.values()
        ]

    def _company(self, sec_code: str) -> Company:
        code = unicodedata.normalize("NFKC", sec_code).strip()
        if len(code) == 5 and code.endswith("0"):  # 証券コードの5桁表記（末尾0）
            code = code[:4]
        company = self._companies.get(code)
        if company is None:
            known = ", ".join(self._companies)
            raise EdinetMcpError(f"証券コード {code!r} は対象外です。対象: {known}")
        return company

    # ---- 検索 ----

    def search_filings(self, query: str, sec_code: str, k: int = 5) -> list[Passage]:
        """会社の当期の有価証券報告書から、質問に近い本文を探す。"""
        if not 1 <= k <= MAX_K:
            raise EdinetMcpError(f"k は 1 から {MAX_K} までです（{k}）")
        if not query.strip():
            raise EdinetMcpError("クエリが空です")
        company = self._company(sec_code)
        filing = company.filings.current
        hits = self._searcher.search(query, k, doc_ids=[filing.doc_id])
        return [
            Passage(
                sec_code=company.sec_code,
                company=company.name,
                doc_id=hit.chunk.doc_id,
                period_end=filing.period_end,
                page_start=hit.chunk.page_start,
                page_end=hit.chunk.page_end,
                heading_path=hit.chunk.heading_path,
                text=hit.chunk.text,
                rank=hit.rank,
                score=hit.score,
            )
            for hit in hits
        ]

    # ---- 財務数値・比率（XBRL） ----

    def _facts(self, doc_id: str) -> list[Fact]:
        try:
            return list(self._load_facts(doc_id))
        except FileNotFoundError as e:
            raise EdinetMcpError(
                f"書類 {doc_id} の財務データ（XBRL の CSV）がありません。先に取得してください"
            ) from e

    def get_financials(self, sec_code: str, period: Period = "current") -> FinancialsResult:
        if period not in ("current", "previous"):
            raise EdinetMcpError(f"period は current か previous です（{period!r}）")
        company = self._company(sec_code)
        # 前期の値も、当期の書類の前期の列から読む
        doc_id = company.filings.current.doc_id
        extraction = self._extract(doc_id, period)
        filing = company.filings.current if period == "current" else company.filings.previous
        return FinancialsResult(
            sec_code=company.sec_code,
            company=company.name,
            doc_id=doc_id,
            period=period,
            period_end=filing.period_end,
            source="XBRL",
            unit="円",
            financials=extraction.financials,
            provenance=extraction.provenance,
        )

    def _extract(self, doc_id: str, period: Period):
        facts = self._facts(doc_id)
        try:
            return extract(facts, period)
        except UnsupportedAccountingStandard as e:
            raise EdinetMcpError(
                f"書類 {doc_id} は IFRS の連結財務諸表で、現在は読めません（J-GAAP のみ対応）"
            ) from e
        except NoConsolidatedStatements as e:
            raise EdinetMcpError(f"書類 {doc_id}: {e}") from e

    def get_ratios(self, sec_code: str) -> RatiosResult:
        company = self._company(sec_code)
        doc_id = company.filings.current.doc_id
        current = self._extract(doc_id, "current").financials
        previous = self._extract(doc_id, "previous").financials
        return RatiosResult(
            sec_code=company.sec_code,
            company=company.name,
            doc_id=doc_id,
            period_end=company.filings.current.period_end,
            ratios=compute_ratios(current, previous),
            policy_note=POLICY_NOTE,
        )

    # ---- ページ ----

    def get_page(self, sec_code: str, page: int) -> PageResult:
        """当期の有価証券報告書の1ページ分の本文（1 始まり）。検索の出典を確かめるのに使う。"""
        company = self._company(sec_code)
        doc_id = company.filings.current.doc_id
        try:
            pages = self._load_pages(doc_id)
        except FileNotFoundError as e:
            raise EdinetMcpError(f"書類 {doc_id} の PDF がありません。先に取得してください") from e
        if not 1 <= page <= len(pages):
            raise EdinetMcpError(f"ページは 1 から {len(pages)} までです（{page}）")
        return PageResult(
            sec_code=company.sec_code,
            company=company.name,
            doc_id=doc_id,
            period_end=company.filings.current.period_end,
            page=page,
            n_pages=len(pages),
            text=pages[page - 1],
        )
