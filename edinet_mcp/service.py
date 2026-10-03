"""ツールの中身。検索・XBRL・PDF の読み込みは外から渡す（テストで差し替えられる）。

- 財務数値は XBRL から取る（機械可読で正確）。PDF は本文の検索と、出典のページに使う。
- 比率と水準の判定は finance/ratios.py（架空の融資内規に基づく）。融資の可否は返さない。
"""

import logging
import unicodedata
import zipfile
from collections.abc import Callable, Sequence
from decimal import Decimal
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
from finance.ratios import PeriodFinancials, compute_ratios
from ingest.pdf_baseline import ExtractedValue, extract_items
from ingest.xbrl_facts import (
    Extraction,
    Fact,
    NoConsolidatedStatements,
    UnsupportedAccountingStandard,
    extract,
)
from retrieval.citations import SourceSpan, SpanNotFoundError, locate_chunk
from retrieval.store import Hit

logger = logging.getLogger(__name__)

MAX_K = 20
POLICY_NOTE = (
    "算式と水準の境界は、架空の融資内規（policies/credit_policy.md、第7条ほか）による。"
    "融資の可否や条件の判断は含まない。"
)


class EdinetMcpError(Exception):
    """呼び出し側（エージェント）に、そのまま見せてよい失敗。"""


class Searcher(Protocol):
    def search(self, query: str, k: int, *, doc_ids: Sequence[str] | None = None) -> list[Hit]:
        """本文のチャンクを検索する。

        Args:
            query: 質問文。
            k: 返す件数の上限。
            doc_ids: 対象にする書類ID。None なら全書類。

        Returns:
            順位つきの検索結果。
        """
        ...

    def is_indexed(self, doc_id: str) -> bool:
        """書類が検索の索引に入っているか。入っていないのに空の結果を返すと、記載なしと誤解される。

        Args:
            doc_id: 書類ID。

        Returns:
            索引に入っていれば True。
        """
        ...


class EdinetService:
    def __init__(
        self,
        searcher: Searcher,
        companies: Sequence[Company],
        load_facts: Callable[[str], list[Fact]],
        load_pages: Callable[[str], list[str]],
        pdf_items: Callable[[list[str]], dict[str, ExtractedValue]] = extract_items,
    ) -> None:
        """サービス層を作る。

        Args:
            searcher: 本文の検索。
            companies: 対象の会社。
            load_facts: 書類IDから XBRL の数値の行を読む関数。
            load_pages: 書類IDからページごとの本文を読む関数。
            pdf_items: ページごとの本文から PDF の項目を読む関数。テストで差し替える。
        """
        self._searcher = searcher
        self._companies = {c.sec_code: c for c in companies}
        self._load_facts = load_facts
        self._load_pages = load_pages
        self._pdf_items = pdf_items
        self._pages_cache: dict[str, list[str]] = {}
        self._pdf_cache: dict[str, dict[str, ExtractedValue]] = {}

    # ---- 会社 ----

    def list_companies(self) -> list[CompanyInfo]:
        """調べられる会社の一覧。

        Returns:
            会社の情報（証券コード・会社名・業種・当期の書類ID・期末）。
        """
        return [
            CompanyInfo(
                sec_code=c.sec_code,
                name=c.name,
                industry=c.industry,
                doc_id=c.filings.current.doc_id,
                period_end=c.filings.current.period_end,
                previous_period_end=c.filings.previous.period_end,
            )
            for c in self._companies.values()
        ]

    def _company(self, sec_code: str) -> Company:
        """証券コードから会社を探す。全角や5桁表記（末尾0）も受け付ける。

        Args:
            sec_code: 証券コード。

        Returns:
            会社。

        Raises:
            EdinetMcpError: 対象外の証券コードのとき。
        """
        code = unicodedata.normalize("NFKC", sec_code).strip().upper()  # 英字入りの証券コードもある
        if len(code) == 5 and code.endswith("0"):  # 証券コードの5桁表記（末尾0）
            code = code[:4]
        company = self._companies.get(code)
        if company is None:
            known = ", ".join(self._companies)
            raise EdinetMcpError(f"証券コード {code!r} は対象外です。対象: {known}")
        return company

    # ---- 検索 ----

    def search_filings(self, query: str, sec_code: str, k: int = 5) -> list[Passage]:
        """会社の当期の有価証券報告書から、質問に近い本文を探す。

        Args:
            query: 質問文。
            sec_code: 証券コード。
            k: 返す件数。1 以上 MAX_K 以下。

        Returns:
            出典スパンつきの本文。出典スパンを作れないものは除く。

        Raises:
            EdinetMcpError: k が範囲外、質問が空、対象外の会社、
                または書類が索引に入っていないとき。
        """
        if not 1 <= k <= MAX_K:
            raise EdinetMcpError(f"k は 1 から {MAX_K} までです（{k}）")
        if not query.strip():
            raise EdinetMcpError("クエリが空です")
        company = self._company(sec_code)
        filing = company.filings.current
        if not self._searcher.is_indexed(filing.doc_id):
            raise EdinetMcpError(
                f"書類 {filing.doc_id} は検索の索引に入っていません（取り込みが必要）。"
                "空の結果と区別するため、エラーにしています"
            )
        hits = self._searcher.search(query, k, doc_ids=[filing.doc_id])
        pages = self._pages(filing.doc_id)
        passages: list[Passage] = []
        for hit in hits:
            try:
                spans = self._spans(hit, pages)
            except EdinetMcpError as e:
                # 出典スパンを作れない本文は、出典にできないので返さない（他の結果は返す）
                logger.warning("出典スパンを作れない検索結果を除きました: %s", e)
                continue
            passages.append(
                Passage(
                    sec_code=company.sec_code,
                    company=company.name,
                    doc_id=hit.chunk.doc_id,
                    period_end=filing.period_end,
                    page_start=hit.chunk.page_start,
                    page_end=hit.chunk.page_end,
                    heading_path=hit.chunk.heading_path,
                    text=hit.chunk.text,
                    spans=spans,
                    rank=hit.rank,
                    score=hit.score,
                )
            )
        return passages

    @staticmethod
    def _spans(hit: Hit, pages: list[str]) -> list[SourceSpan]:
        """検索結果のチャンクを、出典スパンにする。

        Args:
            hit: 検索結果。
            pages: 書類のページごとの本文。

        Returns:
            ページごとの出典スパン。

        Raises:
            EdinetMcpError: チャンクの本文をページ本文から見つけられないとき。
        """
        try:
            return locate_chunk(hit.chunk, pages)
        except SpanNotFoundError as e:
            raise EdinetMcpError(str(e)) from e

    def _pages(self, doc_id: str) -> list[str]:
        """書類のページごとの本文。一度読んだものは覚えておく。失敗は理由つきのエラーにする。

        Args:
            doc_id: 書類ID。

        Returns:
            ページごとの本文。

        Raises:
            EdinetMcpError: PDF が無い、読めない、またはページが0件のとき。
        """
        if doc_id in self._pages_cache:
            return self._pages_cache[doc_id]
        try:
            pages = self._load_pages(doc_id)
        except FileNotFoundError as e:
            raise EdinetMcpError(f"書類 {doc_id} の PDF がありません。先に取得してください") from e
        except Exception as e:  # PDF の解析器は様々な例外を出す。ファイルの読み込みの境界で受ける
            raise EdinetMcpError(
                f"書類 {doc_id} の PDF を読めません: {type(e).__name__}。取得し直してください"
            ) from e
        if not pages:
            raise EdinetMcpError(f"書類 {doc_id} の PDF の本文を読めません（ページが0件）")
        self._pages_cache[doc_id] = pages
        return pages

    # ---- 財務数値・比率（XBRL） ----

    def _facts(self, doc_id: str) -> list[Fact]:
        """書類の XBRL の数値の行を読む。

        Args:
            doc_id: 書類ID。

        Returns:
            数値の行のリスト。

        Raises:
            EdinetMcpError: 財務データが無い、または読めないとき。
        """
        try:
            return list(self._load_facts(doc_id))
        except FileNotFoundError as e:
            raise EdinetMcpError(
                f"書類 {doc_id} の財務データ（XBRL の CSV）がありません。先に取得してください"
            ) from e
        except (zipfile.BadZipFile, UnicodeError) as e:
            raise EdinetMcpError(
                f"書類 {doc_id} の財務データ（XBRL の CSV）を読めません: {type(e).__name__}。"
                "取得し直してください"
            ) from e

    def get_financials(self, sec_code: str, period: Period = "current") -> FinancialsResult:
        """連結の財務数値を XBRL から取る。前期の値も、当期の書類の前期の列から読む。

        Args:
            sec_code: 証券コード。
            period: current（当期）か previous（前期）。

        Returns:
            財務数値と、項目ごとの XBRL の項目名。当期は PDF のページも付く。

        Raises:
            EdinetMcpError: period が不正、対象外の会社、または財務データを読めないとき。
        """
        if period not in ("current", "previous"):
            raise EdinetMcpError(f"period は current か previous です（{period!r}）")
        company = self._company(sec_code)
        # 前期の値も、当期の書類の前期の列から読む
        doc_id = company.filings.current.doc_id
        extraction = self._extract(doc_id, self._facts(doc_id), period)
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
            pdf_pages=self._pdf_pages(doc_id, extraction.financials) if period == "current" else {},
        )

    def _pdf_pages(self, doc_id: str, financials: PeriodFinancials) -> dict[str, int]:
        """PDF の基準線が、XBRL と同じ値を読んだ項目のページ。読めない・値が違う項目は含めない。

        Args:
            doc_id: 書類ID。
            financials: XBRL から読んだ財務データ。

        Returns:
            項目名から PDF のページ（1始まり）への対応。失敗したときは空。
        """
        try:
            if doc_id not in self._pdf_cache:
                self._pdf_cache[doc_id] = self._pdf_items(self._pages(doc_id))
            items = self._pdf_cache[doc_id]
        except Exception as e:  # ページの付与は補助。失敗しても数値は返す
            logger.warning("PDF のページを付けられませんでした（%s）: %s", doc_id, e)
            return {}
        found: dict[str, int] = {}
        for name, item in items.items():
            value: Decimal | None = getattr(financials, name, None)
            if item.value is not None and item.page is not None and item.value == value:
                found[name] = item.page
        return found

    @staticmethod
    def _extract(doc_id: str, facts: list[Fact], period: Period) -> Extraction:
        """XBRL の数値の行から財務データを読む。失敗は理由つきのエラーにする。

        Args:
            doc_id: 書類ID。
            facts: 数値の行。
            period: 当期か前期か。

        Returns:
            読み取った財務データ。

        Raises:
            EdinetMcpError: IFRS の書類、または連結の行が無いとき。
        """
        try:
            return extract(facts, period)
        except UnsupportedAccountingStandard as e:
            raise EdinetMcpError(
                f"書類 {doc_id} は IFRS の連結財務諸表で、現在は読めません（J-GAAP のみ対応）"
            ) from e
        except NoConsolidatedStatements as e:
            raise EdinetMcpError(f"書類 {doc_id}: {e}") from e

    def get_ratios(self, sec_code: str, period: Period = "current") -> RatiosResult:
        """当期（前期との比較つき）または前期の財務比率。前期は前々期が無く、成長率は算定不能。

        Args:
            sec_code: 証券コード。
            period: current（当期）か previous（前期）。

        Returns:
            財務比率と留意事項の判定。

        Raises:
            EdinetMcpError: period が不正、対象外の会社、または財務データを読めないとき。
        """
        if period not in ("current", "previous"):
            raise EdinetMcpError(f"period は current か previous です（{period!r}）")
        company = self._company(sec_code)
        doc_id = company.filings.current.doc_id
        facts = self._facts(doc_id)  # 1回だけ読む（zip の展開と CSV の解析が重い）
        previous = self._extract(doc_id, facts, "previous").financials
        if period == "current":
            report = compute_ratios(self._extract(doc_id, facts, "current").financials, previous)
            filing = company.filings.current
        else:
            report = compute_ratios(previous, None)
            filing = company.filings.previous
        return RatiosResult(
            sec_code=company.sec_code,
            company=company.name,
            doc_id=doc_id,
            period=period,
            period_end=filing.period_end,
            ratios=report,
            policy_note=POLICY_NOTE,
        )

    # ---- ページ ----

    def all_pages(self, sec_code: str) -> list[str]:
        """当期の有価証券報告書の全ページの本文。語句の検索など、コードが全体を調べるのに使う。

        Args:
            sec_code: 証券コード。

        Returns:
            ページごとの本文。

        Raises:
            EdinetMcpError: 対象外の会社、または PDF を読めないとき。
        """
        return list(self._pages(self._company(sec_code).filings.current.doc_id))

    def get_page(self, sec_code: str, page: int) -> PageResult:
        """当期の有価証券報告書の1ページ分の本文（1 始まり）。検索の出典を確かめるのに使う。

        Args:
            sec_code: 証券コード。
            page: ページ番号（1始まり）。

        Returns:
            ページの本文と、全ページ数。

        Raises:
            EdinetMcpError: ページが範囲外、対象外の会社、または PDF を読めないとき。
        """
        company = self._company(sec_code)
        doc_id = company.filings.current.doc_id
        pages = self._pages(doc_id)
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
