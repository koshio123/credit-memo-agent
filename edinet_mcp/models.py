"""ツールが返すデータの型。MCP では、これが出力のスキーマになる。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finance.ratios import PeriodFinancials, RatioReport
from retrieval.citations import SourceSpan

Period = Literal["current", "previous"]


class CompanyInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    sec_code: str
    name: str
    industry: str
    doc_id: str  # 当期の有価証券報告書
    period_end: str
    previous_period_end: str


class Passage(BaseModel):
    """検索で見つかった本文。メモの出典（どの書類の何ページか）になる。"""

    model_config = ConfigDict(frozen=True)

    sec_code: str
    company: str
    doc_id: str
    period_end: str
    page_start: int  # 1 始まり
    page_end: int
    heading_path: list[str]
    text: str
    spans: list[SourceSpan]  # ページごとの出典（引用文は text と同じ内容。位置を検証できる）
    rank: int
    score: float  # 検索方式の内部の点数。順位の比較にだけ使う


class FinancialsResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    sec_code: str
    company: str
    doc_id: str  # 値を読んだ書類（前期の値も、当期の書類の前期の列から読む）
    period: Period
    period_end: str
    source: Literal["XBRL"]
    unit: Literal["円"]
    financials: PeriodFinancials  # None は「XBRL に項目が無かった」こと。0 とは区別される
    provenance: dict[str, str]  # 項目 -> 得た元の XBRL 項目名
    # 項目 -> PDF が同じ値を読んだページ（当期のみ。PDF が読めなかった項目、値が食い違う項目は無い）
    pdf_pages: dict[str, int] = Field(default_factory=dict[str, int])


class RatiosResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    sec_code: str
    company: str
    doc_id: str
    period: Period
    period_end: str
    ratios: RatioReport
    policy_note: str


class PageResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    sec_code: str
    company: str
    doc_id: str
    period_end: str
    page: int  # 1 始まり
    n_pages: int
    text: str
