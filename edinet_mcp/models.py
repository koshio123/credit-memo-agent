"""ツールが返すデータの型。MCP では、これが出力のスキーマになる。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from finance.ratios import PeriodFinancials, RatioReport

Period = Literal["current", "previous"]


class CompanyInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    sec_code: str
    name: str
    industry: str
    doc_id: str  # 当期の有価証券報告書
    period_end: str


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


class RatiosResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    sec_code: str
    company: str
    doc_id: str
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
