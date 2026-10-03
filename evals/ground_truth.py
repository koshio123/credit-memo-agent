"""L1 評価の正解データ: XBRL（財務データ CSV）から、各社の 2 期分の財務データと財務比率を作る。

PDF から抽出した値は、ここで作る値と突き合わせて正答率を出す。

- 当期は、当期の書類の「当期の列」、前期は、当期の書類の「前期の列」から読む。
- 前期の書類の「当期の列」と食い違えば、訂正再表示の可能性として mismatches に記録する。
"""

from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.companies import Company
from finance.ratios import PeriodFinancials, RatioReport, compute_ratios
from ingest.xbrl_facts import Extraction, extract, read_facts


class PeriodGroundTruth(BaseModel):
    model_config = ConfigDict(frozen=True)

    period_end: str
    financials: PeriodFinancials
    provenance: dict[str, str]  # 項目 -> 得た元の XBRL 項目名
    ratios: RatioReport


class CompanyGroundTruth(BaseModel):
    model_config = ConfigDict(frozen=True)

    sec_code: str
    name: str
    periods: list[PeriodGroundTruth]  # 前期、当期の順
    mismatches: list[str]  # 訂正再表示の疑い（前期の列と、前期の書類の当期の列の食い違い）


def _same(a: Decimal | None, b: Decimal | None) -> bool:
    """同じ値か。0 と「行なし」は、表示の違いにすぎないので同じとみなす。

    片方の書類でだけ「－」（0）と載り、もう片方には行が無いことがある。
    ゼロでない値と「行なし」、値の違いは、訂正再表示の疑いとして食い違いにする。

    Args:
        a: 一方の値。
        b: もう一方の値。

    Returns:
        同じとみなせれば True。
    """
    if a == b:
        return True
    return a in (None, Decimal(0)) and b in (None, Decimal(0))


def _csv_zip(data_dir: Path, doc_id: str) -> Path:
    """財務データ CSV の zip のパスを返す。

    Args:
        data_dir: EDINET の書類を置いたディレクトリ。
        doc_id: 書類ID。

    Returns:
        zip のパス。
    """
    return data_dir / doc_id / f"{doc_id}.csv.zip"


def _period(
    period_end: str, extraction: Extraction, previous: PeriodFinancials | None
) -> PeriodGroundTruth:
    """1期分の正解データを組み立てる。

    Args:
        period_end: 期末。
        extraction: XBRL から読んだ財務データ。
        previous: 比率の計算に使う前期の財務データ。無ければ None。

    Returns:
        財務データと、それから計算した比率を含む正解データ。
    """
    return PeriodGroundTruth(
        period_end=period_end,
        financials=extraction.financials,
        provenance=extraction.provenance,
        ratios=compute_ratios(extraction.financials, previous),
    )


def build_company_ground_truth(company: Company, data_dir: Path) -> CompanyGroundTruth:
    """会社の正解データ（前期・当期）を XBRL から作る。

    当期の書類の前期列と、前期の書類の当期列を突き合わせ、食い違いを記録する。

    Args:
        company: 会社。
        data_dir: EDINET の書類を置いたディレクトリ。

    Returns:
        前期・当期の正解データと、訂正再表示の疑いのある項目。

    Raises:
        UnsupportedAccountingStandard: IFRS の書類のとき（黙って空にしない）。
    """
    cur_facts = read_facts(_csv_zip(data_dir, company.filings.current.doc_id))
    prev_facts = read_facts(_csv_zip(data_dir, company.filings.previous.doc_id))

    current = extract(cur_facts, "current")
    previous = extract(cur_facts, "previous")  # 当期の書類の前期列
    previous_filing = extract(prev_facts, "current")  # 前期の書類の当期列

    mismatches = [
        f"{name}: 当期書類の前期列={getattr(previous.financials, name)}"
        f" / 前期書類の当期列={getattr(previous_filing.financials, name)}"
        for name in PeriodFinancials.model_fields
        if not _same(getattr(previous.financials, name), getattr(previous_filing.financials, name))
    ]
    return CompanyGroundTruth(
        sec_code=company.sec_code,
        name=company.name,
        periods=[
            _period(company.filings.previous.period_end, previous, None),
            _period(company.filings.current.period_end, current, previous.financials),
        ],
        mismatches=mismatches,
    )
