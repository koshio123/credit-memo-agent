import zipfile
from decimal import Decimal
from pathlib import Path

import pytest

from evals.companies import Company, load_companies
from evals.ground_truth import build_company_ground_truth
from finance.ratios import Level
from ingest.xbrl_facts import UnsupportedAccountingStandard

HEADER = [
    "要素ID",
    "項目名",
    "コンテキストID",
    "相対年度",
    "連結・個別",
    "期間・時点",
    "ユニットID",
    "単位",
    "値",
]


def _write_csv_zip(data_dir: Path, doc_id: str, rows: dict[tuple[str, str], str]) -> None:
    """rows: (項目名, コンテキストID) -> 値。連結の jppfs_cor の行として書く。"""
    lines = [HEADER] + [
        [f"jppfs_cor:{el}", "", ctx, "", "連結", "", "JPY", "円", value]
        for (el, ctx), value in rows.items()
    ]
    text = "\n".join("\t".join(row) for row in lines) + "\n"
    path = data_dir / doc_id / f"{doc_id}.csv.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("XBRL_TO_CSV/jpcrp030000-asr-001_E00000-000.csv", text.encode("utf-16"))


@pytest.fixture
def company() -> Company:
    return load_companies()[0]


def _full(assets: int, net_assets: int, sales: int) -> dict[tuple[str, str], str]:
    """自己資本比率・営業利益率が計算できる最小限の値（当期の列と前期の列）。"""
    return {
        ("Assets", "CurrentYearInstant"): str(assets),
        ("NetAssets", "CurrentYearInstant"): str(net_assets),
        ("NetSales", "CurrentYearDuration"): str(sales),
        ("OperatingIncome", "CurrentYearDuration"): str(sales // 10),
        ("Assets", "Prior1YearInstant"): str(assets - 100),
        ("NetAssets", "Prior1YearInstant"): str(net_assets - 50),
        ("NetSales", "Prior1YearDuration"): str(sales - 1000),
        ("OperatingIncome", "Prior1YearDuration"): str((sales - 1000) // 10),
    }


def test_2期分の財務データと比率を作る(company: Company, tmp_path: Path) -> None:
    cur_doc, prev_doc = company.filings.current.doc_id, company.filings.previous.doc_id
    _write_csv_zip(tmp_path, cur_doc, _full(10_000, 4_000, 20_000))
    # 前期の書類の当期列 = 当期の書類の前期列（一致する場合）
    _write_csv_zip(
        tmp_path,
        prev_doc,
        {
            ("Assets", "CurrentYearInstant"): "9900",
            ("NetAssets", "CurrentYearInstant"): "3950",
            ("NetSales", "CurrentYearDuration"): "19000",
            ("OperatingIncome", "CurrentYearDuration"): "1900",
        },
    )

    gt = build_company_ground_truth(company, tmp_path)

    assert [p.period_end for p in gt.periods] == [
        company.filings.previous.period_end,
        company.filings.current.period_end,
    ]
    current = gt.periods[1]
    assert current.financials.total_assets == Decimal(10_000)
    assert current.ratios.equity_ratio.value == Decimal(40)
    assert current.ratios.equity_ratio.level == Level.STANDARD
    # 当期の売上高成長率は、書類内の前期列から計算する
    assert current.ratios.sales_growth.value is not None
    assert current.provenance["total_assets"] == "Assets"
    assert gt.mismatches == []


def test_前期の書類と食い違えば_訂正再表示として記録する(company: Company, tmp_path: Path) -> None:
    _write_csv_zip(tmp_path, company.filings.current.doc_id, _full(10_000, 4_000, 20_000))
    _write_csv_zip(
        tmp_path,
        company.filings.previous.doc_id,
        {
            ("Assets", "CurrentYearInstant"): "9000",  # 当期書類の前期列(9,900)と違う
            ("NetAssets", "CurrentYearInstant"): "3950",
            ("NetSales", "CurrentYearDuration"): "19000",
            ("OperatingIncome", "CurrentYearDuration"): "1900",
        },
    )

    gt = build_company_ground_truth(company, tmp_path)

    assert any("total_assets" in m for m in gt.mismatches)


def test_IFRSの書類は失敗させる(company: Company, tmp_path: Path) -> None:
    path = tmp_path / company.filings.current.doc_id / f"{company.filings.current.doc_id}.csv.zip"
    path.parent.mkdir(parents=True)
    header = "\t".join(HEADER)
    row = "\t".join(
        ["jpigp_cor:AssetsIFRS", "", "CurrentYearInstant", "", "その他", "", "JPY", "円", "1"]
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "XBRL_TO_CSV/jpcrp030000-asr-001_E00000-000.csv", f"{header}\n{row}\n".encode("utf-16")
        )
    _write_csv_zip(
        tmp_path, company.filings.previous.doc_id, {("Assets", "CurrentYearInstant"): "1"}
    )

    with pytest.raises(UnsupportedAccountingStandard):
        build_company_ground_truth(company, tmp_path)


def _mismatch_case(
    company: Company, tmp_path: Path, prior_column: str | None, prior_filing: str | None
):
    """当期書類の前期列と、前期書類の当期列で、短期借入金だけが異なるケースを作る。"""
    base = {
        ("Assets", "CurrentYearInstant"): "10000",
        ("NetAssets", "CurrentYearInstant"): "4000",
        ("Assets", "Prior1YearInstant"): "9900",
        ("NetAssets", "Prior1YearInstant"): "3950",
    }
    cur = dict(base)
    if prior_column is not None:
        cur[("ShortTermLoansPayable", "Prior1YearInstant")] = prior_column
    _write_csv_zip(tmp_path, company.filings.current.doc_id, cur)
    prev = {("Assets", "CurrentYearInstant"): "9900", ("NetAssets", "CurrentYearInstant"): "3950"}
    if prior_filing is not None:
        prev[("ShortTermLoansPayable", "CurrentYearInstant")] = prior_filing
    _write_csv_zip(tmp_path, company.filings.previous.doc_id, prev)
    return build_company_ground_truth(company, tmp_path)


def test_ゼロと行なしは表示の違いにすぎず_食い違いとして扱わない(
    company: Company, tmp_path: Path
) -> None:
    # 当期書類の前期列は「－」(0)、前期書類にはその行が無い。意味は同じ
    gt = _mismatch_case(company, tmp_path, prior_column="－", prior_filing=None)
    assert gt.mismatches == []


def test_ゼロと0の行は一致(company: Company, tmp_path: Path) -> None:
    assert _mismatch_case(company, tmp_path, prior_column="0", prior_filing="0").mismatches == []


def test_値が違えば食い違い(company: Company, tmp_path: Path) -> None:
    gt = _mismatch_case(company, tmp_path, prior_column="500", prior_filing="700")
    assert any("short_term_borrowings" in m for m in gt.mismatches)


def test_ゼロでない値と行なしは食い違い(company: Company, tmp_path: Path) -> None:
    gt = _mismatch_case(company, tmp_path, prior_column="500", prior_filing=None)
    assert any("short_term_borrowings" in m for m in gt.mismatches)
