from decimal import Decimal

import pytest

from evals.l1 import CompanyScore, Outcome, classify, render_report, score_company, summarize
from finance.ratios import PeriodFinancials
from ingest.pdf_baseline import ExtractedValue

D = Decimal


@pytest.mark.parametrize(
    ("extracted", "truth", "expected"),
    [
        (D(100), D(100), Outcome.MATCH),
        (D(100), D(101), Outcome.MISMATCH),
        (None, D(100), Outcome.MISSING),
        (D(-5), D(-5), Outcome.MATCH),
        (D(5), D(-5), Outcome.MISMATCH),  # 符号の取り違え
        # 正解が0（XBRL に「－」）または行なしで、PDF にも行が無いなら正解
        (None, D(0), Outcome.BOTH_ABSENT),
        (None, None, Outcome.BOTH_ABSENT),
        # PDF が「－」(0) と読んだ場合も、正解が0または行なしなら一致
        (D(0), D(0), Outcome.MATCH),
        (D(0), None, Outcome.MATCH),
        # 正解が無いのに値を読んだ
        (D(50), None, Outcome.SPURIOUS),
        (D(50), D(0), Outcome.SPURIOUS),
        # 正解があるのに0と読んだ
        (D(0), D(100), Outcome.MISMATCH),
    ],
)
def test_結果の分類(extracted: Decimal | None, truth: Decimal | None, expected: Outcome) -> None:
    assert classify(extracted, truth) == expected


def test_正解として数えるのは一致と両方なしだけ() -> None:
    assert Outcome.MATCH.correct and Outcome.BOTH_ABSENT.correct
    assert not any(o.correct for o in (Outcome.MISMATCH, Outcome.MISSING, Outcome.SPURIOUS))


def _items(**values: int | None) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    for name in (
        "total_assets", "net_assets", "current_assets", "current_liabilities", "net_sales",
        "operating_income", "interest_income", "dividend_income", "interest_expense",
        "net_income_attributable_to_owners", "depreciation",
    ):  # fmt: skip
        v = values.get(name)
        out[name] = (
            ExtractedValue(D(v), "BS", 1, "行")
            if v is not None
            else ExtractedValue(None, reason="label_not_found")
        )
    return out


def test_会社ごとに11項目を採点する() -> None:
    truth = PeriodFinancials(total_assets=D(100), net_assets=D(40), net_sales=D(500))
    score = score_company("9999", "サンプル", _items(total_assets=100, net_assets=41), truth)

    assert score.outcomes["total_assets"] == Outcome.MATCH
    assert score.outcomes["net_assets"] == Outcome.MISMATCH
    assert score.outcomes["net_sales"] == Outcome.MISSING
    assert score.outcomes["interest_income"] == Outcome.BOTH_ABSENT  # 正解も PDF も無い
    assert score.reasons["net_sales"] == "label_not_found"
    assert len(score.outcomes) == 11


def _score(code: str, **outcomes: Outcome) -> CompanyScore:
    base = {name: Outcome.MATCH for name in _items()}
    base.update(outcomes)
    return CompanyScore(code, f"社{code}", base, {})


def test_項目ごとの正答数を集計する() -> None:
    scores = [
        _score("1"),
        _score("2", total_assets=Outcome.MISMATCH),
        _score("3", total_assets=Outcome.MISSING, net_assets=Outcome.BOTH_ABSENT),
    ]
    summary = summarize(scores)

    assert summary.per_field["total_assets"].correct == 1
    assert summary.per_field["total_assets"].total == 3
    assert summary.per_field["net_assets"].correct == 3  # 両方なしも正解
    assert summary.correct == 33 - 2
    assert summary.total == 33


def test_レポートは数値を含まず_失敗の会社と項目と理由だけを書く() -> None:
    scores = [_score("1"), _score("2", total_assets=Outcome.MISSING)]
    scores[1] = CompanyScore("2", "社2", scores[1].outcomes, {"total_assets": "label_not_found"})

    md = render_report("pdfplumber（基準線）", scores)

    assert "pdfplumber（基準線）" in md
    assert "総資産" in md  # 日本語の項目名で表示する
    assert "社2" in md
    assert "label_not_found" in md
    assert "21/22" in md  # 全体の正答数
