import pytest

from evals.companies import Company, load_companies


@pytest.fixture(scope="module")
def companies() -> list[Company]:
    return load_companies()


def test_10社で重複がない(companies: list[Company]) -> None:
    assert len(companies) == 10
    assert len({c.sec_code for c in companies}) == 10
    assert len({c.edinet_code for c in companies}) == 10


def test_業種が分散している(companies: list[Company]) -> None:
    assert len({c.industry for c in companies}) == 10


def test_決算期が3月と12月の両方を含む(companies: list[Company]) -> None:
    assert {c.fiscal_year_end for c in companies} == {"3月31日", "12月31日"}


def test_金融業を含まない(companies: list[Company]) -> None:
    banned = {"銀行業", "保険業", "証券、商品先物取引業", "その他金融業"}
    assert not banned & {c.industry for c in companies}


def test_直近期と前期の有報がちょうど1年違いでそろう(companies: list[Company]) -> None:
    for c in companies:
        cur, prev = c.filings.current, c.filings.previous
        assert cur.doc_id != prev.doc_id, c.name
        assert int(cur.period_end[:4]) - int(prev.period_end[:4]) == 1, c.name
        assert cur.period_end[5:] == prev.period_end[5:], c.name  # 同じ決算日


def test_水準がばらつくよう選んでいる(companies: list[Company]) -> None:
    snaps = [c.selection_snapshot for c in companies]
    assert any(s.equity_ratio_pct < 20 for s in snaps), "自己資本比率が要精査水準"
    assert any(s.current_ratio_pct < 100 for s in snaps), "流動比率が要精査水準"
    assert any(s.operating_margin_pct < 0 for s in snaps), "営業赤字"
    assert any(1 <= s.operating_margin_pct < 3 for s in snaps), "営業利益率が留意水準"
    assert any(
        s.equity_ratio_pct >= 30 and s.current_ratio_pct >= 120 and s.operating_margin_pct >= 3
        for s in snaps
    ), "3指標とも標準水準"
