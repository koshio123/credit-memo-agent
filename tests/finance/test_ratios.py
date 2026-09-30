"""与信管理規程（policies/credit_policy.md）の附則と第8・10・11条の仕様を、そのままテストにしたもの。"""

from decimal import Decimal

import pytest

from finance.ratios import Level, PeriodFinancials, compute_ratios

D = Decimal


def _fin(**kw: int | str | None) -> PeriodFinancials:
    """既定は、すべての指標が算定できる健全な数値。上書きしたい項目だけ渡す。"""
    base: dict[str, int | str | None] = {
        "total_assets": 10_000,
        "net_assets": 4_000,
        "current_assets": 6_000,
        "current_liabilities": 3_000,
        "net_sales": 20_000,
        "operating_income": 1_000,
        "interest_income": 10,
        "dividend_income": 40,
        "interest_expense": 50,
        "net_income_attributable_to_owners": 600,
        "depreciation": 400,
        "trade_receivables": 2_000,
        "inventories": 1_000,
        "trade_payables": 1_500,
        "short_term_borrowings": 0,
        "commercial_paper": 0,
        "current_portion_long_term_borrowings": 0,
        "current_portion_bonds": 0,
        "bonds": 0,
        "long_term_borrowings": 0,
        "lease_obligations": 0,
    }
    base.update(kw)
    return PeriodFinancials(**{k: (None if v is None else D(v)) for k, v in base.items()})


def _report(current: PeriodFinancials | None = None, previous: PeriodFinancials | None = None):
    return compute_ratios(current or _fin(), previous)


# ---- 自己資本比率 = 純資産 ÷ 総資産（標準30%以上 / 留意20%以上30%未満 / 要精査20%未満）


@pytest.mark.parametrize(
    ("net_assets", "expected_level"),
    [
        (3_000, Level.STANDARD),  # ちょうど30% は標準
        (2_999, Level.CAUTION),
        (2_000, Level.CAUTION),  # ちょうど20% は留意
        (1_999, Level.SCRUTINY),
    ],
)
def test_自己資本比率の境界(net_assets: int, expected_level: Level) -> None:
    r = _report(_fin(net_assets=net_assets)).equity_ratio
    assert r.level == expected_level


def test_自己資本比率の値と計算根拠() -> None:
    r = _report(_fin(net_assets=4_100, total_assets=10_000)).equity_ratio
    assert r.value == D("41")
    assert r.unit == "%"
    # 規程 第7条: 算式の入力値と算定過程を併記する
    assert "純資産" in r.basis
    assert "4,100" in r.basis
    assert "10,000" in r.basis


def test_債務超過は負の値で要精査() -> None:
    r = _report(_fin(net_assets=-500)).equity_ratio
    assert r.value == D("-5")
    assert r.level == Level.SCRUTINY


@pytest.mark.parametrize("total_assets", [0, -1])
def test_総資産がゼロ以下なら算定不能(total_assets: int) -> None:
    r = _report(_fin(total_assets=total_assets)).equity_ratio
    assert r.value is None
    assert r.level is None
    assert r.unmeasurable_reason


# ---- 流動比率 = 流動資産 ÷ 流動負債（120%以上 / 100%以上120%未満 / 100%未満）


@pytest.mark.parametrize(
    ("current_assets", "expected_level"),
    [
        (3_600, Level.STANDARD),  # ちょうど120%
        (3_599, Level.CAUTION),
        (3_000, Level.CAUTION),  # ちょうど100%
        (2_999, Level.SCRUTINY),
    ],
)
def test_流動比率の境界(current_assets: int, expected_level: Level) -> None:
    r = _report(_fin(current_assets=current_assets, current_liabilities=3_000)).current_ratio
    assert r.level == expected_level


def test_流動負債がゼロなら算定不能() -> None:
    r = _report(_fin(current_liabilities=0)).current_ratio
    assert r.value is None
    assert r.level is None


# ---- 営業利益率 = 営業利益 ÷ 売上高（3%以上 / 1%以上3%未満 / 1%未満）


@pytest.mark.parametrize(
    ("operating_income", "expected_level"),
    [
        (600, Level.STANDARD),  # 売上20,000 の 3%
        (599, Level.CAUTION),
        (200, Level.CAUTION),  # 1%
        (199, Level.SCRUTINY),
        (-100, Level.SCRUTINY),  # 営業損失
    ],
)
def test_営業利益率の境界(operating_income: int, expected_level: Level) -> None:
    r = _report(_fin(operating_income=operating_income, net_sales=20_000)).operating_margin
    assert r.level == expected_level


def test_売上高がゼロなら営業利益率は算定不能() -> None:
    assert _report(_fin(net_sales=0)).operating_margin.value is None


# ---- インタレスト・カバレッジ・レシオ = (営業利益+受取利息+受取配当金) ÷ 支払利息
#      （5倍以上 / 2倍以上5倍未満 / 2倍未満）


@pytest.mark.parametrize(
    ("interest_expense", "expected_level"),
    [
        (200, Level.STANDARD),  # (1000+10+40)/200 = 5.25
        (210, Level.STANDARD),  # ちょうど5倍
        (211, Level.CAUTION),
        (525, Level.CAUTION),  # ちょうど2倍
        (526, Level.SCRUTINY),
    ],
)
def test_インタレスト_カバレッジの境界(interest_expense: int, expected_level: Level) -> None:
    r = _report(_fin(interest_expense=interest_expense)).interest_coverage
    assert r.level == expected_level


def test_インタレスト_カバレッジは営業利益と受取利息配当金の合計を使う() -> None:
    r = _report(
        _fin(operating_income=1_000, interest_income=10, dividend_income=40, interest_expense=100)
    ).interest_coverage
    assert r.value == D("10.5")
    assert r.unit == "倍"
    assert "受取利息" in r.basis
    assert "受取配当金" in r.basis


def test_支払利息がゼロなら算定不能で支払利息なしと付記する() -> None:
    r = _report(_fin(interest_expense=0)).interest_coverage
    assert r.value is None
    assert r.level is None  # 第8条 第2項: 算定不能の指標は水準を判定しない
    assert "支払利息なし" in (r.unmeasurable_reason or "")


# ---- 債務償還年数 = 要償還債務 ÷ 償還原資


def test_有利子負債はリース債務を含む7項目の合計() -> None:
    fin = _fin(
        short_term_borrowings=1,
        commercial_paper=2,
        current_portion_long_term_borrowings=4,
        current_portion_bonds=8,
        bonds=16,
        long_term_borrowings=32,
        lease_obligations=64,
    )
    assert fin.interest_bearing_debt == D(127)


def test_債務償還年数の算式() -> None:
    # 要償還債務 = 有利子負債 - 正常運転資金 = 8,000 - (2,000+1,000-1,500) = 6,500
    # 償還原資   = 親会社株主に帰属する当期純利益 + 減価償却費 = 600 + 400 = 1,000
    fin = _fin(long_term_borrowings=8_000)
    r = _report(fin).debt_repayment_years
    assert r.value == D("6.5")
    assert r.unit == "年"
    assert r.level == Level.STANDARD
    assert "有利子負債" in r.basis
    assert "正常運転資金" in r.basis


@pytest.mark.parametrize(
    ("long_term_borrowings", "expected_level"),
    [
        (11_500, Level.STANDARD),  # 要償還債務10,000 → ちょうど10年
        (11_501, Level.CAUTION),
        (16_500, Level.CAUTION),  # 要償還債務15,000 → ちょうど15年
        (16_501, Level.SCRUTINY),
    ],
)
def test_債務償還年数の境界(long_term_borrowings: int, expected_level: Level) -> None:
    r = _report(_fin(long_term_borrowings=long_term_borrowings)).debt_repayment_years
    assert r.level == expected_level


def test_要償還債務がゼロ以下なら償還原資にかかわらず0年() -> None:
    # 有利子負債0 - 正常運転資金1,500 = -1,500（実質無借金）。償還原資が負でも0年
    fin = _fin(net_income_attributable_to_owners=-900, depreciation=100)
    r = _report(fin).debt_repayment_years
    assert r.value == D(0)
    assert r.level == Level.STANDARD


def test_償還原資がゼロ以下なら算定不能で要精査() -> None:
    fin = _fin(long_term_borrowings=8_000, net_income_attributable_to_owners=-400, depreciation=400)
    r = _report(fin).debt_repayment_years
    assert r.value is None
    assert r.unmeasurable_reason
    # 第8条 第2項 ただし書き: 債務償還年数の算定不能は要精査水準として扱う
    assert r.level == Level.SCRUTINY


# ---- 売上高成長率 = 当期売上高 ÷ 前期売上高 - 1


def test_売上高成長率() -> None:
    r = _report(_fin(net_sales=11_000), _fin(net_sales=10_000)).sales_growth
    assert r.value == D("10")
    assert r.unit == "%"
    assert r.level is None  # 成長率には水準の区分がない（第11条の留意事項だけ）


def test_前期がなければ売上高成長率は算定不能() -> None:
    r = _report(_fin(), None).sales_growth
    assert r.value is None
    assert "前期" in (r.unmeasurable_reason or "")


def test_前期売上高がゼロ以下なら算定不能() -> None:
    assert _report(_fin(), _fin(net_sales=0)).sales_growth.value is None


# ---- 規程 第10条・第11条


@pytest.mark.parametrize(
    ("growth_sales", "expected"),
    [(9_000, True), (9_001, False)],  # 前期10,000 に対し -10% ちょうどは該当
)
def test_売上高の10パーセント以上の減少_第11条(growth_sales: int, expected: bool) -> None:
    flag = _report(_fin(net_sales=growth_sales), _fin(net_sales=10_000)).sales_drop
    assert flag.applies is expected


def test_営業損失の2期連続_第10条() -> None:
    assert (
        _report(
            _fin(operating_income=-1), _fin(operating_income=-1)
        ).consecutive_operating_loss.applies
        is True
    )
    assert (
        _report(
            _fin(operating_income=-1), _fin(operating_income=1)
        ).consecutive_operating_loss.applies
        is False
    )
    assert (
        _report(
            _fin(operating_income=1), _fin(operating_income=-1)
        ).consecutive_operating_loss.applies
        is False
    )
    assert (
        _report(
            _fin(operating_income=0), _fin(operating_income=-1)
        ).consecutive_operating_loss.applies
        is False
    )


def test_前期がなければ第10条と第11条は判定できない() -> None:
    report = _report(_fin(), None)
    assert report.consecutive_operating_loss.applies is None
    assert report.sales_drop.applies is None


# ---- 入力が無い（XBRLに項目が無い）場合は0とみなさず算定不能にする


def test_入力が無い項目は算定不能にする() -> None:
    r = _report(_fin(net_assets=None)).equity_ratio
    assert r.value is None
    assert "純資産" in (r.unmeasurable_reason or "")


def test_有利子負債の内訳が未計上なら0として扱う() -> None:
    # 借入金・社債などは、無い会社では XBRL に項目自体が出ない。未計上は0として合計する
    fin = _fin(commercial_paper=None, bonds=None, lease_obligations=None, long_term_borrowings=100)
    assert fin.interest_bearing_debt == D(100)


# ---- コードレビューでの指摘への対応 ----


def test_有利子負債の内訳がすべて未計上なら抽出漏れの可能性として算定不能にする() -> None:
    # 全部 None を「借入金ゼロ」とみなすと、抽出に失敗した会社が「0年・標準」という最良の評価になる
    fin = _fin(
        short_term_borrowings=None,
        commercial_paper=None,
        current_portion_long_term_borrowings=None,
        current_portion_bonds=None,
        bonds=None,
        long_term_borrowings=None,
        lease_obligations=None,
    )
    assert fin.interest_bearing_debt is None

    r = _report(fin).debt_repayment_years
    assert r.value is None
    assert r.level is None  # 抽出漏れを、財務の良し悪しの評価にしない
    assert "有利子負債" in (r.unmeasurable_reason or "")


def test_支払利息が無いときは断定せず_確認が必要だと書く() -> None:
    r = _report(_fin(interest_expense=None)).interest_coverage
    assert r.value is None
    reason = r.unmeasurable_reason or ""
    assert "支払利息" in reason
    assert "支払利息なし" not in reason  # ゼロと断定する文言は使わない
    assert "確認" in reason


def test_表示が境界に丸められても判定と食い違わない_自己資本比率() -> None:
    # 29.96% は留意。1桁に丸めて「30.0%」と表示すると、標準と読める
    r = _report(_fin(net_assets=2_996, total_assets=10_000)).equity_ratio
    assert r.level == Level.CAUTION
    assert "29.96%" in r.basis
    assert "30.0%" not in r.basis


def test_表示が境界に丸められても判定と食い違わない_債務償還年数() -> None:
    # 要償還債務10,004 ÷ 償還原資1,000 = 10.004年 は留意。「10.0年」と表示すると標準と読める
    r = _report(_fin(long_term_borrowings=11_504)).debt_repayment_years
    assert r.level == Level.CAUTION
    assert "10.004年" in r.basis
    assert "10.0年" not in r.basis


def test_通常の値は小数1桁で表示する() -> None:
    r = _report(_fin(net_assets=4_100, total_assets=10_000)).equity_ratio
    assert "41.0%" in r.basis


def test_表示が境界に丸められても判定と食い違わない_売上高の減少() -> None:
    # 前期10,000 → 当期9,004 は -9.96%。第11条は該当しない。「-10.0%」と表示すると該当と読める
    report = _report(_fin(net_sales=9_004), _fin(net_sales=10_000))
    assert report.sales_drop.applies is False
    assert "-9.96%" in report.sales_growth.basis
    assert "-10.0%" not in report.sales_growth.basis
    assert "-10.0%" not in report.sales_drop.basis
