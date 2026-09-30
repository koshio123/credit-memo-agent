"""財務比率の計算。与信管理規程（policies/credit_policy.md）の附則・第8条・第10条・第11条の実装。

財務比率は言語モデルに計算させず、必ずここで計算する（CLAUDE.md）。値は Decimal で持ち、
「30%ちょうど」のような境界で浮動小数点の誤差が出ないようにする。

規程との対応:
- 算式は附則第2条、算定不能の扱いは附則第3条、水準の区分は第8条。
- 算式の入力値と算定過程は、第7条に従い basis（計算根拠）に併記する。
"""

from collections.abc import Callable
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

ZERO = Decimal(0)
HUNDRED = Decimal(100)


class Level(StrEnum):
    """規程 第8条の水準。"""

    STANDARD = "標準"
    CAUTION = "留意"
    SCRUTINY = "要精査"


class PeriodFinancials(BaseModel):
    """1期分の連結財務データ（単位: 円）。None は「XBRLに項目が無かった」ことを表す。

    項目が無いことと、値がゼロであることは区別する。無い項目を0とみなして比率を計算すると、
    もっともらしい誤った数値が出るため、比率は「入力なし」として算定不能にする。
    例外は有利子負債の内訳と受取利息・受取配当金: 無い会社では項目自体が出ないので、0として扱う。
    ただし有利子負債の内訳がすべて無い場合は、抽出漏れと区別できないため算定不能にする。
    """

    model_config = ConfigDict(frozen=True)

    # 貸借対照表
    total_assets: Decimal | None = None  # 総資産
    net_assets: Decimal | None = None  # 純資産
    current_assets: Decimal | None = None  # 流動資産
    current_liabilities: Decimal | None = None  # 流動負債
    trade_receivables: Decimal | None = None  # 売上債権
    inventories: Decimal | None = None  # 棚卸資産
    trade_payables: Decimal | None = None  # 仕入債務
    # 有利子負債の内訳（附則第1条）
    short_term_borrowings: Decimal | None = None  # 短期借入金
    commercial_paper: Decimal | None = None  # コマーシャル・ペーパー
    current_portion_long_term_borrowings: Decimal | None = None  # 1年内返済予定の長期借入金
    current_portion_bonds: Decimal | None = None  # 1年内償還予定の社債
    bonds: Decimal | None = None  # 社債
    long_term_borrowings: Decimal | None = None  # 長期借入金
    lease_obligations: Decimal | None = None  # リース債務
    # 損益計算書
    net_sales: Decimal | None = None  # 売上高
    operating_income: Decimal | None = None  # 営業利益
    interest_income: Decimal | None = None  # 受取利息
    dividend_income: Decimal | None = None  # 受取配当金
    interest_expense: Decimal | None = None  # 支払利息
    net_income_attributable_to_owners: Decimal | None = None  # 親会社株主に帰属する当期純利益
    depreciation: Decimal | None = None  # 減価償却費

    @property
    def interest_bearing_debt(self) -> Decimal | None:
        """有利子負債（附則第1条第1項）。

        一部の内訳が未計上なら0として合計する。**すべて未計上なら None**（算定できない）。
        すべて0とみなすと、抽出に失敗した会社が「無借金」の最良の評価になってしまうため。
        本当に借入のない会社は、呼び出し側が0を明示する。
        """
        parts = (
            self.short_term_borrowings,
            self.commercial_paper,
            self.current_portion_long_term_borrowings,
            self.current_portion_bonds,
            self.bonds,
            self.long_term_borrowings,
            self.lease_obligations,
        )
        present = [p for p in parts if p is not None]
        return sum(present, ZERO) if present else None


class RatioResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    unit: str
    value: Decimal | None  # None は算定不能
    level: Level | None  # 算定不能、または水準の区分がない指標は None
    basis: str  # 計算根拠（算式と入力値）。規程 第7条
    unmeasurable_reason: str | None = None


class FlagResult(BaseModel):
    """規程 第10条・第11条の留意・要精査事項に該当するか。"""

    model_config = ConfigDict(frozen=True)

    name: str
    clause: str
    applies: bool | None  # None は判定できない（前期のデータがない、など）
    basis: str


class RatioReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    equity_ratio: RatioResult
    current_ratio: RatioResult
    operating_margin: RatioResult
    interest_coverage: RatioResult
    debt_repayment_years: RatioResult
    sales_growth: RatioResult
    consecutive_operating_loss: FlagResult
    sales_drop: FlagResult


# ---- 表示 ----


def _yen(x: Decimal) -> str:
    return f"{x:,.0f}"


def _show(value: Decimal, classify: Callable[[Decimal], object] | None = None) -> str:
    """値を、小数1桁から始めて、丸めても判定（classify）が変わらない最小の桁数で表示する。

    29.96% を「30.0%」と丸めると、留意なのに標準と読める。判定が変わる場合は桁を増やす。
    """
    for digits in range(1, 9):
        rounded = value.quantize(Decimal(1).scaleb(-digits))
        if classify is None or classify(rounded) == classify(value):
            return f"{rounded:,.{digits}f}"
    return f"{value:,.8f}"


# ---- 水準の判定（第8条） ----


def _level_at_least(value: Decimal, standard: str, caution: str) -> Level:
    """値が大きいほど良い指標。標準は standard 以上、留意は caution 以上。"""
    if value >= Decimal(standard):
        return Level.STANDARD
    if value >= Decimal(caution):
        return Level.CAUTION
    return Level.SCRUTINY


def _level_at_most(value: Decimal, standard: str, caution: str) -> Level:
    """値が小さいほど良い指標。標準は standard 以下、留意は caution 以下。"""
    if value <= Decimal(standard):
        return Level.STANDARD
    if value <= Decimal(caution):
        return Level.CAUTION
    return Level.SCRUTINY


def _at_least(standard: str, caution: str) -> Callable[[Decimal], Level]:
    return lambda v: _level_at_least(v, standard, caution)


def _at_most(standard: str, caution: str) -> Callable[[Decimal], Level]:
    return lambda v: _level_at_most(v, standard, caution)


def _is_sales_drop(growth_pct: Decimal) -> bool:
    """第11条: 売上高成長率が－10%以下。"""
    return growth_pct <= Decimal(-10)


# ---- 算定不能の組み立て ----


def _missing(items: dict[str, Decimal | None]) -> list[str]:
    return [label for label, v in items.items() if v is None]


def _unmeasurable(
    name: str, unit: str, reason: str, basis: str = "", level: Level | None = None
) -> RatioResult:
    return RatioResult(
        name=name,
        unit=unit,
        value=None,
        level=level,
        basis=basis or f"算定不能（{reason}）",
        unmeasurable_reason=reason,
    )


def _no_input(name: str, unit: str, labels: list[str]) -> RatioResult:
    return _unmeasurable(name, unit, "入力なし: " + "、".join(labels))


# ---- 各指標（附則第2条） ----


def _equity_ratio(f: PeriodFinancials) -> RatioResult:
    name, unit = "自己資本比率", "%"
    if f.net_assets is None or f.total_assets is None:
        return _no_input(name, unit, _missing({"純資産": f.net_assets, "総資産": f.total_assets}))
    if f.total_assets <= ZERO:
        return _unmeasurable(name, unit, f"総資産がゼロ以下（{_yen(f.total_assets)}）")
    value = f.net_assets / f.total_assets * HUNDRED
    level_of = _at_least("30", "20")
    return RatioResult(
        name=name,
        unit=unit,
        value=value,
        level=level_of(value),
        basis=(
            f"純資産 {_yen(f.net_assets)} ÷ 総資産 {_yen(f.total_assets)}"
            f" = {_show(value, level_of)}%"
        ),
    )


def _current_ratio(f: PeriodFinancials) -> RatioResult:
    name, unit = "流動比率", "%"
    if f.current_assets is None or f.current_liabilities is None:
        gaps = _missing({"流動資産": f.current_assets, "流動負債": f.current_liabilities})
        return _no_input(name, unit, gaps)
    if f.current_liabilities <= ZERO:
        return _unmeasurable(name, unit, f"流動負債がゼロ以下（{_yen(f.current_liabilities)}）")
    value = f.current_assets / f.current_liabilities * HUNDRED
    level_of = _at_least("120", "100")
    return RatioResult(
        name=name,
        unit=unit,
        value=value,
        level=level_of(value),
        basis=(
            f"流動資産 {_yen(f.current_assets)} ÷ 流動負債 {_yen(f.current_liabilities)}"
            f" = {_show(value, level_of)}%"
        ),
    )


def _operating_margin(f: PeriodFinancials) -> RatioResult:
    name, unit = "営業利益率", "%"
    if f.operating_income is None or f.net_sales is None:
        return _no_input(
            name, unit, _missing({"営業利益": f.operating_income, "売上高": f.net_sales})
        )
    if f.net_sales <= ZERO:
        return _unmeasurable(name, unit, f"売上高がゼロ以下（{_yen(f.net_sales)}）")
    value = f.operating_income / f.net_sales * HUNDRED
    level_of = _at_least("3", "1")
    return RatioResult(
        name=name,
        unit=unit,
        value=value,
        level=level_of(value),
        basis=(
            f"営業利益 {_yen(f.operating_income)} ÷ 売上高 {_yen(f.net_sales)}"
            f" = {_show(value, level_of)}%"
        ),
    )


def _interest_coverage(f: PeriodFinancials) -> RatioResult:
    name, unit = "インタレスト・カバレッジ・レシオ", "倍"
    # 受取利息・受取配当金が無い会社は、XBRL にその項目が出ない。営業外収益が無いだけなので0とする
    if f.operating_income is None:
        return _no_input(name, unit, ["営業利益"])
    if f.interest_expense is None:
        # 「支払利息なし」（ゼロ）と断定しない。無借金で項目が出ないのか、別の科目で計上されていて
        # 取れていないのかを区別できない（docs/decisions.md の未解決事項）
        return _unmeasurable(
            name,
            unit,
            "支払利息の項目がない（ゼロなのか、別の科目で計上されているのか区別できない。確認が必要）",
        )
    income = f.interest_income or ZERO
    dividend = f.dividend_income or ZERO
    if f.interest_expense == ZERO:
        return _unmeasurable(name, unit, "支払利息なし（支払利息がゼロ）")
    if f.interest_expense < ZERO:
        return _unmeasurable(name, unit, f"支払利息が負（{_yen(f.interest_expense)}）")
    numerator = f.operating_income + income + dividend
    value = numerator / f.interest_expense
    level_of = _at_least("5", "2")
    return RatioResult(
        name=name,
        unit=unit,
        value=value,
        level=level_of(value),
        basis=(
            f"（営業利益 {_yen(f.operating_income)} ＋ 受取利息 {_yen(income)}"
            f" ＋ 受取配当金 {_yen(dividend)}）÷ 支払利息 {_yen(f.interest_expense)}"
            f" = {_show(value, level_of)}倍"
        ),
    )


def _debt_repayment_years(f: PeriodFinancials) -> RatioResult:
    name, unit = "債務償還年数", "年"
    if f.trade_receivables is None or f.inventories is None or f.trade_payables is None:
        gaps = _missing(
            {
                "売上債権": f.trade_receivables,
                "棚卸資産": f.inventories,
                "仕入債務": f.trade_payables,
            }
        )
        return _no_input(name, unit, gaps)
    working_capital = f.trade_receivables + f.inventories - f.trade_payables  # 正常運転資金
    debt = f.interest_bearing_debt
    if debt is None:
        return _unmeasurable(
            name, unit, "有利子負債の内訳がすべて未計上（抽出漏れの可能性。確認が必要）"
        )
    to_repay = debt - working_capital  # 要償還債務
    debt_text = (
        f"有利子負債 {_yen(debt)} － 正常運転資金 {_yen(working_capital)}"
        f"（売上債権 {_yen(f.trade_receivables)} ＋ 棚卸資産 {_yen(f.inventories)}"
        f" － 仕入債務 {_yen(f.trade_payables)}）= 要償還債務 {_yen(to_repay)}"
    )

    # 附則第3条第1項: 要償還債務がゼロ以下（実質無借金）なら、償還原資にかかわらず0年
    if to_repay <= ZERO:
        return RatioResult(
            name=name,
            unit=unit,
            value=ZERO,
            level=Level.STANDARD,
            basis=f"{debt_text}。ゼロ以下のため0年",
        )

    if f.net_income_attributable_to_owners is None or f.depreciation is None:
        gaps = _missing(
            {
                "親会社株主に帰属する当期純利益": f.net_income_attributable_to_owners,
                "減価償却費": f.depreciation,
            }
        )
        return _no_input(name, unit, gaps)
    cash_flow = f.net_income_attributable_to_owners + f.depreciation  # 償還原資
    cf_text = (
        f"償還原資 {_yen(cash_flow)}"
        f"（親会社株主に帰属する当期純利益 {_yen(f.net_income_attributable_to_owners)}"
        f" ＋ 減価償却費 {_yen(f.depreciation)}）"
    )
    if cash_flow <= ZERO:
        # 第8条第2項ただし書き: 債務償還年数の算定不能は要精査水準として扱う
        return _unmeasurable(
            name,
            unit,
            f"償還原資がゼロ以下（{_yen(cash_flow)}）",
            basis=f"{debt_text}。{cf_text}。ゼロ以下のため算定不能",
            level=Level.SCRUTINY,
        )
    value = to_repay / cash_flow
    level_of = _at_most("10", "15")
    return RatioResult(
        name=name,
        unit=unit,
        value=value,
        level=level_of(value),
        basis=f"{debt_text}。{cf_text}。要償還債務 ÷ 償還原資 = {_show(value, level_of)}年",
    )


def _sales_growth(current: PeriodFinancials, previous: PeriodFinancials | None) -> RatioResult:
    name, unit = "売上高成長率", "%"
    if previous is None:
        return _unmeasurable(name, unit, "前期のデータなし")
    if current.net_sales is None or previous.net_sales is None:
        gaps = _missing({"当期売上高": current.net_sales, "前期売上高": previous.net_sales})
        return _no_input(name, unit, gaps)
    if previous.net_sales <= ZERO:
        return _unmeasurable(name, unit, f"前期売上高がゼロ以下（{_yen(previous.net_sales)}）")
    value = (current.net_sales / previous.net_sales - 1) * HUNDRED
    return RatioResult(
        name=name,
        unit=unit,
        value=value,
        level=None,  # 成長率には第8条の水準の区分がない（第11条の留意事項だけ）
        basis=(
            f"当期売上高 {_yen(current.net_sales)} ÷ 前期売上高 {_yen(previous.net_sales)}"
            f" － 1 = {_show(value, _is_sales_drop)}%"
        ),
    )


# ---- 第10条・第11条 ----


def _sales_drop(growth: RatioResult) -> FlagResult:
    """第11条: 売上高成長率が－10%以下（前期比で10%以上の減少）。"""
    name, clause = "売上高の10%以上の減少", "第11条"
    if growth.value is None:
        return FlagResult(name=name, clause=clause, applies=None, basis=growth.basis)
    return FlagResult(
        name=name,
        clause=clause,
        applies=_is_sales_drop(growth.value),
        basis=f"売上高成長率 {_show(growth.value, _is_sales_drop)}%（－10%以下で該当）",
    )


def _consecutive_operating_loss(
    current: PeriodFinancials, previous: PeriodFinancials | None
) -> FlagResult:
    """第10条: 営業損失が2期連続している。"""
    name, clause = "営業損失の2期連続", "第10条"
    if previous is None or current.operating_income is None or previous.operating_income is None:
        return FlagResult(
            name=name, clause=clause, applies=None, basis="2期分の営業利益がそろわない"
        )
    both = current.operating_income < ZERO and previous.operating_income < ZERO
    return FlagResult(
        name=name,
        clause=clause,
        applies=both,
        basis=(
            f"当期の営業利益 {_yen(current.operating_income)}、"
            f"前期の営業利益 {_yen(previous.operating_income)}"
        ),
    )


def compute_ratios(
    current: PeriodFinancials, previous: PeriodFinancials | None = None
) -> RatioReport:
    """当期（と、あれば前期）の連結財務データから、規程の財務比率と留意事項を計算する。"""
    growth = _sales_growth(current, previous)
    return RatioReport(
        equity_ratio=_equity_ratio(current),
        current_ratio=_current_ratio(current),
        operating_margin=_operating_margin(current),
        interest_coverage=_interest_coverage(current),
        debt_repayment_years=_debt_repayment_years(current),
        sales_growth=growth,
        consecutive_operating_loss=_consecutive_operating_loss(current, previous),
        sales_drop=_sales_drop(growth),
    )
