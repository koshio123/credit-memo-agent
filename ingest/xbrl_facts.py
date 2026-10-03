"""XBRL（EDINET の財務データ CSV）から、規程の財務比率の入力項目（PeriodFinancials）を読む。

これが L1 評価の**正解データ**になる（PDF から抽出した値を、この値と突き合わせる）。

読み方の規則は、対象 10 社の実データ（2026-09-30 時点）で確かめた項目名に基づく。
売上債権・棚卸資産・仕入債務は、会社によって行の分け方が違う「合成項目」なので、
優先順位つきの規則で決め、どの項目から得たかを provenance（出所）に残す。

- 連結の、当期（または前期）の、軸のない値だけを読む。
- 「－」（行はあるが金額なし）は 0 として扱い、行そのものが無い場合（None）と区別する。
- 合計の行と内訳の行が同時に出るので、合計の行があれば内訳は足さない（二重に数えない）。
- IFRS の書類（連結が jpigp_cor）は対象外として、明示的に失敗させる。
"""

import csv
import io
import zipfile
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from finance.ratios import PeriodFinancials

Period = Literal["current", "previous"]

# CSV の「コンテキストID」。軸（連結の範囲など）がつくと _Xxx が付くので、完全一致だけを読む
_CONTEXTS: dict[Period, tuple[str, str]] = {
    "current": ("CurrentYearInstant", "CurrentYearDuration"),
    "previous": ("Prior1YearInstant", "Prior1YearDuration"),
}
_NAMESPACES = ("jppfs_cor", "jpigp_cor")  # J-GAAP / IFRS の財務諸表の項目


class UnsupportedAccountingStandard(Exception):
    """J-GAAP の連結財務諸表を読めない書類（IFRS など）。"""


class NoConsolidatedStatements(Exception):
    """連結の財務諸表の行が1つも無い書類（個別のみ、または形式を読めない）。

    すべて None の結果を黙って返さないために、失敗させる。
    """


@dataclass(frozen=True)
class Fact:
    element: str  # 名前空間を除いた項目名
    context: str
    consolidated: str  # 連結 / 個別 / その他
    value: Decimal
    nil: bool  # 「－」（行はあるが金額なし）
    namespace: str = "jppfs_cor"


@dataclass(frozen=True)
class Extraction:
    financials: PeriodFinancials
    provenance: dict[str, str] = field(default_factory=dict[str, str])  # 項目 -> 得た元の項目名


def read_facts(csv_zip: Path) -> list[Fact]:
    """EDINET の財務データ CSV（zip）から、財務諸表の数値の行を読む。

    Args:
        csv_zip: 財務データ CSV の zip のパス。

    Returns:
        数値の行（項目名・コンテキスト・値など）のリスト。数値でない行は除く。
    """
    facts: list[Fact] = []
    with zipfile.ZipFile(csv_zip) as z:
        for name in z.namelist():
            # 主たる書類の CSV（jpcrp...）。監査報告書（jpaud...）などは読まない
            if not (name.endswith(".csv") and "jpcrp" in name):
                continue
            text = z.read(name).decode("utf-16")
            for row in csv.reader(io.StringIO(text), delimiter="\t"):
                if len(row) < 9:
                    continue
                namespace, _, element = row[0].partition(":")
                if namespace not in _NAMESPACES:
                    continue
                raw = row[8]
                if raw == "－":
                    facts.append(
                        Fact(element, row[2], row[4], Decimal(0), nil=True, namespace=namespace)
                    )
                    continue
                try:
                    value = Decimal(raw)
                except InvalidOperation:
                    continue  # 文章など、数値でない行
                if not value.is_finite():
                    continue  # "NaN" や "Infinity" は Decimal が受け付けるが、金額ではない
                facts.append(Fact(element, row[2], row[4], value, nil=False, namespace=namespace))
    return facts


def _index(facts: list[Fact], period: Period) -> dict[str, Fact]:
    """連結の J-GAAP の行から、期のコンテキストに合うものを項目名で引けるようにする。

    Args:
        facts: 読み込んだ数値の行。
        period: 当期か前期か。

    Returns:
        項目名から行への対応。同じ項目が複数あれば最初の行。

    Raises:
        UnsupportedAccountingStandard: 連結財務諸表が IFRS のとき。
        NoConsolidatedStatements: 連結の財務諸表の行が無いとき。
    """
    contexts = _CONTEXTS[period]
    jgaap = [f for f in facts if f.namespace == "jppfs_cor" and f.consolidated == "連結"]
    if not jgaap:
        if any(f.namespace == "jpigp_cor" for f in facts):
            raise UnsupportedAccountingStandard(
                "連結財務諸表が IFRS（jpigp_cor）の書類です。J-GAAP の項目では読めません"
            )
        raise NoConsolidatedStatements(
            "連結の財務諸表の行が見つかりません（個別のみ、または形式が違う）"
        )
    index: dict[str, Fact] = {}
    for f in jgaap:
        if f.context in contexts:
            index.setdefault(f.element, f)
    return index


# ---- 規則 ----


def _sum(index: dict[str, Fact], names: list[str]) -> tuple[Decimal | None, list[str]]:
    """存在する項目の合計。1つも無ければ None。

    Args:
        index: 項目名から行への対応。
        names: 合計する項目名の候補。

    Returns:
        (合計, 使った項目名のリスト)。1つも無ければ (None, [])。
    """
    used = [n for n in names if n in index]
    if not used:
        return None, []
    return sum((index[n].value for n in used), Decimal(0)), used


def _has(index: dict[str, Fact], name: str) -> bool:
    """値のある行か。「－」（行はあるが金額なし）は、値のある行とは扱わない。

    Args:
        index: 項目名から行への対応。
        name: 項目名。

    Returns:
        値のある行があれば True。
    """
    return name in index and not index[name].nil


def _first(index: dict[str, Fact], names: list[str]) -> tuple[Decimal | None, list[str]]:
    """値のある最初の候補。値のある候補が無く「－」の行だけがあれば 0、行が無ければ None。

    Args:
        index: 項目名から行への対応。
        names: 項目名の候補（優先順）。

    Returns:
        (値, 使った項目名のリスト)。行が無ければ (None, [])。
    """
    for n in names:
        if _has(index, n):
            return index[n].value, [n]
    for n in names:
        if n in index:
            return Decimal(0), [n]
    return None, []


_ELECTRONIC_RECEIVABLES = "ElectronicallyRecordedMonetaryClaimsOperatingCA"  # 電子記録債権
_ELECTRONIC_PAYABLES = "ElectronicallyRecordedObligationsOperatingCL"  # 電子記録債務（営業）


def _receivables(index: dict[str, Fact]) -> tuple[Decimal | None, list[str]]:
    """売上債権。合計の行があれば内訳は足さず、合計の行に含まれない行だけを足す。

    Args:
        index: 項目名から行への対応。

    Returns:
        (売上債権, 使った項目名のリスト)。行が無ければ (None, [])。
    """
    if _has(index, "NotesAndAccountsReceivableTradeAndContractAssets"):
        # 受取手形、売掛金及び契約資産（契約資産を含む）。電子記録債権は別の行
        return _sum(
            index, ["NotesAndAccountsReceivableTradeAndContractAssets", _ELECTRONIC_RECEIVABLES]
        )
    if _has(index, "NotesAndAccountsReceivableTrade"):
        # 受取手形及び売掛金（契約資産・電子記録債権を含まない）
        return _sum(
            index, ["NotesAndAccountsReceivableTrade", _ELECTRONIC_RECEIVABLES, "ContractAssets"]
        )
    return _sum(
        index,
        [
            "NotesReceivableTrade",
            "AccountsReceivableTrade",
            _ELECTRONIC_RECEIVABLES,
            "ContractAssets",
            # ここに来るのは、合計の行が無いか「－」の場合。「－」の合計の行だけなら 0 になる
            "NotesAndAccountsReceivableTradeAndContractAssets",
            "NotesAndAccountsReceivableTrade",
        ],
    )


def _inventories(index: dict[str, Fact]) -> tuple[Decimal | None, list[str]]:
    """棚卸資産。合計の行があればそれ、無ければ内訳（不動産の販売用不動産、
    建設の未成工事支出金を含む）。

    Args:
        index: 項目名から行への対応。

    Returns:
        (棚卸資産, 使った項目名のリスト)。行が無ければ (None, [])。
    """
    if _has(index, "Inventories"):
        return index["Inventories"].value, ["Inventories"]
    return _sum(
        index,
        [
            "MerchandiseAndFinishedGoods",
            "WorkInProcess",
            "RawMaterialsAndSupplies",
            "RawMaterialsAndSuppliesCNS",
            "CostsOnUncompletedConstructionContractsCNS",
            "RealEstateForSale",
            "RealEstateForSaleInProcess",
            "Inventories",  # 合計の行が「－」ならここで 0 になる（値があれば上で返している）
        ],
    )


def _payables(index: dict[str, Fact]) -> tuple[Decimal | None, list[str]]:
    """仕入債務。電子記録債務（営業）は別の行なので足す。営業外の電子記録債務は含めない。

    Args:
        index: 項目名から行への対応。

    Returns:
        (仕入債務, 使った項目名のリスト)。行が無ければ (None, [])。
    """
    if _has(index, "NotesAndAccountsPayableTrade"):
        return _sum(index, ["NotesAndAccountsPayableTrade", _ELECTRONIC_PAYABLES])
    if _has(index, "NotesPayableAccountsPayableForConstructionContractsAndOtherCNS"):
        return _sum(
            index,
            [
                "NotesPayableAccountsPayableForConstructionContractsAndOtherCNS",
                _ELECTRONIC_PAYABLES,
            ],
        )
    return _sum(
        index,
        [
            "NotesPayableTrade",
            "AccountsPayableTrade",
            _ELECTRONIC_PAYABLES,
            # 合計の行が「－」の場合に 0 になる（値があれば上で返している）
            "NotesAndAccountsPayableTrade",
            "NotesPayableAccountsPayableForConstructionContractsAndOtherCNS",
        ],
    )


# 項目 -> 項目名の候補（先に書いたものを優先。合成項目は別の関数）
_DIRECT: dict[str, list[str]] = {
    "total_assets": ["Assets"],
    "net_assets": ["NetAssets"],
    "current_assets": ["CurrentAssets"],
    "current_liabilities": ["CurrentLiabilities"],
    "net_sales": ["NetSales", "OperatingRevenue1"],
    "operating_income": ["OperatingIncome"],
    # 受取利息と受取配当金が合算の行しかない会社は、合算を受取利息として扱う
    # （インタレスト・カバレッジの分子は合計なので足りる）
    "interest_income": ["InterestIncomeNOI", "InterestAndDividendsIncomeNOI"],
    "dividend_income": [
        "DividendsIncomeNOI"
    ],  # 保険配当金（DividendsIncomeOfInsuranceNOI）は含めない
    "interest_expense": ["InterestExpensesNOE", "InterestExpensesOpeCF"],
    "net_income_attributable_to_owners": ["ProfitLossAttributableToOwnersOfParent"],
    "depreciation": ["DepreciationAndAmortizationOpeCF"],
}

# 有利子負債の内訳（附則第1条）。リース債務は流動と固定の両方
_DEBT: dict[str, list[str]] = {
    "short_term_borrowings": ["ShortTermLoansPayable"],
    "commercial_paper": ["CommercialPapersLiabilities"],  # 対象10社では未観測
    "current_portion_long_term_borrowings": ["CurrentPortionOfLongTermLoansPayable"],
    "current_portion_bonds": [
        "CurrentPortionOfBonds",
        "CurrentPortionOfBondsWithSubscriptionRightsToShares",
    ],
    "bonds": ["BondsPayable", "BondsWithSubscriptionRightsToShares"],
    "long_term_borrowings": ["LongTermLoansPayable"],
    "lease_obligations": ["LeaseObligationsCL", "LeaseObligationsNCL"],
}


def extract(facts: list[Fact], period: Period) -> Extraction:
    """当期（または前期）の連結財務データから、規程の入力項目を読む。

    Args:
        facts: 読み込んだ数値の行。
        period: 当期か前期か。

    Returns:
        読み取った財務データと、項目ごとの XBRL の項目名。

    Raises:
        UnsupportedAccountingStandard: 連結財務諸表が IFRS のとき。
        NoConsolidatedStatements: 連結の財務諸表の行が無いとき。
    """
    index = _index(facts, period)
    values: dict[str, Decimal | None] = {}
    provenance: dict[str, str] = {}

    def put(field_name: str, result: tuple[Decimal | None, list[str]]) -> None:
        """読み取り結果を、値と項目名の記録に入れる。

        Args:
            field_name: 財務データの項目名。
            result: (値, 使った XBRL の項目名のリスト)。
        """
        value, used = result
        values[field_name] = value
        if used:
            provenance[field_name] = " + ".join(used)

    for field_name, names in _DIRECT.items():
        put(field_name, _first(index, names))
    for field_name, names in _DEBT.items():
        put(field_name, _sum(index, names))
    put("trade_receivables", _receivables(index))
    put("inventories", _inventories(index))
    put("trade_payables", _payables(index))

    return Extraction(PeriodFinancials(**values), provenance)
