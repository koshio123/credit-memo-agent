"""XBRL（CSV）から規程の入力項目を読む規則のテスト。規則は実データ（対象10社）で確かめた項目名に基づく。"""

import zipfile
from decimal import Decimal
from pathlib import Path

import pytest

from ingest.xbrl_facts import (
    Fact,
    NoConsolidatedStatements,
    UnsupportedAccountingStandard,
    extract,
    read_facts,
)

D = Decimal


def F(
    element: str,
    value: int | str,
    context: str = "CurrentYearInstant",
    consolidated: str = "連結",
) -> Fact:
    if value == "－":
        return Fact(element, context, consolidated, D(0), nil=True)
    return Fact(element, context, consolidated, D(value), nil=False)


def _cur(*facts: Fact):
    return extract(list(facts), "current")


# ---- どの値を読むか ----


def test_連結の当期で_修飾のない値だけを読む() -> None:
    result = _cur(
        F("Assets", 100),
        F("Assets", 999, consolidated="個別"),  # 個別は読まない
        F("Assets", 888, context="CurrentYearInstant_NonConsolidatedMember"),  # 軸つきは読まない
        F("Assets", 777, context="Prior1YearInstant"),  # 前期は当期として読まない
    )
    assert result.financials.total_assets == D(100)


def test_前期は_同じ書類のPrior1Yearの列から読む() -> None:
    facts = [F("Assets", 100), F("Assets", 80, context="Prior1YearInstant")]
    assert extract(facts, "previous").financials.total_assets == D(80)


def test_項目が無ければNone() -> None:
    assert _cur(F("Assets", 100)).financials.net_assets is None


def test_貸借対照表と損益計算書の直接の項目() -> None:
    dur = "CurrentYearDuration"
    fin = _cur(
        F("Assets", 1000),
        F("NetAssets", 400),
        F("CurrentAssets", 600),
        F("CurrentLiabilities", 300),
        F("NetSales", 2000, dur),
        F("OperatingIncome", 100, dur),
        F("InterestIncomeNOI", 5, dur),
        F("DividendsIncomeNOI", 7, dur),
        F("InterestExpensesNOE", 9, dur),
        F("ProfitLossAttributableToOwnersOfParent", 60, dur),
        F("DepreciationAndAmortizationOpeCF", 40, dur),
    ).financials
    assert (fin.total_assets, fin.net_assets) == (D(1000), D(400))
    assert (fin.current_assets, fin.current_liabilities) == (D(600), D(300))
    assert (fin.net_sales, fin.operating_income) == (D(2000), D(100))
    assert (fin.interest_income, fin.dividend_income, fin.interest_expense) == (D(5), D(7), D(9))
    assert fin.net_income_attributable_to_owners == D(60)
    assert fin.depreciation == D(40)


def test_売上高は_営業収益の会社もある() -> None:
    fin = _cur(F("OperatingRevenue1", 500, "CurrentYearDuration")).financials
    assert fin.net_sales == D(500)


def test_受取利息と受取配当金が合算の会社は_合算を受取利息として扱う() -> None:
    fin = _cur(F("InterestAndDividendsIncomeNOI", 175, "CurrentYearDuration")).financials
    assert fin.interest_income == D(175)
    assert fin.dividend_income is None


def test_保険配当金は受取配当金に含めない() -> None:
    fin = _cur(
        F("DividendsIncomeNOI", 135, "CurrentYearDuration"),
        F("DividendsIncomeOfInsuranceNOI", 34, "CurrentYearDuration"),
    ).financials
    assert fin.dividend_income == D(135)


# ---- 売上債権（合成項目。合計の行と内訳の行を二重に数えない） ----


def test_売上債権_契約資産を含む合計の行があればそれを使い_内訳は足さない() -> None:
    # 5,334 + 36,039 + 20,986 = 62,359 ≒ 合計の行 62,360。内訳を足すと二重になる
    result = _cur(
        F("NotesAndAccountsReceivableTradeAndContractAssets", 62_360),
        F("NotesReceivableTrade", 5_334),
        F("AccountsReceivableTrade", 36_039),
        F("ContractAssets", 20_986),
    )
    assert result.financials.trade_receivables == D(62_360)
    assert (
        "NotesAndAccountsReceivableTradeAndContractAssets" in result.provenance["trade_receivables"]
    )


def test_売上債権_受取手形及び売掛金の合計の行には_電子記録債権を足す() -> None:
    # 電子記録債権は別の行で、合計の行に含まれない
    result = _cur(
        F("NotesAndAccountsReceivableTrade", 22_652),
        F("ElectronicallyRecordedMonetaryClaimsOperatingCA", 11_464),
        F("NotesReceivableTrade", 2_164),  # 内訳。足さない
        F("AccountsReceivableTrade", 20_488),
    )
    assert result.financials.trade_receivables == D(34_116)


def test_売上債権_受取手形及び売掛金の合計の行には_契約資産も足す() -> None:
    fin = _cur(F("NotesAndAccountsReceivableTrade", 100), F("ContractAssets", 30)).financials
    assert fin.trade_receivables == D(130)


def test_売上債権_合計の行が無ければ内訳を足す() -> None:
    fin = _cur(
        F("NotesReceivableTrade", 251),
        F("ElectronicallyRecordedMonetaryClaimsOperatingCA", 6_504),
        F("ContractAssets", 69_181),
    ).financials
    assert fin.trade_receivables == D(75_936)


def test_売上債権_ハイフンの内訳は0として足す() -> None:
    fin = _cur(F("AccountsReceivableTrade", 43_573), F("ContractAssets", "－")).financials
    assert fin.trade_receivables == D(43_573)


# ---- 棚卸資産 ----


def test_棚卸資産_合計の行があればそれを使う() -> None:
    fin = _cur(
        F("Inventories", 4_161), F("MerchandiseAndFinishedGoods", 633), F("WorkInProcess", 75)
    ).financials
    assert fin.inventories == D(4_161)


def test_棚卸資産_合計の行が無ければ内訳を足す() -> None:
    fin = _cur(
        F("MerchandiseAndFinishedGoods", 10_226),
        F("WorkInProcess", 2_605),
        F("RawMaterialsAndSupplies", 404),
    ).financials
    assert fin.inventories == D(13_235)


def test_棚卸資産_不動産会社の販売用不動産と建設の未成工事支出金を含む() -> None:
    real_estate = _cur(
        F("RealEstateForSale", 2_791), F("RealEstateForSaleInProcess", 280_764)
    ).financials
    assert real_estate.inventories == D(283_555)
    construction = _cur(
        F("CostsOnUncompletedConstructionContractsCNS", 322), F("RawMaterialsAndSupplies", 4_812)
    ).financials
    assert construction.inventories == D(5_134)


# ---- 仕入債務 ----


def test_仕入債務_支払手形及び買掛金に電子記録債務を足す() -> None:
    fin = _cur(
        F("NotesAndAccountsPayableTrade", 15_165),
        F("ElectronicallyRecordedObligationsOperatingCL", 15_344),
        F("ElectronicallyRecordedObligationsNonOperatingCL", 783),  # 営業外の電子記録債務は含めない
    ).financials
    assert fin.trade_payables == D(30_509)


def test_仕入債務_建設業の独自の項目() -> None:
    fin = _cur(
        F("NotesPayableAccountsPayableForConstructionContractsAndOtherCNS", 46_283),
        F("ElectronicallyRecordedObligationsOperatingCL", 5_198),
    ).financials
    assert fin.trade_payables == D(51_481)


def test_仕入債務_買掛金だけの会社() -> None:
    assert _cur(F("AccountsPayableTrade", 2_612)).financials.trade_payables == D(2_612)


# ---- 有利子負債 ----


def test_有利子負債の内訳_リース債務は流動と固定の両方() -> None:
    fin = _cur(
        F("ShortTermLoansPayable", 1),
        F("CurrentPortionOfLongTermLoansPayable", 4),
        F("CurrentPortionOfBonds", 8),
        F("BondsPayable", 16),
        F("LongTermLoansPayable", 32),
        F("LeaseObligationsCL", 64),
        F("LeaseObligationsNCL", 128),
    ).financials
    assert fin.interest_bearing_debt == D(253)
    assert fin.lease_obligations == D(192)


def test_有利子負債_行はあるが金額なし_ハイフン_は借入なしとして0にする() -> None:
    # 能美防災: 短期借入金の行が「－」。行そのものが無い場合(None)とは違い、借入なしを表す
    fin = _cur(F("ShortTermLoansPayable", "－")).financials
    assert fin.interest_bearing_debt == D(0)


def test_有利子負債_行が1つも無ければ抽出漏れと区別できないのでNone() -> None:
    assert _cur(F("Assets", 100)).financials.interest_bearing_debt is None


def test_有利子負債_転換社債型の新株予約権付社債も1年内償還予定に含める() -> None:
    fin = _cur(F("CurrentPortionOfBondsWithSubscriptionRightsToShares", 500)).financials
    assert fin.current_portion_bonds == D(500)


# ---- 出所の記録・会計基準 ----


def test_どの項目から得た値かを記録する() -> None:
    result = _cur(
        F("NotesAndAccountsReceivableTrade", 10),
        F("ElectronicallyRecordedMonetaryClaimsOperatingCA", 5),
    )
    assert "NotesAndAccountsReceivableTrade" in result.provenance["trade_receivables"]
    assert (
        "ElectronicallyRecordedMonetaryClaimsOperatingCA" in result.provenance["trade_receivables"]
    )


def test_IFRSの書類は対象外として明示的に失敗する() -> None:
    # 連結が jpigp_cor（IFRS）の会社は、J-GAAP の項目では読めない。黙って空の結果にしない
    facts = [
        Fact("AssetsIFRS", "CurrentYearInstant", "その他", D(1), nil=False, namespace="jpigp_cor")
    ]
    with pytest.raises(UnsupportedAccountingStandard):
        extract(facts, "current")


# ---- CSV の読み込み ----


def _zip_with_csv(path: Path, lines: list[list[str]]) -> Path:
    header = [
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
    text = "\n".join("\t".join(row) for row in [header, *lines]) + "\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "XBRL_TO_CSV/jpcrp030000-asr-001_E00000-000_2026-03-31_01_2026-06-10.csv",
            text.encode("utf-16"),
        )
        z.writestr("XBRL_TO_CSV/jpaud-aai-cc-001_E00000-000.csv", "無関係".encode("utf-16"))
    return path


def test_CSVのzipを読む(tmp_path: Path) -> None:
    z = _zip_with_csv(
        tmp_path / "x.zip",
        [
            [
                "jppfs_cor:Assets",
                "資産",
                "CurrentYearInstant",
                "当期末",
                "連結",
                "時点",
                "JPY",
                "円",
                "1234",
            ],
            [
                "jppfs_cor:ShortTermLoansPayable",
                "短期借入金",
                "CurrentYearInstant",
                "当期末",
                "連結",
                "時点",
                "JPY",
                "円",
                "－",
            ],
            [
                "jppfs_cor:Assets",
                "資産",
                "CurrentYearInstant",
                "当期末",
                "個別",
                "時点",
                "JPY",
                "円",
                "999",
            ],
            [
                "jpcrp_cor:Remarks",
                "注記",
                "CurrentYearDuration",
                "当期",
                "その他",
                "期間",
                "",
                "",
                "文章です",
            ],
            [
                "jppfs_cor:NetAssets",
                "純資産",
                "CurrentYearInstant",
                "当期末",
                "連結",
                "時点",
                "JPY",
                "円",
                "-50",
            ],
        ],
    )
    facts = {(f.element, f.consolidated): f for f in read_facts(z)}

    assert facts[("Assets", "連結")].value == D(1234)
    assert facts[("ShortTermLoansPayable", "連結")].nil is True
    assert facts[("NetAssets", "連結")].value == D(-50)
    assert ("Remarks", "その他") not in facts  # 数値でない行は読まない


# ---- コードレビューでの指摘への対応 ----


def test_有利子負債_固定負債の新株予約権付社債も社債に含める() -> None:
    fin = _cur(F("BondsPayable", 100), F("BondsWithSubscriptionRightsToShares", 40)).financials
    assert fin.bonds == D(140)


def test_先頭の候補が_ハイフン_なら_値のある後ろの候補を使う_売上高() -> None:
    fin = _cur(
        F("NetSales", "－", "CurrentYearDuration"),
        F("OperatingRevenue1", 500, "CurrentYearDuration"),
    ).financials
    assert fin.net_sales == D(500)


def test_候補がすべてハイフンなら0() -> None:
    assert _cur(F("NetSales", "－", "CurrentYearDuration")).financials.net_sales == D(0)


def test_棚卸資産の合計の行がハイフンなら_値のある内訳を使う() -> None:
    fin = _cur(F("Inventories", "－"), F("MerchandiseAndFinishedGoods", 300)).financials
    assert fin.inventories == D(300)


def test_売上債権の合計の行がハイフンなら_値のある内訳を使う() -> None:
    fin = _cur(
        F("NotesAndAccountsReceivableTradeAndContractAssets", "－"),
        F("NotesReceivableTrade", 10),
        F("AccountsReceivableTrade", 90),
    ).financials
    assert fin.trade_receivables == D(100)


def test_連結の財務諸表が1行も無い書類は_黙って空にせず失敗する() -> None:
    # 個別しか無い、または形式を読めない書類。すべて None の結果を返さない
    with pytest.raises(NoConsolidatedStatements):
        extract([F("Assets", 100, consolidated="個別")], "current")
    with pytest.raises(NoConsolidatedStatements):
        extract([], "current")


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity", "sNaN"])
def test_NaNやInfinityは数値として読まない(tmp_path: Path, raw: str) -> None:
    z = _zip_with_csv(
        tmp_path / "x.zip",
        [
            [
                "jppfs_cor:Assets",
                "資産",
                "CurrentYearInstant",
                "当期末",
                "連結",
                "時点",
                "JPY",
                "円",
                raw,
            ]
        ],
    )
    assert read_facts(z) == []
