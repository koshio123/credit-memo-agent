"""証拠の収集: サービスの結果から、数値の証拠と本文の証拠を作り、ID を振る。"""

from decimal import Decimal

from agents.evidence import collect_metrics, collect_passages
from agents.state import EvidencePool
from edinet_mcp.service import EdinetService
from retrieval.citations import verify_span
from tests.edinet_fakes import FakeSearcher

PAGES = ["表紙", "3 【事業等のリスク】\n原材料価格が高騰する可能性があります。", "従業員の状況"]


# ---- 数値 ----


def test_財務比率の証拠は_当期と前期の両方に作る(service: EdinetService) -> None:
    pool = EvidencePool()
    metrics = collect_metrics(service, "9999", pool)

    cur = metrics["equity_ratio.current"]
    prev = metrics["equity_ratio.previous"]
    assert (cur.label, cur.period, cur.display) == ("自己資本比率", "current", "50.0%")
    assert cur.level == "標準"
    assert (prev.period, prev.display) == ("previous", "50.0%")
    assert cur.value == Decimal("50")
    assert "純資産" in cur.basis and "総資産" in cur.basis
    assert pool.get(cur.id) is cur


def test_証拠には_XBRLの項目名と_PDFのページがつく(service: EdinetService) -> None:
    cur = collect_metrics(service, "9999", EvidencePool())["equity_ratio.current"]
    assert cur.xbrl_items == {"net_assets": "NetAssets", "total_assets": "Assets"}
    assert cur.pdf_pages == [3]  # 総資産だけ PDF が同じ値を読んでいる
    assert cur.doc_id == "S100CUR0"


def test_算定できない指標は_算定不能と理由を示す(service: EdinetService) -> None:
    m = collect_metrics(service, "9999", EvidencePool())["interest_coverage.current"]
    assert m.value is None
    assert m.display == "算定不能"
    assert "算定不能" in m.basis  # 理由つきの説明
    assert m.level is None


def test_売上高成長率は当期だけ_前期の分は作らない(service: EdinetService) -> None:
    metrics = collect_metrics(service, "9999", EvidencePool())
    assert "sales_growth.current" in metrics
    assert "sales_growth.previous" not in metrics
    assert metrics["sales_growth.current"].display.endswith("%")


def test_規程の留意事項は_該当か非該当かを示す(service: EdinetService) -> None:
    metrics = collect_metrics(service, "9999", EvidencePool())
    assert metrics["sales_drop.current"].display == "非該当"  # 1800 → 2000 は増収
    assert metrics["consecutive_operating_loss.current"].display == "非該当"


def test_主要な金額の証拠は_円か百万円で表記する(service: EdinetService) -> None:
    m = collect_metrics(service, "9999", EvidencePool())
    assert m["net_sales.current"].display == "2,000円"  # 小さい金額は円のまま
    assert m["net_sales.previous"].display == "1,800円"
    assert "NetSales" in m["net_sales.current"].basis


def test_百万円以上の金額は百万円で表記する() -> None:
    from agents.evidence import format_yen

    assert format_yen(Decimal("133696000000")) == "133,696百万円"
    assert format_yen(Decimal("-2500000")) == "-3百万円"  # 四捨五入（0.5 は切り上げ）
    assert format_yen(Decimal("1500000")) == "2百万円"
    assert format_yen(Decimal("999999")) == "999,999円"
    assert format_yen(Decimal("0")) == "0円"


def test_証拠のIDは_収集の順にE1から振られる(service: EdinetService) -> None:
    pool = EvidencePool()
    metrics = collect_metrics(service, "9999", pool)
    assert sorted(pool.items) == sorted(m.id for m in metrics.values())
    assert "E1" in pool.items


# ---- 有利子負債の構成（第15条） ----


def test_有利子負債の合計と内訳を証拠にする(service: EdinetService) -> None:
    m = collect_metrics(service, "9999", EvidencePool())
    total = m["interest_bearing_debt.current"]
    assert total.value == Decimal(400)
    assert total.display == "400円"
    assert "短期借入金 100" in total.basis and "長期借入金 250" in total.basis
    assert m["short_term_borrowings.current"].display == "100円"
    assert "long_term_borrowings.current" in m
    assert "bonds.current" not in m  # 項目が無い内訳は作らない


def test_1年以内に返済する有利子負債の割合を計算する(service: EdinetService) -> None:
    ratio = collect_metrics(service, "9999", EvidencePool())["debt_due_within_1y_ratio.current"]
    assert ratio.value == Decimal(150) / Decimal(400) * 100  # (100 + 50) / 400
    assert ratio.display == "37.5%"
    assert "リース債務" in ratio.basis  # 1年以内の分が分けられない旨


def test_有利子負債が無ければ_割合は算定不能() -> None:
    # 合成データには内訳があるので、ゼロの場合は関数を直接確かめる
    from agents.evidence import debt_due_within_1y
    from finance.ratios import PeriodFinancials

    value, basis = debt_due_within_1y(PeriodFinancials(short_term_borrowings=Decimal(0)))
    assert value is None
    assert "算定不能" in basis


# ---- 本文 ----


def test_検索した本文の証拠は_出典スパンを持ち_引用文がページ本文と一致する(
    service: EdinetService,
) -> None:
    pool = EvidencePool()
    found = collect_passages(service, "9999", ["原材料価格の高騰"], pool, k=3)

    (evidence,) = found["原材料価格の高騰"]
    assert evidence.sec_code == "9999"
    assert evidence.heading_path == ["第2 【事業の状況】", "3 【事業等のリスク】"]
    assert all(verify_span(s, PAGES) for s in evidence.spans)
    assert pool.get(evidence.id) is evidence


def test_同じ本文が複数の問いで見つかっても_証拠は1つにまとめる(
    service: EdinetService, searcher: FakeSearcher
) -> None:
    pool = EvidencePool()
    found = collect_passages(service, "9999", ["問いA", "問いB"], pool, k=3)

    assert found["問いA"][0].id == found["問いB"][0].id
    assert len(pool.items) == 1
    assert len(searcher.calls) == 2


def test_問いが空なら何も集めない(service: EdinetService) -> None:
    assert collect_passages(service, "9999", [], EvidencePool()) == {}
