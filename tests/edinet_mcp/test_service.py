"""edinet_mcp のサービス層のテスト。検索・XBRL・PDF の読み込みは差し替えた関数で行う。"""

from decimal import Decimal

import pytest

from edinet_mcp.service import EdinetMcpError, EdinetService
from ingest.xbrl_facts import Fact

from .conftest import CUR_DOC, FACTS, PAGES, FakeSearcher, company, expected_ratios

# ---- 会社の一覧 ----


def test_会社の一覧は_証券コードと名前と当期の書類を返す(service: EdinetService) -> None:
    listed = service.list_companies()
    assert [c.sec_code for c in listed] == ["9999", "8888"]
    first = listed[0]
    assert (first.name, first.doc_id, first.period_end) == ("サンプル工業", CUR_DOC, "2026-03-31")


# ---- 証券コードの解決 ----


@pytest.mark.parametrize("code", ["9999", " 9999 ", "９９９９", "99990"])
def test_証券コードは_全角_空白_5桁表記も受け付ける(service: EdinetService, code: str) -> None:
    assert service.get_page(code, 1).doc_id == CUR_DOC


def test_知らない証券コードは_分かるエラーにする(service: EdinetService) -> None:
    with pytest.raises(EdinetMcpError, match="1234"):
        service.get_ratios("1234")


# ---- 検索 ----


def test_検索は_会社の書類に絞り_出典つきの本文を返す(
    service: EdinetService, searcher: FakeSearcher
) -> None:
    passages = service.search_filings("原材料価格の高騰", "9999", k=3)

    assert searcher.calls == [{"query": "原材料価格の高騰", "k": 3, "doc_ids": [CUR_DOC]}]
    p = passages[0]
    assert (p.sec_code, p.company, p.doc_id) == ("9999", "サンプル工業", CUR_DOC)
    assert (p.page_start, p.page_end) == (2, 3)
    assert p.heading_path == ["第2 【事業の状況】", "3 【事業等のリスク】"]
    assert p.text == PAGES[1]
    assert p.rank == 1


@pytest.mark.parametrize("k", [0, -1, 21])
def test_検索の件数は1から20まで(service: EdinetService, k: int) -> None:
    with pytest.raises(EdinetMcpError, match="k"):
        service.search_filings("x", "9999", k=k)


def test_空のクエリは断る(service: EdinetService) -> None:
    with pytest.raises(EdinetMcpError, match="クエリ"):
        service.search_filings("  ", "9999")


# ---- 財務数値（XBRL） ----


def test_財務数値は_当期と前期を分けて返し_出典の項目名もつける(service: EdinetService) -> None:
    cur = service.get_financials("9999", "current")
    prev = service.get_financials("9999", "previous")

    assert cur.financials.total_assets == Decimal(1000)
    assert prev.financials.net_sales == Decimal(1800)
    assert cur.provenance["total_assets"] == "Assets"
    assert (cur.doc_id, cur.period_end) == (CUR_DOC, "2026-03-31")
    assert prev.period_end == "2025-03-31"
    assert prev.doc_id == CUR_DOC  # 前期の列も、当期の書類から読む
    assert cur.source == "XBRL"


def test_期の指定が不正なら断る(service: EdinetService) -> None:
    with pytest.raises(EdinetMcpError, match="period"):
        service.get_financials("9999", "next")  # type: ignore[arg-type]


def test_IFRSの書類は_読めないことを説明して失敗する() -> None:
    ifrs = [
        Fact("Assets", "CurrentYearInstant", "連結", Decimal(1), nil=False, namespace="jpigp_cor")
    ]
    svc = EdinetService(
        searcher=FakeSearcher(),
        companies=[company()],
        load_facts=lambda _doc: ifrs,
        load_pages=lambda _doc: PAGES,
    )
    with pytest.raises(EdinetMcpError, match="IFRS"):
        svc.get_financials("9999")


def test_書類のデータが無ければ_取り込みが必要と伝える() -> None:
    def missing(doc_id: str) -> list[Fact]:
        raise FileNotFoundError(doc_id)

    svc = EdinetService(
        searcher=FakeSearcher(),
        companies=[company()],
        load_facts=missing,
        load_pages=lambda _doc: PAGES,
    )
    with pytest.raises(EdinetMcpError, match=CUR_DOC):
        svc.get_financials("9999")


# ---- 財務比率 ----


def test_財務比率は_ratiosモジュールの計算結果そのもの(service: EdinetService) -> None:
    result = service.get_ratios("9999")
    assert result.ratios == expected_ratios()
    assert result.ratios.equity_ratio.value == Decimal("50.0")  # 500 / 1000
    assert (result.sec_code, result.period_end) == ("9999", "2026-03-31")
    assert "第7条" in result.policy_note or "内規" in result.policy_note


def test_財務比率は_可否の判断を含まない(service: EdinetService) -> None:
    dumped = service.get_ratios("9999").model_dump_json()
    for word in ("融資可", "融資不可", "承認", "否決"):
        assert word not in dumped


# ---- ページ ----


def test_ページは1始まりで返す(service: EdinetService) -> None:
    page = service.get_page("9999", 2)
    assert (page.page, page.n_pages, page.doc_id) == (2, 3, CUR_DOC)
    assert page.text == PAGES[1]


@pytest.mark.parametrize("n", [0, 4, -1])
def test_範囲外のページは_ページ数を添えて断る(service: EdinetService, n: int) -> None:
    with pytest.raises(EdinetMcpError, match="3"):
        service.get_page("9999", n)


def test_検索のヒットは_ページで取り直せる(service: EdinetService) -> None:
    hit = service.search_filings("原材料", "9999")[0]
    assert PAGES[1] in service.get_page("9999", hit.page_start).text


def test_ファクトはモジュール定数のまま(service: EdinetService) -> None:
    # 読み込み関数が返したリストを、サービスが書き換えない
    before = list(FACTS)
    service.get_financials("9999")
    assert before == FACTS
