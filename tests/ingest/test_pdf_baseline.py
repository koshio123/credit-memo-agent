"""pdfplumber の基準線の、テキスト解析部分のテスト。

文面は実データ（有価証券報告書の PDF）の書式に基づく。
"""

from decimal import Decimal
from pathlib import Path

import pytest

from ingest.pdf_baseline import (
    ExtractedValue,
    cached_pages,
    extract_items,
    find_sections,
    parse_line,
    unit_of,
)

D = Decimal

BS_PAGE = """EDINET提出書類
サンプル株式会社(E00000)
有価証券報告書
1 【連結財務諸表等】
(1) 【連結財務諸表】
1【連結貸借対照表】
(単位:百万円)
前連結会計年度 当連結会計年度
(2025年3月31日) (2026年3月31日)
資産の部
流動資産
現金及び預金 43,408 35,442
受取手形、売掛金及び契約資産 ※5 55,319 ※5 62,360
貸倒引当金 △589 △596
流動資産合計 117,150 117,172
資産合計 166,877 181,811
"""

BS_PAGE_2 = """EDINET提出書類
サンプル株式会社(E00000)
有価証券報告書
(単位:百万円)
負債の部
流動負債
支払手形及び買掛金 5,000 6,000
流動負債合計 30,032 34,982
純資産合計 130,030 138,986
"""

PL_PAGE = """EDINET提出書類
サンプル株式会社(E00000)
有価証券報告書
2【連結損益計算書及び連結包括利益計算書】
(1)【連結損益計算書】
(単位:百万円)
前連結会計年度 当連結会計年度
売上高 ※1 133,696 ※1 139,657
販売費及び一般管理費
減価償却費 1,117 1,157
営業利益 15,677 18,349
営業外収益
受取利息 32 68
受取配当金 111 135
営業外費用
支払利息 20 23
親会社株主に帰属する当期純利益 11,098 13,648
"""

CI_PAGE = """EDINET提出書類
サンプル株式会社(E00000)
有価証券報告書
【連結包括利益計算書】
(単位:百万円)
当期純利益 11,200 13,700
親会社株主に係る包括利益 11,000 15,000
"""

CF_PAGE = """EDINET提出書類
サンプル株式会社(E00000)
有価証券報告書
4【連結キャッシュ・フロー計算書】
(単位:百万円)
営業活動によるキャッシュ・フロー
税金等調整前当期純利益 16,000 19,000
減価償却費 2,526 2,445
受取利息及び受取配当金 △144 △203
支払利息 20 23
"""

NOTES_PAGE = """EDINET提出書類
サンプル株式会社(E00000)
有価証券報告書
【注記事項】
(連結財務諸表作成のための基本となる重要な事項)
売上高 999,999 999,999
減価償却費 9,999 9,999
"""

# 表紙や経営指標の推移にも見出しの語は出るが、財務諸表の本体ではない
INDEX_PAGE = """EDINET提出書類
サンプル株式会社(E00000)
有価証券報告書
連結貸借対照表、連結損益計算書の詳細は第5【経理の状況】を参照してください。
売上高 100,000 200,000
"""

PAGES = [INDEX_PAGE, BS_PAGE, BS_PAGE_2, PL_PAGE, CI_PAGE, CF_PAGE, NOTES_PAGE]


# ---- 行の解析 ----


@pytest.mark.parametrize(
    ("line", "label", "values"),
    [
        ("流動資産合計 117,150 117,172", "流動資産合計", [D(117150), D(117172)]),
        ("貸倒引当金 △589 △596", "貸倒引当金", [D(-589), D(-596)]),
        (
            "受取手形、売掛金及び契約資産 ※5 55,319 ※5 62,360",
            "受取手形、売掛金及び契約資産",
            [D(55319), D(62360)],
        ),
        (
            "建物及び構築物(純額) ※1,4 12,139 ※1,4 12,475",
            "建物及び構築物(純額)",
            [D(12139), D(12475)],
        ),
        ("固定資産売却益 － 304", "固定資産売却益", [D(0), D(304)]),
        ("営業利益又は営業損失(△) △647 3,536", "営業利益又は営業損失(△)", [D(-647), D(3536)]),
        ("売上高 ※1 133,696 ※1 139,657", "売上高", [D(133696), D(139657)]),
    ],
)
def test_行をラベルと数値に分ける(line: str, label: str, values: list[Decimal]) -> None:
    parsed = parse_line(line)
    assert parsed is not None
    assert parsed.label == label
    assert parsed.values == values


@pytest.mark.parametrize(
    "line", ["流動資産", "(単位:百万円)", "前連結会計年度 当連結会計年度", "資産の部", ""]
)
def test_数値のない行は読まない(line: str) -> None:
    assert parse_line(line) is None


# ---- 単位 ----


@pytest.mark.parametrize(
    ("text", "unit"),
    [
        ("(単位:百万円)", 1_000_000),
        ("(単位:千円)", 1_000),
        ("(単位:円)", 1),
        ("(単位:億円)", 100_000_000),
    ],
)
def test_単位を読む(text: str, unit: int) -> None:
    assert unit_of(text) == unit


def test_単位の記載がなければNone() -> None:
    assert unit_of("流動資産合計 1 2") is None


# ---- 範囲の特定 ----


def test_連結財務諸表の範囲を見出しで特定する() -> None:
    sections = find_sections(PAGES)
    assert sections["BS"] == [1, 2]  # 貸借対照表は2ページにわたる
    assert sections["PL"] == [3, 4] or sections["PL"] == [3]
    assert sections["CF"] == [5]  # 注記事項のページは含めない


def test_本文中の言及や目次は範囲に含めない() -> None:
    assert 0 not in find_sections(PAGES)["BS"]


def test_キャッシュフロー計算書は見出しから最大3ページまで() -> None:
    cf = CF_PAGE
    more = "EDINET提出書類\n(単位:百万円)\n減価償却費 1 1\n"
    sections = find_sections([BS_PAGE, PL_PAGE, cf, more, more, more, more])
    assert sections["CF"] == [2, 3, 4]


# ---- 項目の抽出 ----


def _items():
    return extract_items(PAGES)


def test_貸借対照表の項目を円で読む() -> None:
    items = _items()
    assert items["total_assets"].value == D(181_811) * 1_000_000
    assert items["current_assets"].value == D(117_172) * 1_000_000
    assert items["current_liabilities"].value == D(34_982) * 1_000_000  # 2ページ目から
    assert items["net_assets"].value == D(138_986) * 1_000_000


def test_損益計算書の項目を当期の列から読む() -> None:
    items = _items()
    assert items["net_sales"].value == D(139_657) * 1_000_000
    assert items["operating_income"].value == D(18_349) * 1_000_000
    assert items["interest_income"].value == D(68) * 1_000_000
    assert items["dividend_income"].value == D(135) * 1_000_000
    assert items["interest_expense"].value == D(23) * 1_000_000
    assert items["net_income_attributable_to_owners"].value == D(13_648) * 1_000_000


def test_減価償却費はキャッシュフロー計算書から読む() -> None:
    # 損益計算書の販管費の内訳にも「減価償却費」が出るが、正解データはCFの値
    assert _items()["depreciation"].value == D(2_445) * 1_000_000


def test_支払利息は損益計算書の値でありCFの値ではない_同額でも範囲で選ぶ() -> None:
    item = _items()["interest_expense"]
    assert item.section == "PL"


def test_読み取った場所を記録する() -> None:
    item = _items()["total_assets"]
    assert isinstance(item, ExtractedValue)
    assert item.page == 2  # 1始まりのページ番号
    assert "資産合計" in item.line


def test_行が無い項目はNoneで_理由を持つ() -> None:
    items = extract_items([BS_PAGE, PL_PAGE, CF_PAGE])  # 純資産合計・流動負債合計のページが無い
    assert items["net_assets"].value is None
    assert items["net_assets"].reason == "label_not_found"


def test_範囲が見つからなければ理由はsection_not_found() -> None:
    items = extract_items([INDEX_PAGE])
    assert items["total_assets"].value is None
    assert items["total_assets"].reason == "section_not_found"


def test_単位が分からなければ抽出しない() -> None:
    page = BS_PAGE.replace("(単位:百万円)", "")
    items = extract_items([page, PL_PAGE, CF_PAGE])
    assert items["total_assets"].value is None
    assert items["total_assets"].reason == "unit_not_found"


# ---- 実データの失敗から: ラベルの折り返し・合算の行 ----


def _pl_with(*lines: str) -> str:
    body = "\n".join(lines)
    return f"""EDINET提出書類
サンプル株式会社(E00000)
有価証券報告書
2【連結損益計算書及び連結包括利益計算書】
(1)【連結損益計算書】
(単位:百万円)
前連結会計年度 当連結会計年度
売上高 100 200
営業利益 10 20
{body}
"""


def test_ラベルが2行に折り返されて_数値が2行目にある() -> None:
    page = _pl_with(
        "親会社株主に帰属する当期純利益又は親会社株",
        "主に帰属する当期純損失(△) 2,436 △1,843",
    )
    item = extract_items([BS_PAGE, page, CF_PAGE])["net_income_attributable_to_owners"]
    assert item.value == D(-1_843) * 1_000_000


def test_ラベルの行の次が数値だけの行() -> None:
    page = _pl_with("親会社株主に帰属する当期純損失", "△1,843 △1,843", "(△)")
    item = extract_items([BS_PAGE, page, CF_PAGE])["net_income_attributable_to_owners"]
    assert item.value == D(-1_843) * 1_000_000


def test_直前の見出し行が混ざっても_本来のラベルで読める() -> None:
    page = _pl_with("営業外収益", "受取利息 32 68")
    assert extract_items([BS_PAGE, page, CF_PAGE])["interest_income"].value == D(68) * 1_000_000


def test_折り返しの結合が_無関係な項目に誤って一致しない() -> None:
    # 「営業外収益」+「支払利息」を結合しても、支払利息の項目にしか使われない
    page = _pl_with("営業外収益", "受取配当金 111 135")
    items = extract_items([BS_PAGE, page, CF_PAGE])
    assert items["dividend_income"].value == D(135) * 1_000_000
    assert items["interest_expense"].value is None


def test_受取利息が合算の行しかない会社は_合算を受取利息として読む() -> None:
    # XBRL の正解データ側も、合算の行しかない会社は合算を受取利息として扱う
    page = _pl_with("受取利息及び配当金 156 175")
    items = extract_items([BS_PAGE, page, CF_PAGE])
    assert items["interest_income"].value == D(175) * 1_000_000
    assert items["dividend_income"].value is None


def test_受取利息の行があれば_合算の行より優先する() -> None:
    page = _pl_with("受取利息 32 68", "受取利息及び配当金 156 175")
    assert extract_items([BS_PAGE, page, CF_PAGE])["interest_income"].value == D(68) * 1_000_000


# ---- コードレビューでの指摘への対応 ----


def test_損失のラベルは正の数で印字される_符号を負にする() -> None:
    # 「営業損失」は損失額を正の数で印字する。XBRL では営業利益が負の値
    page = _pl_with("営業損失 3,536 647").replace("営業利益 10 20\n", "")
    item = extract_items([BS_PAGE, page, CF_PAGE])["operating_income"]
    assert item.value == D(-647) * 1_000_000


def test_利益又は損失_のラベルは印字された符号のまま読む() -> None:
    # 「営業利益又は営業損失(△)」は、損失なら △ 付きで印字される
    page = _pl_with("営業利益又は営業損失(△) 3,536 △647").replace("営業利益 10 20\n", "")
    assert extract_items([BS_PAGE, page, CF_PAGE])["operating_income"].value == D(-647) * 1_000_000


def test_親会社株主に帰属する当期純損失も_正の数で印字されたら負にする() -> None:
    page = _pl_with("親会社株主に帰属する当期純損失 2,436 1,843")
    item = extract_items([BS_PAGE, page, CF_PAGE])["net_income_attributable_to_owners"]
    assert item.value == D(-1_843) * 1_000_000


SINGLE_COLUMN_PL = """EDINET提出書類
サンプル株式会社(E00000)
有価証券報告書
2【連結損益計算書及び連結包括利益計算書】
(1)【連結損益計算書】
(単位:百万円)
当連結会計年度
売上高 139,657
営業利益 18,349
"""

SINGLE_COLUMN_BS = """EDINET提出書類
サンプル株式会社(E00000)
有価証券報告書
1【連結貸借対照表】
(単位:百万円)
当連結会計年度
資産の部
資産合計 181,811
"""


def test_初回の有報など_1列だけの財務諸表も読む() -> None:
    items = extract_items([SINGLE_COLUMN_BS, SINGLE_COLUMN_PL, CF_PAGE])
    assert items["net_sales"].value == D(139_657) * 1_000_000
    assert items["total_assets"].value == D(181_811) * 1_000_000


def test_2列の財務諸表で片方しか数値が無い行は_読まない() -> None:
    # 前期の列を持つ表で、数値が1つの行は、列の対応が分からないので読まない
    page = _pl_with("受取利息 68")
    assert extract_items([BS_PAGE, page, CF_PAGE])["interest_income"].value is None


# ---- 本文のキャッシュ ----


def _pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    return pdf


def test_本文のキャッシュを使い_2回目は読まない(tmp_path: Path) -> None:
    calls: list[Path] = []

    def reader(path: Path) -> list[str]:
        calls.append(path)
        return ["p1", "p2"]

    pdf = _pdf(tmp_path)
    assert cached_pages(pdf, tmp_path / "cache", reader=reader) == ["p1", "p2"]
    assert cached_pages(pdf, tmp_path / "cache", reader=reader) == ["p1", "p2"]
    assert len(calls) == 1


def test_抽出方法の版が変わったら_古いキャッシュを使わない(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ingest.pdf_baseline as mod

    pdf = _pdf(tmp_path)
    cached_pages(pdf, tmp_path / "cache", reader=lambda _: ["古い本文"])

    monkeypatch.setattr(mod, "TEXT_EXTRACTION_VERSION", mod.TEXT_EXTRACTION_VERSION + 1)
    assert cached_pages(pdf, tmp_path / "cache", reader=lambda _: ["新しい本文"]) == ["新しい本文"]


@pytest.mark.parametrize(
    "content", ["", "これはJSONではない", "[]", "{}", '{"version": 999, "pages": ["x"]}']
)
def test_壊れた_空の_版の違うキャッシュは使わない(tmp_path: Path, content: str) -> None:
    pdf = _pdf(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "x.json").write_text(content, encoding="utf-8")

    assert cached_pages(pdf, cache, reader=lambda _: ["読み直した"]) == ["読み直した"]


def test_ページが0件の抽出結果はキャッシュしない(tmp_path: Path) -> None:
    cached_pages(_pdf(tmp_path), tmp_path / "cache", reader=lambda _: [])
    assert not list((tmp_path / "cache").glob("*.json"))
