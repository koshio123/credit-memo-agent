"""引用ビューア: 保存した結果から、単体の静的 HTML を作る。"""

import re

import pytest

from agents.export import SavedResult
from agents.viewer import render_html
from edinet_mcp.service import EdinetService

from .test_review import make_claim, make_saved

pytestmark = pytest.mark.anyio


async def test_主張に出典の番号のボタンがつき_押すと出る証拠の欄がある(
    service: EdinetService,
) -> None:
    html = render_html(await make_saved(service))

    assert html.startswith("<!doctype html>")
    assert "機械を製造している。" in html
    buttons = re.findall(r'<button class="cite" data-target="(p\d+)"', html)
    assert buttons
    for target in set(buttons):
        assert f'id="{target}"' in html  # 押した先の証拠の欄がある


async def test_証拠の欄には_引用文_ページ_算式が入る(service: EdinetService) -> None:
    html = render_html(await make_saved(service))
    assert "原材料価格が高騰する可能性があります" in html  # 本文の引用
    assert "p.2" in html
    assert "純資産" in html and "総資産" in html  # 数値の算式


async def test_PDFのリンクは_基底のURLを渡したときだけ_ページつきで付く(
    service: EdinetService,
) -> None:
    saved = await make_saved(service)
    assert "#page=" not in render_html(saved)
    html = render_html(saved, pdf_base="file:///data/edinet/{doc_id}/{doc_id}.pdf")
    assert 'href="file:///data/edinet/S100CUR0/S100CUR0.pdf#page=2"' in html


async def test_文字は_エスケープする_スクリプトを混ぜられない(service: EdinetService) -> None:
    saved = await make_saved(
        service, {"overview": [make_claim("<script>alert(1)</script>と書いてある。", "E1")]}
    )
    html = render_html(saved)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


async def test_外部の資源を読み込まない(service: EdinetService) -> None:
    html = render_html(await make_saved(service))
    assert not re.search(r'(src|href)="https?://', html)
    assert "<link " not in html


async def test_検査の結果として_判定と隔離を示す(service: EdinetService) -> None:
    from agents.memo import VerificationRecord

    saved = await make_saved(service)
    text = saved.memo.overview[0].text
    saved.memo.verifications = [
        VerificationRecord(
            section="overview", claim_text=text, verdict="partial", reason="言い過ぎ"
        )
    ]
    html = render_html(saved)
    assert "言い過ぎ" in html and "verdict-partial" in html


async def test_書類の本体は埋め込まない_引用している範囲だけ(service: EdinetService) -> None:
    html = render_html(await make_saved(service))
    assert "従業員の状況" not in html  # 引かれていないページの本文は出ない


def test_型の確認() -> None:
    assert SavedResult


async def test_PDFのひな形に他の波括弧があっても_落ちない(service: EdinetService) -> None:
    html = render_html(await make_saved(service), pdf_base="file:///a{b}/{doc_id}.pdf")
    assert 'href="file:///a{b}/S100CUR0.pdf#page=2"' in html


async def test_出典の欄は_押した項目の中にあり_ページ末尾にまとめない(
    service: EdinetService,
) -> None:
    # 実機（ブラウザ）で、欄がページ末尾に開き、押しても何も起きないように見えた
    html = render_html(await make_saved(service))
    assert "<h2>出典</h2>" not in html and 'id="sources"' not in html
    for li in re.findall(r"<li>.*?</li>", html, re.S):
        for target in re.findall(r'data-target="(p\d+)"', li):
            assert f'id="{target}"' in li  # 欄は、押す項目と同じ段落の中にある


async def test_欄のidは引用ごとに別で_重複しない_同じ証拠を複数の主張が引いても(
    service: EdinetService,
) -> None:
    saved = await make_saved(
        service,
        {"overview": [make_claim("a。", "E1"), make_claim("b。", "E1")]},
    )
    html = render_html(saved)
    ids = re.findall(r'<div class="panel" id="(p\d+)"', html)
    assert len(ids) == len(set(ids))
    targets = re.findall(r'data-target="(p\d+)"', html)
    assert sorted(targets) == sorted(ids)  # ボタンと欄が1対1


async def test_表の根拠も_行の直後に隠した行として置き_1つの欄を1つのボタンで開く(
    service: EdinetService,
) -> None:
    html = render_html(await make_saved(service))
    rows = re.findall(r"<tr><td>第.*?</tr>(?:<tr hidden>.*?</tr>)*", html, re.S)
    assert rows
    for row in rows:
        n_buttons = row.count('class="cite"')
        assert row.count("<tr hidden>") == n_buttons


async def test_JSなしや印刷でも_欄を読めるようにする(service: EdinetService) -> None:
    html = render_html(await make_saved(service))
    assert "@media print" in html and "<noscript><style>[hidden]" in html
