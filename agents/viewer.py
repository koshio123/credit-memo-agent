"""引用ビューア: 保存した結果から、単体の静的 HTML を作る（サーバー不要）。

主張の出典番号を押すと、引用文・ページ・算式が出る。PDF へのリンクは、基底の URL を渡したときだけ
付ける。**書類の本体（PDF・ページ全体の本文）は埋め込まない**（EDINET の利用規約を確認するまで
再配布しない）。埋め込むのは、メモが引用している範囲の引用文だけ。外部の資源は読み込まない。
"""

from collections.abc import Sequence
from html import escape

from agents.export import SavedResult
from agents.state import CLAIM_SECTIONS, Claim, MetricEvidence, PassageEvidence

_CSS = """
:root{--bg:#fff;
--fg:#1c1c1e;
--muted:#6b6b70;
--line:#d8d8dc;
--accent:#1a5fb4;
--panel:#f4f6fa;

--ok:#1b7f3b;
--warn:#a15c00;
--ng:#b3261e}
@media (prefers-color-scheme: dark){:root{--bg:#16171a;
--fg:#ececf0;
--muted:#9a9aa3;
--line:#33343a;

--accent:#7ab0ff;
--panel:#1e2026;
--ok:#58c27d;
--warn:#e0a548;
--ng:#ff8a80}}
body{background:var(--bg);
color:var(--fg);
font:16px/1.7 system-ui,"Hiragino Sans",sans-serif;

margin:0 auto;
max-width:860px;
padding:16px}
h1{font-size:1.4rem}h2{font-size:1.15rem;
margin-top:2rem;
border-bottom:1px solid var(--line)}
.note{color:var(--muted);
font-size:.9rem}
li{margin:.5rem 0}
button.cite{font:inherit;
font-size:.8rem;
color:var(--accent);
background:none;
border:1px solid var(--line);

border-radius:4px;
padding:0 .35rem;
margin-left:.2rem;
cursor:pointer}
button.cite:hover,button.cite[aria-expanded=true]{background:var(--panel)}
.panel{background:var(--panel);
border-left:3px solid var(--accent);
padding:.5rem .8rem;
margin:.4rem 0;

font-size:.92rem}
.panel[hidden]{display:none}
@media print{.panel[hidden],tr[hidden]{display:block!important}}
blockquote{margin:.3rem 0;
padding-left:.6rem;
border-left:2px solid var(--line);
white-space:pre-wrap}
.tag{font-size:.78rem;
border-radius:4px;
padding:0 .35rem;
margin-left:.3rem;
border:1px solid currentColor}
.verdict-supported{color:var(--ok)}.verdict-partial,.verdict-cannot_judge{color:var(--warn)}
.verdict-unsupported{color:var(--ng)}
table{border-collapse:collapse;
width:100%;
font-size:.92rem}
td,th{border:1px solid var(--line);
padding:.3rem .5rem;
text-align:left}
@media (max-width:600px){td,th{padding:.2rem .3rem}}
"""

_JS = """
document.addEventListener('click',function(e){
  var b=e.target.closest('button.cite'); if(!b) return;
  var p=document.getElementById(b.dataset.target); if(!p) return;
  var open=p.hasAttribute('hidden');
  if(open){p.removeAttribute('hidden')}else{p.setAttribute('hidden','')}
  var row=p.closest('tr');
  if(row){ if(open){row.removeAttribute('hidden')}else{row.setAttribute('hidden','')} }
  b.setAttribute('aria-expanded', open?'true':'false');
});
"""


def _pdf_link(doc_id: str, page: int, pdf_base: str | None) -> str:
    """PDF のページへのリンク（基底の URL があるときだけ）。"""
    if not pdf_base:
        return ""
    href = pdf_base.replace("{doc_id}", doc_id) + f"#page={page}"
    return f' <a href="{escape(href, quote=True)}">PDF を開く</a>'


def _panel(
    panel_id: str,
    number: int,
    evidence: PassageEvidence | MetricEvidence,
    pdf_base: str | None,
    doc_id: str,
) -> str:
    """出典番号を押したときに出る欄。"""
    if isinstance(evidence, PassageEvidence):
        heading = escape(" > ".join(evidence.heading_path))
        body = "".join(
            f"<div>{escape(s.doc_id)} p.{s.page}{_pdf_link(s.doc_id, s.page, pdf_base)}"
            f"<blockquote>{escape(s.quote)}</blockquote></div>"
            for s in evidence.spans
        )
        inner = f'<div class="note">本文（{heading}）</div>{body}'
    else:
        pages = "".join(
            f" p.{p}{_pdf_link(evidence.doc_id or doc_id, p, pdf_base)}" for p in evidence.pdf_pages
        )
        where = f"PDF{pages}" if pages else "PDF のページなし"
        inner = (
            f'<div class="note">数値（{escape(evidence.label)} '
            f"{escape(evidence.display)}。{where}）"
            f"</div><blockquote>{escape(evidence.basis)}</blockquote>"
        )
    return f'<div class="panel" id="{panel_id}" hidden><b>[{number}]</b> {inner}</div>'


def render_html(saved: SavedResult, pdf_base: str | None = None) -> str:
    """メモと出典を、単体の HTML にする。

    Args:
        saved: 保存した生成結果。
        pdf_base: PDF の場所のひな形（例: file:///…/{doc_id}/{doc_id}.pdf）。None ならリンクしない。

    Returns:
        HTML 文書の文字列。
    """
    memo = saved.memo
    order: dict[str, int] = {}
    counter = 0
    verdicts = {(v.section, v.claim_text): v for v in memo.verifications}

    def cite(evidence_ids: Sequence[str]) -> tuple[str, list[str]]:
        """引用ごとに、押すボタンと、その直後に置く隠した欄（引用ごとに別の id）を作る。"""
        nonlocal counter
        buttons: list[str] = []
        panels: list[str] = []
        for evidence_id in evidence_ids:
            n = order.setdefault(evidence_id, len(order) + 1)
            counter += 1
            panel_id = f"p{counter}"
            buttons.append(
                f'<button class="cite" data-target="{panel_id}" aria-expanded="false">{n}</button>'
            )
            panels.append(_panel(panel_id, n, saved.evidence[evidence_id], pdf_base, memo.doc_id))
        return "".join(buttons), panels

    body: list[str] = []
    for key, label in CLAIM_SECTIONS:
        claims: list[Claim] = getattr(memo, key)
        if not claims:
            continue
        body.append(f"<h2>{escape(label)}</h2><ul>")
        for claim in claims:
            judged = verdicts.get((key, claim.text))
            tag = ""
            if judged and judged.verdict != "not_applicable":
                tag = (
                    f'<span class="tag verdict-{escape(judged.verdict)}" '
                    f'title="{escape(judged.reason, quote=True)}">'
                    f"Verifier: {escape(judged.verdict)}</span>"
                )
                if judged.reason:
                    tag += f'<span class="note"> {escape(judged.reason)}</span>'
            buttons, panels = cite(claim.evidence_ids)
            body.append(f"<li>{escape(claim.text)}{buttons}{tag}{''.join(panels)}</li>")
        body.append("</ul>")

    body.append(
        "<h2>内規への照合結果</h2>"
        "<table><tr><th>条項</th><th>確認内容</th><th>結果</th><th>根拠</th></tr>"
    )
    for row in memo.policy_rows:
        buttons, panels = cite(row.evidence_ids)
        body.append(
            f"<tr><td>{escape(row.clause)}</td><td>{escape(row.check)}</td>"
            f"<td>{escape(row.result)}</td><td>{buttons}</td></tr>"
        )
        # 欄は行の下に、引用ごとに1行で置く（隠しておき、押すと開く）
        body.extend(f'<tr hidden><td colspan="4">{panel}</td></tr>' for panel in panels)
    body.append("</table>")

    rounds = "".join(
        f"<li>ラウンド {r.round_no}: {r.n_verified} 件を検証（"
        + escape("、".join(f"{k} {v}" for k, v in sorted(r.counts.items())))
        + "）</li>"
        for r in memo.rounds
    )
    inspection = (
        f"<h2>検査の記録</h2><ul>{rounds}</ul>"
        if rounds
        else '<h2>検査の記録</h2><p class="note">Verifier 未実施（意味は未検証）。</p>'
    )
    inspection += (
        f'<p class="note">除外した主張 {len(memo.rejected)} 件、警告つきで残した主張 '
        f"{len(memo.warnings)} 件、隔離した本文 {len(memo.quarantined)} 件。</p>"
    )
    title = f"{memo.company}（{memo.sec_code}）与信メモ（草案）"
    return (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_CSS}</style>"
        "<noscript><style>[hidden]{display:block!important}</style></noscript></head><body>"
        f"<h1>{escape(title)}</h1>"
        '<p class="note">システムによる草案。人によるレビューを経るまで、審査資料として使用しない。'
        "融資の可否・金利・限度額・担保の要否について、結論も推奨も含まない。"
        "出典の番号を押すと、引用文・ページ・算式が出る。</p>"
        f"{''.join(body)}{inspection}"
        f"<script>{_JS}</script></body></html>"
    )
