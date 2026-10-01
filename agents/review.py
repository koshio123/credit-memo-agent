"""人が、主張と出典を突き合わせるための確認用の出力。

メモの主張が出典の内容に支えられているか（意味）は、コードでは検証できない（W4 の Verifier の
仕事）。それまでは人が確かめる。主張と、その出典の引用文・ページを並べ、判定の欄をつける。
"""

from typing import Any

_SECTIONS = (
    ("overview", "企業概要"),
    ("financial_findings", "財務の所見"),
    ("business_risks", "事業リスク"),
    ("positives", "肯定的な要素"),
    ("negatives", "否定的な要素"),
    ("open_items", "確認が必要な事項"),
)


def _evidence_lines(evidence: dict[str, Any]) -> list[str]:
    if evidence["kind"] == "passage":
        pages = sorted({s["page"] for s in evidence["spans"]})
        where = f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"
        heading = " > ".join(evidence["heading_path"])
        quote = "\n".join(s["quote"] for s in evidence["spans"])
        quoted = "\n".join(f"    > {line}" for line in quote.splitlines())
        return [f"  - **{evidence['id']}** 本文 {where}（{heading}）", quoted]
    pdf = ", ".join(str(p) for p in evidence["pdf_pages"])
    pages = f"PDF p.{pdf}" if pdf else "PDF のページなし"
    return [
        f"  - **{evidence['id']}** 数値 {evidence['label']} {evidence['display']}（{pages}）",
        f"    > {evidence['basis']}",
    ]


def render_review(payload: dict[str, Any]) -> str:
    memo = payload["memo"]
    evidence = payload["evidence"]
    lines = [
        "# 主張と出典の突き合わせ",
        "",
        f"対象: {memo['company']}（{memo['sec_code']}）/ 構成: {payload['mode']} / "
        f"LLM: {payload['backend']['name']} {payload['backend']['model']}",
        "",
        "各主張が、その出典の内容に支えられているかを確かめ、判定を付ける。"
        "本文は書類のページを開いて、引用が実際にそこにあることも確かめる。",
        "",
    ]
    number = 0
    for key, label in _SECTIONS:
        claims = memo[key]
        if not claims:
            continue
        lines += [f"## {label}", ""]
        for claim in claims:
            number += 1
            lines.append(f"### {number}. {claim['text']}")
            lines.append("")
            ids = claim["evidence_ids"]
            if not ids:
                lines.append("  - 出典なし（確認が必要な事項として挙げた項目）")
            for evidence_id in ids:
                lines += _evidence_lines(evidence[evidence_id])
            lines += [
                "",
                "  - 判定:",
                "    - [ ] 支持する",
                "    - [ ] 一部だけ支持する",
                "    - [ ] 支持しない",
                "    - [ ] 判断できない",
                "  - メモ: ",
                "",
            ]
    return "\n".join(lines)
