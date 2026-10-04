"""人が、主張と出典を突き合わせるための確認用の出力。

主張と、その出典の引用文・ページ・算式を並べ、判定の欄をつける。機械検査の警告と、内規照合の各行も
同じ形で確かめられる。Verifier（LLM）の判定は、通常は主張の下に出す。**blind=True では出さない**:
先に人が判定してから Verifier の判定と比べる（一致率を測る）ときに、人の判定が引きずられないように
するため。blind の出力は、別のファイル名（.blind.review.md）で保存する。書き直された主張かどうかも、
blind では示さない。
"""

from agents.export import SavedResult
from agents.state import CLAIM_SECTIONS, MetricEvidence, PassageEvidence

_JUDGEMENT = [
    "  - 判定:",
    "    - [ ] 支持する",
    "    - [ ] 一部だけ支持する",
    "    - [ ] 支持しない",
    "    - [ ] 判断できない",
    "  - メモ: ",
    "",
]


def _evidence_lines(evidence: PassageEvidence | MetricEvidence) -> list[str]:
    """証拠を、確認用の箇条書きの行にする。

    Args:
        evidence: 本文または数値の証拠。

    Returns:
        Markdown の行のリスト。
    """
    if isinstance(evidence, PassageEvidence):
        heading = " > ".join(evidence.heading_path)
        lines = [f"  - **{evidence.id}** 本文（{heading}）"]
        for span in evidence.spans:
            quoted = span.quote.replace("\n", "\n      > ")
            lines.append(f"    - {span.doc_id} p.{span.page}: \n      > {quoted}")
        return lines
    pdf = ", ".join(str(p) for p in evidence.pdf_pages)
    pages = f"PDF p.{pdf}" if pdf else "PDF のページなし"
    return [
        f"  - **{evidence.id}** 数値 {evidence.label} {evidence.display}（{pages}）",
        f"    > {evidence.basis}",
    ]


def render_review(saved: SavedResult, blind: bool = False) -> str:
    """主張と出典を突き合わせるための確認用の Markdown を作る。

    Args:
        saved: 保存された生成結果。
        blind: True なら Verifier の判定を出さない（人の判定が引きずられないようにする）。

    Returns:
        判定の欄つきの Markdown。
    """
    memo = saved.memo
    warned = {f.claim.text: f for f in memo.warnings}
    verified = {(v.section, v.claim_text): v for v in memo.verifications}
    lines = [
        "# 主張と出典の突き合わせ",
        "",
        f"対象: {memo.company}（{memo.sec_code}）/ 構成: {saved.mode} / "
        f"LLM: {saved.backend.name} {saved.backend.model}",
        "",
        "各主張が、その出典の内容に支えられているかを確かめ、判定を付ける。"
        "本文は書類のページを開いて、引用が実際にそこにあることも確かめる。"
        "⚠は、機械検査が警告を出した主張。",
        "",
    ]
    number = 0
    for key, label in CLAIM_SECTIONS:
        claims = getattr(memo, key)
        if not claims:
            continue
        lines += [f"## {label}", ""]
        for claim in claims:
            number += 1
            lines += [f"### {number}. {claim.text}", ""]
            judged = verified.get((key, claim.text))
            if not blind and judged is not None and judged.verdict != "not_applicable":
                lines.append(f"  - Verifier（LLM）の判定: {judged.verdict}（{judged.reason}）")
            if claim.text in warned:
                reasons = "; ".join(i.message for i in warned[claim.text].issues)
                lines += [f"  - ⚠ 機械検査の警告: {reasons}"]
            if not claim.evidence_ids:
                lines.append("  - 出典なし（確認が必要な事項として挙げた項目）")
            for evidence_id in claim.evidence_ids:
                lines += _evidence_lines(saved.evidence[evidence_id])
            lines += ["", *_JUDGEMENT]
    lines += ["## 内規照合（転記の確認）", ""]
    for row in memo.policy_rows:
        number += 1
        lines += [f"### {number}. {row.clause} {row.check}: {row.result}", ""]
        for evidence_id in row.evidence_ids:
            lines += _evidence_lines(saved.evidence[evidence_id])
        lines += ["", *_JUDGEMENT]
    return "\n".join(lines)
