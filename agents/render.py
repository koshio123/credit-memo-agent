"""メモの下書きを、テンプレート（templates/credit_memo.md）の構成の Markdown にする。

表と出典の番号付けはコードが行う。LLM が書いた主張は、出典の番号を末尾に付けて並べる。
"""

from collections.abc import Sequence

from agents.evidence import DEBT_LABELS, LEVEL_RATIOS
from agents.memo import Flagged, MemoDraft
from agents.pipeline import MemoResult
from agents.state import CLAIM_SECTIONS, Claim, EvidencePool, MetricEvidence, PassageEvidence

_QUOTE_CHARS = 80
_BASIS_CHARS = 120


def _cell(text: str) -> str:
    """表のセルに入れられる形にする。空白をまとめ、縦線をエスケープする。

    Args:
        text: セルの文字列。

    Returns:
        1行に収めた文字列。
    """
    return " ".join(text.split()).replace("|", "\\|")


def _clip(text: str, limit: int) -> str:
    """空白をまとめ、長ければ切って「…」を付ける。

    Args:
        text: 元の文字列。
        limit: 最大文字数。

    Returns:
        切った文字列。
    """
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


class _Refs:
    """出典の番号。本文に最初に出た順に 1 から振り、同じ証拠は同じ番号にする。"""

    def __init__(self) -> None:
        self.order: dict[str, int] = {}

    def of(self, evidence_ids: Sequence[str]) -> str:
        """証拠のIDを、出典の番号に直す。初めて出たIDには次の番号を振る。

        Args:
            evidence_ids: 証拠のID。

        Returns:
            例: "[1][3]"。同じ番号は一度だけ。
        """
        numbers: list[int] = []
        for evidence_id in evidence_ids:
            number = self.order.setdefault(evidence_id, len(self.order) + 1)
            if number not in numbers:
                numbers.append(number)
        return "".join(f"[{n}]" for n in numbers)


def _claims(claims: Sequence[Claim], refs: _Refs, warned: dict[str, str]) -> list[str]:
    """節の主張を、出典の番号つきの箇条書きにする。

    Args:
        claims: 節の主張。
        refs: 出典の番号の振り分け。
        warned: 主張の本文 → 末尾に付ける印（数値の警告、検証者が判断できなかった、など）。

    Returns:
        Markdown の行。主張が無ければ、その旨の1行。
    """
    if not claims:
        return ["- （主張を得られなかった。資料に記載がないことを意味しない。検査の記録を参照）"]
    return [
        f"- {c.text} {refs.of(c.evidence_ids)}".rstrip() + warned.get(c.text, "") for c in claims
    ]


def _pages(item: PassageEvidence) -> str:
    """本文の証拠のページを表記にする。

    Args:
        item: 本文の証拠。

    Returns:
        例: "p.12"、"p.12-13"。
    """
    pages = sorted({s.page for s in item.spans})
    return f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"


def _source_row(number: int, item: MetricEvidence | PassageEvidence) -> str:
    """出典一覧の1行を作る。

    Args:
        number: 出典の番号。
        item: 本文または数値の証拠。

    Returns:
        Markdown の表の行。
    """
    if isinstance(item, PassageEvidence):
        where = f"{item.heading_path[-1]}: " if item.heading_path else ""
        doc = item.spans[0].doc_id
        company = item.company
        return (
            f"| {number} | {_cell(company)} 有価証券報告書（{doc}） | {_pages(item)} | "
            f"{_cell(_clip(where + item.text, _QUOTE_CHARS))} |"
        )
    pages = ", ".join(str(p) for p in item.pdf_pages)
    company = _cell(item.company)
    kind = (
        "語句検索の記録（コードによる）" if item.origin == "search_record" else "財務データ（XBRL）"
    )
    return (
        f"| {number} | {company} 有価証券報告書（{item.doc_id}）の{kind} | "
        f"{'PDF p.' + pages if pages else '—'} | "
        f"{_cell(item.label)} {_cell(item.display)}: {_cell(_clip(item.basis, _BASIS_CHARS))} |"
    )


def _header(memo: MemoDraft) -> list[str]:
    """メモの見出しと、対象・注意書きの行を作る。

    Args:
        memo: 下書きのメモ。

    Returns:
        Markdown の行。
    """
    return [
        "# 与信メモ（草案）",
        "",
        f"- 対象企業: {memo.company}（証券コード {memo.sec_code}）",
        f"- 対象期間: {memo.period_end} 期 および {memo.previous_period_end} 期",
        f"- 資料: {memo.company} 有価証券報告書（EDINET 書類ID {memo.doc_id}）",
        "- 作成: システムによる草案。**人によるレビューを経るまで、審査資料として使用しない。**",
        "",
        "> このメモは判断材料の整理であり、融資の可否・金利・限度額・担保の要否について、"
        "結論も推奨も含まない（与信管理規程 第3条）。",
        "",
    ]


def _going_concern(memo: MemoDraft, pool: EvidencePool, refs: _Refs) -> list[str]:
    """最優先の確認事項（継続企業の前提）の節を作る。

    Args:
        memo: 下書きのメモ。
        pool: 証拠の集まり。
        refs: 出典の番号の振り分け。

    Returns:
        Markdown の行。記載があれば引用を付ける。
    """
    row = next(r for r in memo.policy_rows if r.clause == "第13条")
    cite = refs.of(row.evidence_ids)
    if memo.going_concern_signal:
        evidence = pool.get(row.evidence_ids[0])
        quote = _clip(evidence.text, 160) if isinstance(evidence, PassageEvidence) else ""
        return [
            "**最優先で確認する**: 継続企業の前提に関する記載（重要事象等・重要な疑義・"
            f"重要な不確実性）が見つかった。{cite}",
            "",
            f"> {quote}",
        ]
    return [
        "継続企業の前提に関する記載（重要事象等・重要な疑義・重要な不確実性が存在する旨）は、"
        f"語句の検索では確認できなかった（言い換えは検出できない）。{cite}"
    ]


def _metric_table(result: MemoResult, refs: _Refs) -> list[str]:
    """主要指標の2期比較の表を作る。

    Args:
        result: 生成結果。
        refs: 出典の番号の振り分け。

    Returns:
        Markdown の行（表と注記）。
    """
    lines = [
        "| 指標 | 前期 | 直近期 | 規程の水準（直近期） | 計算根拠 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in (*LEVEL_RATIOS, "sales_growth"):
        cur = result.metrics[f"{name}.current"]
        prev = result.metrics.get(f"{name}.previous")
        prev_cell = f"{prev.display} {refs.of([prev.id])}" if prev else "—"
        lines.append(
            f"| {_cell(cur.label)} | {_cell(prev_cell)} | {_cell(cur.display)} | "
            f"{cur.level or '—'} | {_cell(_clip(cur.basis, _BASIS_CHARS))} {refs.of([cur.id])} |"
        )
    lines += [
        "",
        "- 数値はすべてコードで算定したものを記載する。算式は与信管理規程 附則第2条による。",
        "- 算定できない指標は「算定不能」とし、理由を付す（規程 第7条）。",
    ]
    return lines


def _debt_table(result: MemoResult, refs: _Refs) -> list[str]:
    """有利子負債の構成の表を作る。

    Args:
        result: 生成結果。
        refs: 出典の番号の振り分け。

    Returns:
        Markdown の表の行。
    """
    lines = ["| 項目 | 金額・割合 | 根拠 |", "| --- | --- | --- |"]
    keys = [*DEBT_LABELS, "interest_bearing_debt", "debt_due_within_1y_ratio"]
    for key in keys:
        m = result.metrics.get(f"{key}.current")
        if m is None:
            continue
        label = m.label.replace("（直近期）", "")
        lines.append(
            f"| {_cell(label)} | {_cell(m.display)} | "
            f"{_cell(_clip(m.basis, _BASIS_CHARS))} {refs.of([m.id])} |"
        )
    return lines


def _policy_table(memo: MemoDraft, refs: _Refs) -> list[str]:
    """内規への照合結果の表を作る。

    Args:
        memo: 下書きのメモ。
        refs: 出典の番号の振り分け。

    Returns:
        Markdown の表の行。
    """
    lines = ["| 条項 | 確認内容 | 結果 | 根拠 |", "| --- | --- | --- | --- |"]
    for row in memo.policy_rows:
        lines.append(
            f"| {row.clause} | {_cell(row.check)} | {_cell(row.result)} | "
            f"{refs.of(row.evidence_ids)} |"
        )
    return lines


def _open_items(
    memo: MemoDraft, result: MemoResult, refs: _Refs, warned: dict[str, str]
) -> list[str]:
    """確認が必要な事項を作る。LLM が挙げたものに、算定不能の指標と借換えへの依存の確認を足す。

    Args:
        memo: 下書きのメモ。
        result: 生成結果。
        refs: 出典の番号の振り分け。
        warned: 主張の本文 → 末尾に付ける印（数値の警告、検証者が判断できなかった、など）。

    Returns:
        Markdown の行。何も無ければ、その旨の1行。
    """
    lines = _claims(memo.open_items, refs, warned) if memo.open_items else []
    for name in LEVEL_RATIOS:
        m = result.metrics[f"{name}.current"]
        if m.value is None:
            lines.append(f"- {m.label}: {m.basis}。確認が必要。 {refs.of([m.id])}")
    debt = result.metrics.get("interest_bearing_debt.current")
    if debt is not None:
        lines.append(
            "- 有利子負債の借換えへの依存の有無は、財務データだけでは判断できない。"
            f"確認が必要（規程 第15条）。 {refs.of([debt.id])}"
        )
    return lines or ["- （該当する記載はありません）"]


def _flag_lines(items: Sequence[Flagged]) -> list[str]:
    """検査に引っかかった主張を、箇条書きにする。

    Args:
        items: 検査に引っかかった主張。

    Returns:
        Markdown の行。無ければ「なし」。
    """
    return [
        f"- [{f.section}] {f.claim.text}（{'; '.join(i.message for i in f.issues)}）" for f in items
    ] or ["- なし"]


def _verifier_lines(memo: MemoDraft) -> list[str]:
    """Verifier による検証の要約。実施していなければ、意味は未検証と明記する。

    Args:
        memo: 下書きのメモ。

    Returns:
        Markdown の行。
    """
    if not memo.rounds:
        return [
            "**Verifier による検証は実施していない。主張が出典の内容に支えられているか（意味）は"
            "未検証。**"
        ]
    lines = [
        "Verifier（LLM）で、主張が出典の内容に支えられているかを検証した。"
        "LLM による判定で、誤ることがある（人の判定との一致率は未測定）。"
    ]
    for r in memo.rounds:
        counts = "、".join(f"{k} {v}" for k, v in sorted(r.counts.items()))
        lines.append(f"- ラウンド {r.round_no}: {r.n_verified} 件を検証（{counts}）")
    return lines


def _inspection(memo: MemoDraft) -> list[str]:
    """検査の記録の節を作る。

    Args:
        memo: 下書きのメモ。

    Returns:
        Markdown の行。
    """
    empty = [
        label + ("（LLM が挙げたもの）" if key == "open_items" else "")
        for key, label in CLAIM_SECTIONS
        if not getattr(memo, key)
    ]
    return [
        "## 検査の記録",
        "",
        "このメモに対して、コードで次の検査をした: 出典の有無、出典の存在、数値が出典と合うか、"
        "結論を示す語。",
        "",
        *_verifier_lines(memo),
        "",
        "本文に指示の形が混入していたため、証拠にしなかった箇所: "
        + (
            "、".join(f"p.{q.page_start}（{'・'.join(q.rules)}）" for q in memo.quarantined)
            if memo.quarantined
            else "なし"
        ),
        "",
        "主張が1件も得られなかった節: " + ("、".join(empty) if empty else "なし"),
        "",
        "生成に失敗した節（LLM の出力が指定の形を満たせなかった）: "
        + (
            "、".join(f"{f.section}（{f.message[:80]}）" for f in memo.failures)
            if memo.failures
            else "なし"
        ),
        "",
        "検査で除外した主張（メモの本文には入れていない）:",
        *_flag_lines(memo.rejected),
        "",
        "数値などの警告があるが、残した主張:",
        *_flag_lines(memo.warnings),
        "",
    ]


def render_memo(result: MemoResult) -> str:
    """メモの下書きを、テンプレートの構成の Markdown にする。

    Args:
        result: 生成結果。

    Returns:
        Markdown の文字列。
    """
    memo, pool = result.memo, result.pool
    refs = _Refs()
    warned = {f.claim.text: " ⚠数値要確認" for f in memo.warnings}
    for record in memo.verifications:
        if record.verdict == "cannot_judge":
            warned[record.claim_text] = (
                warned.get(record.claim_text, "") + " ❓検証者が判断できなかった"
            )
    out: list[str] = _header(memo)
    out += ["## 0. 最優先の確認事項", "", *_going_concern(memo, pool, refs), ""]
    out += ["## 1. 企業概要", "", *_claims(memo.overview, refs, warned), ""]
    out += ["## 2. 財務分析", "", "### 2.1 主要指標（2期比較）", ""]
    out += [*_metric_table(result, refs), ""]
    out += ["### 2.2 所見", "", *_claims(memo.financial_findings, refs, warned), ""]
    out += ["### 2.3 有利子負債の構成", "", *_debt_table(result, refs), ""]
    out += ["## 3. 事業リスク", "", *_claims(memo.business_risks, refs, warned), ""]
    out += ["## 4. 内規への照合結果", "", *_policy_table(memo, refs), ""]
    out += ["## 5. 与信判断上の論点", "", "### 肯定的な要素", ""]
    out += [*_claims(memo.positives, refs, warned), "", "### 否定的な要素", ""]
    out += [*_claims(memo.negatives, refs, warned), "", "（結論は記載しない。規程 第17条）", ""]
    out += ["## 6. 確認が必要な事項", "", *_open_items(memo, result, refs, warned), ""]

    out += ["## 出典一覧", "", "| 番号 | 資料 | ページ | 該当箇所 |", "| --- | --- | --- | --- |"]
    by_number = sorted((number, evidence_id) for evidence_id, number in refs.order.items())
    out += [_source_row(number, pool.get(evidence_id)) for number, evidence_id in by_number]
    out += ["", *_inspection(memo)]
    return "\n".join(out)
