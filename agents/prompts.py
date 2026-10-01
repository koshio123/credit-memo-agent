"""LLM への依頼文。証拠は ID つきの一覧で渡し、ID を選ばせる（引用文も数値も書かせない）。"""

from collections.abc import Sequence

from agents.state import MetricEvidence, PassageEvidence

SYSTEM = """\
あなたは与信審査の補助をするアシスタントです。融資の可否・金利・限度額・担保の要否について、
結論も推奨も述べません。与えられた「証拠」だけから、事実を簡潔に書きます。

守ること:
1. 1つの主張は1〜2文で、証拠に書かれた事実か、証拠から直接言える関係だけを書く。
2. 各主張の evidence_ids に、根拠にした証拠の ID（例: E3）を1つ以上入れる。提示された ID のみ使う。
3. 数値は、証拠に書かれた表記（例: 76.4%、139,657百万円）をそのまま使う。計算しない。丸めない。
4. 証拠にないことは書かない。理由や原因を推測しない。
5. 「融資可能」「懸念なし」「推奨」など、可否や推奨を示す語を使わない。
6. 出力は JSON だけ。説明やコードフェンスを付けない。
"""

_QUOTE_LIMIT = 500


def render_evidence(evidence: Sequence[MetricEvidence | PassageEvidence]) -> str:
    """証拠を、ID つきの一覧にする。"""
    lines: list[str] = []
    for item in evidence:
        if isinstance(item, MetricEvidence):
            level = f" | 水準: {item.level}" if item.level else ""
            lines.append(f"[{item.id}] 数値 | {item.label} | {item.display}{level} | {item.basis}")
        else:
            pages = sorted({s.page for s in item.spans})
            where = f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"
            quote = item.text.replace("\n", " ")
            if len(quote) > _QUOTE_LIMIT:
                quote = quote[:_QUOTE_LIMIT] + "…"
            heading = " > ".join(item.heading_path)
            lines.append(f"[{item.id}] 本文 | {where} | {heading} | {quote}")
    return "\n".join(lines) if lines else "（証拠なし）"


def claims_task(task: str, evidence_text: str, max_claims: int) -> str:
    return (
        f"【依頼】\n{task}\n主張は最大{max_claims}件。\n\n"
        f"【証拠】\n{evidence_text}\n\n"
        '【出力の形】\n{"claims": [{"text": "主張", "evidence_ids": ["E1"]}]}'
    )


FINANCIAL_TASK = (
    "財務の所見を書いてください。直近期と前期を比べた変化（増減の方向）と、規程の水準"
    "（証拠の「水準」。標準・留意・要精査）を中心に。変化の要因は証拠にないので書かないでください。"
    "算定不能の指標は、算定不能であることと、その理由だけを書いてください。"
)
RISK_TASK = (
    "事業等のリスクなどの本文から、返済能力に影響しうる事項を挙げ、本文に書かれている範囲で、"
    "何が・どう影響しうるかを書いてください。取引先への売上依存については、割合が本文にあれば"
    "その割合を、無ければ書かないでください。"
)
OVERVIEW_TASK = (
    "企業概要を書いてください。事業の内容、従業員数、株主構成など、本文から分かることだけを。"
)

DRAFT_TASK = (
    "次の「これまでの主張」をもとに、与信判断上の論点と、確認が必要な事項を書いてください。\n"
    "- positives（肯定的な要素）と negatives（否定的な要素）を各2〜4件。それぞれ、これまでの主張と"
    "同じ証拠 ID を使う。結論や推奨は書かない。\n"
    "- open_items（確認が必要な事項）は、資料から確認できなかったこと・追加で確かめるべきことを"
    "1〜4件。証拠がなくてもよい（evidence_ids は空でよい）。推測で補わない。"
)


def draft_prompt(prior_claims: str, evidence_text: str) -> str:
    return (
        f"【依頼】\n{DRAFT_TASK}\n\n【これまでの主張】\n{prior_claims}\n\n【証拠】\n{evidence_text}\n\n"
        '【出力の形】\n{"positives": [{"text": "…", "evidence_ids": ["E1"]}], '
        '"negatives": [{"text": "…", "evidence_ids": ["E2"]}], '
        '"open_items": [{"text": "…", "evidence_ids": []}]}'
    )


def memo_prompt(evidence_text: str) -> str:
    """単一エージェントのベースライン: 全部の節を 1 回で書かせる。"""
    return (
        "【依頼】\n与信メモの下書きの、次の6つの節の主張を書いてください。\n"
        "- overview（企業概要）、financial_findings（財務の所見）、business_risks（事業リスク）、"
        "positives（肯定的な要素）、negatives（否定的な要素）、open_items（確認が必要な事項）。\n"
        f"- {FINANCIAL_TASK}\n- {RISK_TASK}\n- {OVERVIEW_TASK}\n"
        "- open_items は、確認できなかったこと・追加で確かめるべきこと。証拠はなくてもよい。\n"
        "- 各節は最大5件。\n\n"
        f"【証拠】\n{evidence_text}\n\n"
        '【出力の形】\n{"overview": [], "financial_findings": [], "business_risks": [], '
        '"positives": [], "negatives": [], "open_items": []}\n'
        '各要素は {"text": "主張", "evidence_ids": ["E1"]}。'
    )


PLAN_PROMPT = """\
【依頼】
有価証券報告書から調べる問いを決めてください。問いは、有価証券報告書の本文を検索するための
短い日本語の語句（例: 原材料価格の高騰の影響、為替変動のリスク）です。
- overview_queries: 企業概要のための問い。最大2件（事業の内容・従業員・株主の問いは自動で加える）。
- risk_queries: この会社の事業リスクのための問い。業種と財務指標の特徴に合わせる。最大4件
  （返済能力・取引先依存の問いは自動で加える）。

【会社】{company}（{industry}）

【財務指標の概要】
{summary}

【出力の形】
{{"overview_queries": ["…"], "risk_queries": ["…"]}}
"""
