"""ガードレール: 有報の本文に混入した「指示」を、LLM に渡す前に検出する。

有報の本文は LLM にとって「データ」で、指示ではない。PDF に仕込まれた指示
（プロンプトインジェクション）の典型的な言い回しを、規則で検出する。**規則は網羅的ではない**（言い換えは検出できない）ので、
出力側の検査（出典・結論の語・Verifier）も防御として残す。誤検出を避けるため、
「指示」という語だけでは検出せず、無視の命令・役割の変更・システムプロンプトの開示・出力の強制の形を対象にする。
"""

import re
import unicodedata

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous_ja",
        re.compile(r"(以前|これまで|上記|先ほど|前)の?(指示|命令|プロンプト)[^。]{0,12}無視"),
    ),
    (
        "ignore_previous_en",
        re.compile(r"ignore (all |any |the )?(previous|prior|above|earlier)", re.I),
    ),
    ("reveal_system_prompt", re.compile(r"システムプロンプト[^。]{0,10}(出力|表示|開示|教え)")),
    ("role_change_ja", re.compile(r"あなたは(今から|これから)")),
    ("role_change_en", re.compile(r"you are now\b", re.I)),
    (
        "force_output",
        re.compile(
            r"「[^」]{1,20}」と(出力|回答|記載|記述|書)(せよ|すること|してください|しなさい)"
        ),
    ),
    (
        "force_answer",
        re.compile(
            r"(承認|融資可能|問題なし|推奨)と(出力|回答|記載|記述)(せよ|すること|してください|しなさい)"
        ),
    ),
    ("role_marker", re.compile(r"</?\s*(system|assistant)\s*>|(^|\s)(assistant|system)\s*:", re.I)),
)


def scan_injection(text: str) -> list[str]:
    """本文が指示の形をしていれば、一致した規則の名前を返す（無ければ空）。

    Args:
        text: 有報の本文（チャンクなど）。

    Returns:
        一致した規則の名前の一覧。
    """
    normalized = unicodedata.normalize("NFKC", text)
    flat = re.sub(r"\s+", "", normalized)
    spaced = re.sub(r"\s+", " ", normalized)
    hits: list[str] = []
    for name, pattern in _RULES:
        # 改行・空白で崩された言い回しは空白を除いて、英語と役割の印は空白を残して照合する
        if pattern.search(flat) or pattern.search(spaced):
            hits.append(name)
    return hits
