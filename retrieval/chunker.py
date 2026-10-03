"""PDF の本文を、見出しの階層とページ範囲つきのチャンクに分ける。

検索の単位であり、メモの出典（どのページのどの節か）の単位でもある。

- ページの体裁（先頭の3行と末尾のページ番号）は本文に含めない。
- 見出し（第一部 → 第2 → 1 → (1)）の階層を、ページをまたいで引き継ぐ。見出しの行は本文に入れず、
  階層としてチャンクに持たせる。チャンクは見出しをまたがない。
- 行の途中では切らない（表の行を壊さないため）。1行が長すぎるときだけ、句点、無ければ文字数で切る。
"""

import re
from dataclasses import dataclass

# 見出しの階層（浅い順）。EDINET の有価証券報告書の書式（NFKC で正規化した本文）に基づく
_HEADINGS: list[re.Pattern[str]] = [
    re.compile(r"^第[一二三四五六七八九十]+部\s*【.+】$"),  # 第一部 【企業情報】
    re.compile(r"^第\d+\s*【.+】$"),  # 第2 【事業の状況】
    re.compile(r"^\d+\s*【.+】$"),  # 1 【事業等のリスク】
    re.compile(
        r"^\(\d+\)\s*(?:【.+】|[^\d\s。][^。]{0,39})$"
    ),  # (1) 経営方針 / (1)【連結財務諸表】
]
# 表の行や本文の文に見える (1) を見出しにしないための条件: 数値が2つ以上並ぶ行は見出しにしない
_NUMBER_RUN = re.compile(r"[\d,]+\s+[\d,]+")
_DATE = re.compile(r"\d{4}年")  # 日付の並ぶ行（書類の一覧の表など）も見出しにしない
_PAGE_NUMBER = re.compile(r"^\d+/\d+$")
_COMPANY_LINE = re.compile(r"\(E\d{5}\)$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    heading_path: list[str]
    page_start: int  # 1 始まり
    page_end: int
    text: str


def _body_lines(page: str) -> list[str]:
    """ページから、EDINET が付ける先頭の3行と末尾のページ番号を除いた本文の行を返す。

    Args:
        page: 1ページ分の本文。

    Returns:
        前後の空白を取り、空行を除いた本文の行。
    """
    lines = [line.strip() for line in page.split("\n")]
    if lines and lines[0] == "EDINET提出書類":
        lines = lines[1:]
        if lines and _COMPANY_LINE.search(lines[0]):
            lines = lines[1:]
            if lines and lines[0] == "有価証券報告書":
                lines = lines[1:]
    while lines and not lines[-1]:
        lines.pop()
    if lines and _PAGE_NUMBER.match(lines[-1]):
        lines.pop()
    return [line for line in lines if line]


def _heading_level(line: str) -> int | None:
    """行が見出しなら、その階層を返す。

    Args:
        line: 本文の1行。

    Returns:
        見出しの階層（0 が最も浅い）。見出しでなければ None。
    """
    for level, pattern in enumerate(_HEADINGS):
        if pattern.match(line):
            if level == len(_HEADINGS) - 1 and (_NUMBER_RUN.search(line) or _DATE.search(line)):
                return None
            return level
    return None


def _split_long_line(line: str, max_chars: int) -> list[str]:
    """max_chars より長い1行を、句点で区切って詰める。句点が無ければ文字数で切る。

    Args:
        line: 長い1行。
        max_chars: 1片の最大文字数。

    Returns:
        max_chars 以下に分けた文字列のリスト。
    """
    sentences = [s for s in re.split(r"(?<=。)", line) if s]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        # 1文が長すぎるときは、文字数で切る
        parts = [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]
        for part in parts:
            if current and len(current) + len(part) > max_chars:
                pieces.append(current)
                current = ""
            current += part
    if current:
        pieces.append(current)
    return pieces


# e5-small の入力上限（512トークン）に収まる最大の大きさ。800文字では5〜8%が上限を超えた（実測）
DEFAULT_MAX_CHARS = 600


def chunk_pages(doc_id: str, pages: list[str], max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """ページごとの本文（1始まりのページ番号は並び順）を、チャンクに分ける。

    Args:
        doc_id: 書類ID。空にはできない。
        pages: ページごとの本文。
        max_chars: 1チャンクの最大文字数。

    Returns:
        文書内の連番で ID を振ったチャンクのリスト。

    Raises:
        ValueError: doc_id が空のとき。
    """
    if not doc_id:
        raise ValueError("doc_id が空です")

    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []  # (見出しの階層, 見出し)
    buffer: list[str] = []
    buffer_pages: list[int] = []

    def heading_path() -> list[str]:
        """いまの見出しの階層を、浅い順の見出しの文字列にして返す。

        Returns:
            見出しのリスト。
        """
        return [text for _, text in path]

    def flush() -> None:
        if not buffer:
            return
        chunks.append(
            Chunk(
                chunk_id="",  # 連番は最後に振る
                doc_id=doc_id,
                heading_path=heading_path(),
                page_start=buffer_pages[0],
                page_end=buffer_pages[-1],
                text="\n".join(buffer),
            )
        )
        buffer.clear()
        buffer_pages.clear()

    for page_number, page in enumerate(pages, start=1):
        for line in _body_lines(page):
            level = _heading_level(line)
            if level is not None:
                flush()
                path[:] = [(lv, t) for lv, t in path if lv < level] + [(level, line)]
                continue
            if len(line) > max_chars:
                flush()
                for piece in _split_long_line(line, max_chars):
                    buffer.append(piece)
                    buffer_pages.append(page_number)
                    flush()
                continue
            if buffer and len("\n".join([*buffer, line])) > max_chars:
                flush()
            buffer.append(line)
            buffer_pages.append(page_number)
    flush()

    return [
        Chunk(f"{doc_id}:{i}", c.doc_id, c.heading_path, c.page_start, c.page_end, c.text)
        for i, c in enumerate(chunks)
    ]
