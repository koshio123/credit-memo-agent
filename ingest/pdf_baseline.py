"""PDF の基準線: pdfplumber で本文を取り出し、規則で連結財務諸表から規程の入力項目を読む。

モデルを使わない最も単純な方式。Docling などの構造化と比べるための物差し（L1 評価の基準線）。
実データ（対象 10 社の有価証券報告書）の書式に基づく。

- 連結貸借対照表・連結損益計算書・連結キャッシュ・フロー計算書の見出しで範囲を決める。
  同じ項目名（売上高、資産合計など）は注記や個別財務諸表にも出るので、範囲を限らないと誤る。
- 項目ごとに載っている書類（BS/PL/CF）が決まっている。減価償却費は損益計算書の販管費の内訳にも
  出るが、正解データは CF の値。
- 「前期 当期」の 2 列から、最後の数値を当期として読む。
- 注記の印（※1）は小さい文字で数値に癒着することがあるので、文字の大きさで取り除く。
"""

import re
import statistics
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

import pdfplumber
from pydantic import BaseModel, ValidationError

Section = Literal["BS", "PL", "CF"]

_UNITS = {"百万円": 1_000_000, "千円": 1_000, "億円": 100_000_000, "円": 1}
_UNIT_RE = re.compile(r"\(単位[:：]\s*(百万円|千円|億円|円)\)")
_NUMBER_RE = re.compile(r"^[△▲-]?\d[\d,]*(?:\.\d+)?$")
_DASHES = {"-", "－", "―", "—", "−"}  # 「該当なし」を表す記号（0として扱う）

# 見出しの行（先頭の「1」や「(1)」は許す）。本文中の言及と区別するため、行全体で照合する
_HEAD = r"^(?:\d+\s*|\(\d+\)\s*)?"
_HEADINGS: dict[str, re.Pattern[str]] = {
    "BS": re.compile(_HEAD + r"【連結貸借対照表】\s*$"),
    "PL": re.compile(_HEAD + r"【連結損益[^】]*】\s*$"),
    "CI": re.compile(_HEAD + r"【連結包括利益計算書】\s*$"),
    "SS": re.compile(_HEAD + r"【連結株主資本等変動計算書】\s*$"),
    "CF": re.compile(_HEAD + r"【連結キャッシュ・フロー計算書】\s*$"),
}
# キャッシュ・フロー計算書の後に続く注記の見出し。ここから先は範囲に含めない
_NOTES_RE = re.compile(
    r"【注記事項】|連結財務諸表作成のための基本となる重要な事項|【連結附属明細表】"
)
_CF_MAX_PAGES = 3


@dataclass(frozen=True)
class ParsedLine:
    label: str
    values: list[Decimal]  # 行の右端から読んだ数値（前期、当期の順）


@dataclass(frozen=True)
class ExtractedValue:
    value: Decimal | None  # 円。読めなければ None
    section: Section | None = None
    page: int | None = None  # 1 始まり
    line: str = ""
    reason: str | None = (
        None  # 読めなかった理由: section_not_found / unit_not_found / label_not_found
    )


def _to_number(token: str) -> Decimal | None:
    """数値の文字列を Decimal にする。

    Args:
        token: 空白で区切った1語。

    Returns:
        数値。「△」「▲」「-」は負、ダッシュだけの語は 0。数値でなければ None。
    """
    if token in _DASHES:
        return Decimal(0)
    if not _NUMBER_RE.match(token):
        return None
    negative = token[0] in "△▲-"
    digits = token.lstrip("△▲-").replace(",", "")
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None
    return -value if negative else value


def _split(line: str) -> tuple[str, list[Decimal]]:
    """行を、ラベルと、右端の数値（前期、当期の順）に分ける。数値が無ければ values は空。

    Args:
        line: 本文の1行。

    Returns:
        (ラベル, 数値のリスト)。注記の印（※）で始まる語は除く。
    """
    tokens = [t for t in line.split() if not t.startswith("※")]
    values: list[Decimal] = []
    while tokens:
        number = _to_number(tokens[-1])
        if number is None:
            break
        values.append(number)
        tokens.pop()
    values.reverse()
    return "".join(tokens), values


def parse_line(line: str) -> ParsedLine | None:
    """「ラベル ※注記 前期 当期」の行を、ラベルと右端の数値に分ける。数値が無ければ None。

    Args:
        line: 本文の1行。

    Returns:
        分けた結果。数値が無い、ラベルが無い、またはラベルが数値のときは None。
    """
    label, values = _split(line)
    if not values or not label or _to_number(label) is not None:
        return None
    return ParsedLine(label, values)


@dataclass(frozen=True)
class _Row:
    labels: list[str]  # 照合に使うラベルの候補（その行だけのもの、折り返しを結合したもの）
    values: list[Decimal]
    raw: str


def _rows(page: str) -> list[_Row]:
    """ページの本文から、数値のある行を取り出す。折り返されたラベルも読む。

    長いラベルは 2 行にまたがることがある。
    - 前半の行に数値が無く、後半の行の末尾に数値がある
    - ラベルだけの行の次が、数値だけの行
    どちらも、数値の無い直前の最大 2 行を結合したラベルも、照合の候補に加える。

    Args:
        page: ページの本文。

    Returns:
        数値のある行（ラベルの候補・数値・元の行）。
    """
    rows: list[_Row] = []
    pending: list[str] = []  # 数値の無い直前の行
    for raw in page.split("\n"):
        line = raw.strip()
        label, values = _split(line)
        if values and _to_number(label) is None:
            if label:
                labels = [label] + [
                    "".join(pending[-k:]) + label for k in (1, 2) if len(pending) >= k
                ]
            else:  # 数値だけの行。ラベルは直前の行
                labels = ["".join(pending[-k:]) for k in (1, 2) if len(pending) >= k]
            if labels:
                rows.append(_Row(labels, values, line))
            pending = []
        elif line:
            pending = [*pending, line][-2:]
    return rows


def unit_of(text: str) -> int | None:
    """本文に書かれた金額の単位を探す。

    Args:
        text: ページの本文。

    Returns:
        単位（円に直す倍率）。書かれていなければ None。
    """
    match = _UNIT_RE.search(text)
    return _UNITS[match.group(1)] if match else None


def _heading_pages(
    pages: list[str], key: str, start: int = 0, need: str | None = None
) -> int | None:
    """見出しの行があるページを探す。

    Args:
        pages: ページごとの本文。
        key: 探す見出しの種類（_HEADINGS のキー）。
        start: 探し始めるページ（0 始まり）。
        need: 指定すると、この文字列を含むページだけを調べる。

    Returns:
        最初に見つかったページ（0 始まり）。無ければ None。
    """
    pattern = _HEADINGS[key]
    for i in range(start, len(pages)):
        if need is not None and need not in pages[i]:
            continue
        if any(pattern.match(line.strip()) for line in pages[i].split("\n")):
            return i
    return None


def find_sections(pages: list[str]) -> dict[str, list[int]]:
    """連結 BS / PL / CF のページ（0 始まり）を、見出しの行で特定する。

    見つからない範囲は含めない。

    Args:
        pages: ページごとの本文。

    Returns:
        BS・PL・CF から、そのページ番号のリストへの対応。
    """
    sections: dict[str, list[int]] = {}
    bs = _heading_pages(pages, "BS", need="資産の部")
    if bs is None:
        return sections
    pl = _heading_pages(pages, "PL", start=bs + 1)
    if pl is None:
        return sections
    cf = _heading_pages(pages, "CF", start=pl + 1)

    # PL は、包括利益計算書・株主資本等変動計算書・CF のいずれかの見出しの手前まで
    ends = [
        i
        for key in ("CI", "SS", "CF")
        if (i := _heading_pages(pages, key, start=pl + 1)) is not None
    ]
    pl_end = min(ends) if ends else len(pages)
    sections["BS"] = list(range(bs, pl))
    sections["PL"] = list(range(pl, pl_end))
    if cf is not None:
        cf_pages: list[int] = []
        for i in range(cf, min(cf + _CF_MAX_PAGES, len(pages))):
            cf_pages.append(i)
            if _NOTES_RE.search(pages[i]):
                break  # このページで注記が始まる。見出し以降は抽出のときに切り捨てる
        sections["CF"] = cf_pages
    return sections


# 項目 -> (載っている書類, ラベルの候補。先に書いたものを優先)
_ITEMS: dict[str, tuple[Section, list[str]]] = {
    "total_assets": ("BS", [r"^資産合計$"]),
    "net_assets": ("BS", [r"^純資産合計$"]),
    "current_assets": ("BS", [r"^流動資産合計$"]),
    "current_liabilities": ("BS", [r"^流動負債合計$"]),
    "net_sales": ("PL", [r"^売上高$", r"^営業収益$", r"^売上高及び営業収益$"]),
    "operating_income": ("PL", [r"^営業(利益|損失)"]),
    # 受取利息と受取配当金が合算の行しかない会社は、合算を受取利息として読む（正解データと同じ扱い）
    "interest_income": ("PL", [r"^受取利息$", r"^受取利息及び配当金$"]),
    "dividend_income": ("PL", [r"^受取配当金$"]),
    "interest_expense": ("PL", [r"^支払利息$"]),
    "net_income_attributable_to_owners": ("PL", [r"^親会社株主に帰属する当期純(利益|損失)"]),
    "depreciation": ("CF", [r"^減価償却費"]),
}


# 符号を印字のとおりに読む項目のうち、「損失」だけのラベルは損失額を正の数で印字する
_SIGNED_BY_LABEL = {"operating_income", "net_income_attributable_to_owners"}


def _is_pure_loss_label(label: str) -> bool:
    """「営業損失」のように、利益を含まない損失のラベルか。「営業損失(△)」は印字どおりに読む。

    Args:
        label: 行のラベル。

    Returns:
        利益を含まない損失のラベルなら True。
    """
    return "損失" in label and "利益" not in label and "△" not in label


_PRIOR_COLUMN_HEADER = re.compile(r"^前連結会計年度\s*当連結会計年度", re.MULTILINE)


def _before_notes(page: str) -> str:
    """注記の見出しより前の本文。CF と注記が同じページにあっても、注記の同名の行を読まない。

    Args:
        page: ページの本文。

    Returns:
        注記の見出しより前の部分。見出しが無ければページ全体。
    """
    match = _NOTES_RE.search(page)
    return page[: match.start()] if match else page


def _page_unit(page: str, carried: int | None) -> int | None:
    """ページの単位を決める。ページに無ければ直前のページの単位を引き継ぐ。

    Args:
        page: ページの本文。
        carried: 直前のページの単位。

    Returns:
        単位（円に直す倍率）。どちらにも無ければ None。
    """
    return unit_of(page) or carried


def extract_items(pages: list[str]) -> dict[str, ExtractedValue]:
    """連結財務諸表から、規程の入力項目のうち財務諸表にそのまま載っているものを読む。

    Args:
        pages: ページごとの本文。

    Returns:
        項目名から抽出結果への対応。読めない項目は値が None で、理由が入る。
    """
    sections = find_sections(pages)
    result: dict[str, ExtractedValue] = {}
    for name, (section, patterns) in _ITEMS.items():
        page_indexes = sections.get(section)
        if not page_indexes:
            result[name] = ExtractedValue(None, reason="section_not_found")
            continue

        # ページごとの単位（無いページは直前のページの単位を引き継ぐ）
        units: dict[int, int | None] = {}
        carried: int | None = None
        for i in page_indexes:
            carried = _page_unit(pages[i], carried)
            units[i] = carried
        if all(u is None for u in units.values()):
            result[name] = ExtractedValue(None, reason="unit_not_found")
            continue

        # 「前連結会計年度 当連結会計年度」の見出しの行がある表は2列。数値が1つの行は、列の対応が
        # 分からないので読まない。見出しの無い1列の表（初回の有報など）では読む。継続ページには
        # 見出しが無いので、ページではなく節全体で判定する。文中の言及は根拠にしない
        bodies = {i: _before_notes(pages[i]) if section == "CF" else pages[i] for i in page_indexes}
        two_columns = any(_PRIOR_COLUMN_HEADER.search(bodies[i]) for i in page_indexes)

        found: ExtractedValue | None = None
        for pattern in patterns:
            regex = re.compile(pattern)
            for i in page_indexes:
                unit = units[i]
                if unit is None:
                    continue
                for row in _rows(bodies[i]):
                    if two_columns and len(row.values) < 2:
                        continue
                    label = next((lb for lb in row.labels if regex.search(lb)), None)
                    if label is None:
                        continue
                    value = row.values[-1] * unit
                    if name in _SIGNED_BY_LABEL and _is_pure_loss_label(label):
                        # 損失のラベルの値は常に負（正の数で印字されていれば反転、△付きはそのまま）
                        value = -abs(value)
                    found = ExtractedValue(value, section, i + 1, row.raw)
                    break
                if found:
                    break
            if found:
                break
        result[name] = found or ExtractedValue(None, section, reason="label_not_found")
    return result


def read_pages(path: Path) -> list[str]:
    """PDF から、ページごとの本文を取り出す（NFKC で全角の数字・記号を半角にそろえる）。

    注記の印（※1 など）は小さい文字で、数値に癒着して取り出されることがある
    （例: 「※1423,923」）。本文の中央値の 8 割より小さい文字を除いてから取り出す。

    Args:
        path: PDF のパス。

    Returns:
        ページごとの本文。
    """
    texts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            sizes = [c["size"] for c in page.chars]
            cutoff = statistics.median(sizes) * 0.8 if sizes else 0.0
            body = page.filter(lambda o, c=cutoff: o.get("object_type") != "char" or o["size"] >= c)
            texts.append(unicodedata.normalize("NFKC", body.extract_text() or ""))
    return texts


# 本文の取り出し方（read_pages）を変えたら上げる。古いキャッシュを使い続けないための版番号
TEXT_EXTRACTION_VERSION = 1


class _TextCache(BaseModel):
    version: int
    pages: list[str]


def _load_cache(cache: Path) -> list[str] | None:
    """本文のキャッシュを読む。

    Args:
        cache: キャッシュのファイル。

    Returns:
        ページごとの本文。無い、壊れている、版が違う、空のときは None。
    """
    try:
        data = _TextCache.model_validate_json(cache.read_text(encoding="utf-8"))
    except OSError, ValidationError:
        return None
    if data.version != TEXT_EXTRACTION_VERSION or not data.pages:
        return None
    return data.pages


def cached_pages(
    pdf_path: Path,
    cache_dir: Path,
    reader: Callable[[Path], list[str]] = read_pages,
) -> list[str]:
    """本文を、版番号つきでキャッシュして返す。版が違う・壊れている・空のキャッシュは使わない。

    Args:
        pdf_path: PDF のパス。
        cache_dir: キャッシュの保存先。
        reader: PDF から本文を取り出す関数。テストで差し替える。

    Returns:
        ページごとの本文。
    """
    cache = cache_dir / f"{pdf_path.stem}.json"
    if (pages := _load_cache(cache)) is not None:
        return pages
    pages = reader(pdf_path)
    if pages:  # ページが0件の結果（読み取りの失敗）は保存しない
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(
                _TextCache(version=TEXT_EXTRACTION_VERSION, pages=pages).model_dump_json(),
                encoding="utf-8",
            )
            tmp.replace(cache)
        finally:
            tmp.unlink(missing_ok=True)
    return pages
