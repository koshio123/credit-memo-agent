"""文字2-gram の BM25。検索の語句側（レキシカル）の基準線。

日本語は単語の区切りがないので、形態素解析を使わず、連続する2文字を語として扱う。
pg_bigm の類似度（質問文全体を1つの文字列として比べる）と違い、2-gram ごとに
「少ない文書にしか出ない2文字ほど重く見る」重み（IDF）を付けるので、長い質問文でも語句の一致が効く。
"""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping

# 文字・数字の連なり。空白や句読点、記号では切れる（「収益、圧迫」の「益圧」は作らない）
_RUN = re.compile(r"[^\W_]+")


def bigrams(text: str) -> list[str]:
    """NFKC で正規化し、文字の連なりごとに連続する2文字を取り出す。"""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return [run[i : i + 2] for run in _RUN.findall(normalized) for i in range(len(run) - 1)]


class Bm25Index:
    def __init__(self, documents: Mapping[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 は正の値にする")
        if not 0 <= b <= 1:
            raise ValueError("b は 0 以上 1 以下にする")
        self._k1 = k1
        self._b = b
        self._postings: dict[str, dict[str, int]] = {}
        self._lengths: dict[str, int] = {}
        for doc_id, text in documents.items():
            grams = bigrams(text)
            self._lengths[doc_id] = len(grams)
            for gram, count in Counter(grams).items():
                self._postings.setdefault(gram, {})[doc_id] = count
        total = sum(self._lengths.values())
        self._avg_length = total / len(self._lengths) if self._lengths else 0.0

    def _idf(self, document_frequency: int) -> float:
        n = len(self._lengths)
        return math.log(1 + (n - document_frequency + 0.5) / (document_frequency + 0.5))

    def search(
        self, query: str, k: int, restrict: Iterable[str] | None = None
    ) -> list[tuple[str, float]]:
        """スコアの高い順に (文書ID, スコア) を返す。共通の2文字が無い文書は返さない。

        restrict を渡すと、その文書だけを対象にする。IDF などの統計は索引全体のまま。
        """
        allowed = set(restrict) if restrict is not None else None
        scores: dict[str, float] = {}
        for gram in set(bigrams(query)):
            postings = self._postings.get(gram)
            if not postings:
                continue
            idf = self._idf(len(postings))
            for doc_id, tf in postings.items():
                if allowed is not None and doc_id not in allowed:
                    continue
                norm = 1 - self._b + self._b * self._lengths[doc_id] / self._avg_length
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (self._k1 + 1) / (
                    tf + self._k1 * norm
                )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]
