"""複数の検索結果の順位を1つにまとめる（Reciprocal Rank Fusion）。

全文検索とベクトル検索のスコアは尺度が違うので、スコアではなく順位で融合する。
順位 r の項目に 1 / (k + r) を与え、順位表をまたいで足す。k が大きいほど、上位の差を小さく見る。
"""

from collections.abc import Sequence


def rrf(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """順位表（上位から並んだ ID のリスト）を融合し、(ID, スコア) をスコアの高い順に返す。

    - 同じ順位表に同じ ID が複数あれば、最初の出現だけ数える。
    - 同点は、先に現れた順。結果は毎回同じになる。

    Args:
        rankings: 順位表のリスト。
        k: 順位に足す定数。大きいほど上位の差を小さく見る。1以上。
        weights: 順位表ごとの重み。None なら全部 1.0。

    Returns:
        (ID, 融合スコア) をスコアの高い順に並べたリスト。

    Raises:
        ValueError: k が1未満、または重みの数が順位表の数と違うとき。
    """
    if k <= 0:
        raise ValueError("k は1以上にしてください")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("重みの数が、順位表の数と違います")

    scores: dict[str, float] = {}
    for n, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[n]
        seen: set[str] = set()
        rank = 0
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    # dict は挿入順を保つので、同点は先に現れた順になる（sorted は安定）
    return sorted(scores.items(), key=lambda kv: -kv[1])
