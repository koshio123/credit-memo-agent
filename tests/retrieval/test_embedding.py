"""埋め込みの抽象のテスト。モデルは読み込まず、差し替えた関数で挙動を確かめる。"""

import math

import pytest

from retrieval.embedding import MODELS, HashEmbedder, SentenceTransformerEmbedder


class FakeModel:
    """SentenceTransformer の代わり。渡された文字列を記録し、決まった長さのベクトルを返す。"""

    def __init__(self, dim: int = 4, token_counts: list[int] | None = None) -> None:
        self.dim = dim
        self.seen: list[list[str]] = []
        self.token_counts = token_counts

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.seen.append(list(texts))
        self.kwargs = kwargs
        return [[float(len(t)), 1.0, 0.0, 0.0][: self.dim] for t in texts]

    def count_tokens(self, texts: list[str]) -> list[int]:
        return self.token_counts or [len(t) for t in texts]


def _embedder(key: str, model: FakeModel) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(MODELS[key], model=model)  # type: ignore[arg-type]


# ---- モデルの仕様（モデルカードで確認した値） ----


def test_e5の接頭辞と次元と最大長() -> None:
    spec = MODELS["e5-small"]
    assert spec.hf_id == "intfloat/multilingual-e5-small"
    assert (spec.query_prefix, spec.document_prefix) == ("query: ", "passage: ")
    assert (spec.dim, spec.max_tokens) == (384, 512)


def test_ruriの接頭辞と次元と最大長() -> None:
    spec = MODELS["ruri-v3-30m"]
    assert spec.hf_id == "cl-nagoya/ruri-v3-30m"
    assert (spec.query_prefix, spec.document_prefix) == ("検索クエリ: ", "検索文書: ")
    assert (spec.dim, spec.max_tokens) == (256, 8192)


# ---- 接頭辞: クエリと文書で違う。付け忘れると精度が落ちる ----


@pytest.mark.parametrize(
    ("key", "query_prefix", "doc_prefix"),
    [("e5-small", "query: ", "passage: "), ("ruri-v3-30m", "検索クエリ: ", "検索文書: ")],
)
def test_クエリと文書に別々の接頭辞を付けて埋め込む(
    key: str, query_prefix: str, doc_prefix: str
) -> None:
    model = FakeModel()
    embedder = _embedder(key, model)

    embedder.embed_queries(["自己資本比率とは"])
    embedder.embed_documents(["有価証券報告書の本文"])

    assert model.seen[0] == [f"{query_prefix}自己資本比率とは"]
    assert model.seen[1] == [f"{doc_prefix}有価証券報告書の本文"]


def test_埋め込みは正規化を依頼する() -> None:
    model = FakeModel()
    _embedder("e5-small", model).embed_documents(["x"])
    assert model.kwargs.get("normalize_embeddings") is True


def test_空の入力は空を返し_モデルを呼ばない() -> None:
    model = FakeModel()
    assert _embedder("e5-small", model).embed_documents([]) == []
    assert model.seen == []


def test_ベクトルは浮動小数のリストで返る() -> None:
    vectors = _embedder("ruri-v3-30m", FakeModel(dim=4)).embed_queries(["a", "bb"])
    assert len(vectors) == 2
    assert all(isinstance(x, float) for v in vectors for x in v)


# ---- 入力上限を超える文書の検出（黙って切り捨てられるのを防ぐ） ----


def test_最大長を超える文書を検出する() -> None:
    model = FakeModel(token_counts=[100, 513, 512])
    over = _embedder("e5-small", model).over_limit(["a", "b", "c"])
    # 接頭辞のトークン分も含めて数えるため、512ちょうどは超過に含めない/含めるの境界を明示する
    assert over == [False, True, False]


# ---- テスト用の決定的な埋め込み ----


def test_ハッシュ埋め込みは決定的で_正規化されている() -> None:
    e = HashEmbedder(dim=16)
    a1, a2 = e.embed_documents(["同じ文", "同じ文"])
    assert a1 == a2
    assert math.isclose(sum(x * x for x in a1), 1.0, rel_tol=1e-9)


def test_ハッシュ埋め込みは_似た文ほど近い() -> None:
    e = HashEmbedder(dim=64)
    (q,) = e.embed_queries(["原材料価格の高騰"])
    near, far = e.embed_documents(
        ["原材料価格の高騰が収益を圧迫する", "従業員の平均年齢は四十歳です"]
    )

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(q, near) > dot(q, far)
