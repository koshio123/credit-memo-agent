import pytest

from retrieval.fusion import rrf


def ids(result: list[tuple[str, float]]) -> list[str]:
    return [i for i, _ in result]


def test_1つの順位表なら_順序をそのまま保つ() -> None:
    assert ids(rrf([["a", "b", "c"]])) == ["a", "b", "c"]


def test_両方の順位表に出る項目は_片方だけの項目より上になる() -> None:
    # a は両方で2位。b は片方で1位。RRF は複数の順位表で上位に出るものを重く見る
    fused = rrf([["b", "a", "x"], ["y", "a", "z"]])
    assert ids(fused)[0] == "a"


def test_スコアは_順位の逆数の和() -> None:
    fused = dict(rrf([["a", "b"], ["b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_同点なら_先に出た順で並べ_結果は毎回同じ() -> None:
    a = rrf([["a", "b"], ["b", "a"]])
    b = rrf([["a", "b"], ["b", "a"]])
    assert a == b
    assert ids(a) == ["a", "b"]  # a と b は同点。最初の順位表で先に出た a が先


def test_同じ順位表に重複があれば_最初の出現だけ数える() -> None:
    fused = dict(rrf([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)  # 2位ではなく、重複を除いた2番目


def test_空の順位表は無視する() -> None:
    assert ids(rrf([[], ["a"], []])) == ["a"]
    assert rrf([]) == []


def test_重みで_片方の順位表を重く見る() -> None:
    fused = rrf([["a", "b"], ["b", "a"]], weights=[1.0, 3.0])
    assert ids(fused) == ["b", "a"]


def test_重みの数が順位表の数と違えばエラー() -> None:
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])


def test_kが0以下ならエラー() -> None:
    with pytest.raises(ValueError):
        rrf([["a"]], k=0)
