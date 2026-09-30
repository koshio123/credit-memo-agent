import pytest

from retrieval.bm25 import Bm25Index, bigrams


def ids(result: list[tuple[str, float]]) -> list[str]:
    return [i for i, _ in result]


# ---- 文字2-gram ----


def test_連続する2文字を取り出す() -> None:
    assert bigrams("原材料価格") == ["原材", "材料", "料価", "価格"]


def test_句読点や空白をまたぐ2文字は作らない() -> None:
    # 「収益、圧迫」の「益圧」は文中の並びではない
    assert bigrams("収益、圧迫") == ["収益", "圧迫"]
    assert bigrams("A B") == []  # 1文字ずつの語は2-gramにならない


def test_全角半角をそろえてから取り出す() -> None:
    assert bigrams("ＲＯＩＣ経営") == bigrams("ROIC経営")


def test_1文字や空文字は空() -> None:
    assert bigrams("") == []
    assert bigrams("あ") == []


# ---- 索引と検索 ----

DOCS = {
    "risk": "原材料価格が高騰した場合、収益を圧迫する可能性があります。",
    "staff": "従業員の平均年齢は四十歳で、平均勤続年数は十五年です。",
    "div": "配当性向は三十パーセントを目安として、安定的な配当を継続します。",
}


def test_語句が一致する文書が上位になる() -> None:
    index = Bm25Index(DOCS)
    assert ids(index.search("原材料価格の高騰", k=3))[0] == "risk"
    assert ids(index.search("配当性向の目安", k=3))[0] == "div"


def test_共通の2文字が無い文書は返さない() -> None:
    index = Bm25Index(DOCS)
    assert index.search("ABCDEFG", k=5) == []
    assert index.search("", k=5) == []


def test_多くの文書に出る2文字より_少ない文書にしか出ない2文字を重く見る() -> None:
    # 「する」はどの文書にもある。「圧迫」は1つにしかない
    docs = {"a": "圧迫する", "b": "検討する", "c": "実施する", "d": "確認する"}
    result = Bm25Index(docs).search("圧迫する", k=4)
    assert ids(result)[0] == "a"
    assert result[0][1] > result[1][1]


def test_同じ語が何度も出る文書を_上限なく有利にしない() -> None:
    # BM25 は出現回数の効果が飽和する。極端に繰り返す文書が、内容の合う文書を上回らない
    docs = {
        "spam": "配当" * 200,
        "good": "配当性向は三十パーセントを目安として安定的な配当を継続します",
    }
    assert ids(Bm25Index(docs).search("配当性向の目安", k=2))[0] == "good"


def test_restrictで検索対象を絞れる_索引の統計は全体のまま() -> None:
    index = Bm25Index(DOCS)
    result = index.search("平均", k=3, restrict={"risk", "div"})
    assert set(ids(result)) <= {"risk", "div"}
    assert "staff" not in ids(result)


def test_kで件数を絞り_同点はIDの順で結果が毎回同じ() -> None:
    docs = {"b": "決算短信", "a": "決算短信", "c": "決算短信"}
    first = ids(Bm25Index(docs).search("決算短信", k=2))
    assert first == ids(Bm25Index(docs).search("決算短信", k=2)) == ["a", "b"]


def test_文書が0件でも作れる() -> None:
    assert Bm25Index({}).search("何か", k=3) == []


def test_パラメータは正の値() -> None:
    with pytest.raises(ValueError):
        Bm25Index(DOCS, k1=0)
    with pytest.raises(ValueError):
        Bm25Index(DOCS, b=1.5)
