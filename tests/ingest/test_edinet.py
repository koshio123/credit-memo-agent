import logging
from pathlib import Path

import httpx
import pytest

from evals.companies import load_companies
from ingest.edinet import EdinetClient, EdinetError, Kind, fetch_all

CANARY = "TESTKEY-must-never-appear-anywhere"
PDF = b"%PDF-1.7\n...body..."
ZIP = b"PK\x03\x04...body..."


class Recorder:
    """MockTransport のハンドラ。呼ばれたリクエストを記録し、種別ごとに応答を返す。"""

    def __init__(self, responses: dict[str, httpx.Response] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.responses = responses or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        kind = request.url.params["type"]
        default = {"1": ZIP, "2": PDF, "5": ZIP}[kind]
        return self.responses.get(kind, httpx.Response(200, content=default))


def _client(handler: Recorder, tmp_path: Path, sleeps: list[float] | None = None) -> EdinetClient:
    return EdinetClient(
        api_key=CANARY,
        data_dir=tmp_path,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


def test_PDFを保存して_再実行では取得しない(tmp_path: Path) -> None:
    rec = Recorder()
    client = _client(rec, tmp_path)

    first = client.download("S100AAAA", Kind.PDF)
    second = client.download("S100AAAA", Kind.PDF)

    assert first.downloaded is True
    assert second.downloaded is False
    assert first.path == tmp_path / "S100AAAA" / "S100AAAA.pdf"
    assert first.path.read_bytes() == PDF
    assert len(rec.requests) == 1


@pytest.mark.parametrize(
    ("kind", "type_param", "suffix"),
    [(Kind.XBRL, "1", ".xbrl.zip"), (Kind.PDF, "2", ".pdf"), (Kind.CSV, "5", ".csv.zip")],
)
def test_種別ごとにAPIのtypeと拡張子が決まる(
    tmp_path: Path, kind: Kind, type_param: str, suffix: str
) -> None:
    rec = Recorder()
    result = _client(rec, tmp_path).download("S100BBBB", kind)

    assert rec.requests[0].url.params["type"] == type_param
    assert result.path.name == f"S100BBBB{suffix}"


def test_エラーがJSONで返ってきたら保存しない(tmp_path: Path) -> None:
    # EDINET は失敗時に、HTTP 200 でエラーの JSON を返すことがある
    error_json = httpx.Response(200, json={"metadata": {"status": "404", "message": "Not Found"}})
    rec = Recorder({"2": error_json})

    with pytest.raises(EdinetError, match="PDF"):
        _client(rec, tmp_path).download("S100CCCC", Kind.PDF)

    assert not (tmp_path / "S100CCCC").exists() or not list((tmp_path / "S100CCCC").iterdir())


def test_HTTPエラーは_リトライしてから失敗する(tmp_path: Path) -> None:
    rec = Recorder({"2": httpx.Response(503)})
    sleeps: list[float] = []

    with pytest.raises(EdinetError, match="503"):
        _client(rec, tmp_path, sleeps).download("S100DDDD", Kind.PDF)

    assert len(rec.requests) == 3  # 初回 + リトライ2回
    assert len(sleeps) >= 2  # リトライの間に待つ


def test_一時的な失敗の後に成功すれば保存する(tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else httpx.Response(200, content=PDF)

    client = EdinetClient(
        api_key=CANARY,
        data_dir=tmp_path,
        http=httpx.Client(transport=httpx.MockTransport(flaky)),
        sleep=lambda _: None,
    )

    assert client.download("S100EEEE", Kind.PDF).path.read_bytes() == PDF


def test_リクエスト間に待つ(tmp_path: Path) -> None:
    sleeps: list[float] = []
    client = _client(Recorder(), tmp_path, sleeps)

    client.download("S100FFFF", Kind.PDF)
    client.download("S100FFFF", Kind.XBRL)

    assert sleeps  # 続けて叩かない


def test_ログにもエラーにもキーが出ない(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    client = EdinetClient(
        api_key=CANARY,
        data_dir=tmp_path,
        http=httpx.Client(transport=httpx.MockTransport(boom)),
        sleep=lambda _: None,
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(EdinetError) as exc:
        client.download("S100GGGG", Kind.PDF)

    assert CANARY not in str(exc.value)
    assert CANARY not in caplog.text
    assert CANARY not in repr(exc.value.__cause__ or "")


def test_成功時のログにもキーが出ない(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        _client(Recorder(), tmp_path).download("S100HHHH", Kind.PDF)

    assert CANARY not in caplog.text


def test_一括取得は_会社の2期分を3種類ずつ取得する(tmp_path: Path) -> None:
    companies = load_companies()[:2]
    rec = Recorder()

    results = fetch_all(_client(rec, tmp_path), companies)

    assert len(results) == 2 * 2 * 3  # 2社 × 2期 × (PDF, XBRL, CSV)
    assert all(r.path.exists() for r in results)


def test_一括取得は_1件失敗しても続けて_失敗を報告する(tmp_path: Path) -> None:
    companies = load_companies()[:1]
    bad_doc = companies[0].filings.current.doc_id

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(bad_doc) and request.url.params["type"] == "2":
            return httpx.Response(200, json={"metadata": {"status": "404"}})
        default = {"1": ZIP, "2": PDF, "5": ZIP}[request.url.params["type"]]
        return httpx.Response(200, content=default)

    client = EdinetClient(
        api_key=CANARY,
        data_dir=tmp_path,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    results = fetch_all(client, companies)

    failed = [r for r in results if r.error]
    assert len(failed) == 1
    assert failed[0].doc_id == bad_doc
    assert CANARY not in (failed[0].error or "")
    assert sum(1 for r in results if not r.error) == 5
