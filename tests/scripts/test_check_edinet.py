import logging

import httpx
import pytest

from scripts import check_edinet

CANARY = "TESTKEY-must-never-appear-in-logs"


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_ログにもエラーにもキーが出ない(caplog: pytest.LogCaptureFixture) -> None:
    ok = httpx.MockTransport(
        lambda _: httpx.Response(
            200, json={"metadata": {"status": "200", "message": "OK", "resultset": {"count": 3}}}
        )
    )

    with caplog.at_level(logging.DEBUG):
        code = check_edinet.run(CANARY, "2026-09-29", _client(ok))

    assert code == 0
    assert CANARY not in caplog.text


def test_失敗時の出力にもキーが出ない(caplog: pytest.LogCaptureFixture) -> None:
    def echo_key(request: httpx.Request) -> httpx.Response:
        # サーバーがキーをそのまま本文に返してきても、出力には出さない
        return httpx.Response(401, text=f"invalid key: {request.url.params['Subscription-Key']}")

    with caplog.at_level(logging.DEBUG):
        code = check_edinet.run(CANARY, "2026-09-29", _client(httpx.MockTransport(echo_key)))

    assert code == 1
    assert CANARY not in caplog.text


def test_接続エラーでもキーが出ない(caplog: pytest.LogCaptureFixture) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    with caplog.at_level(logging.DEBUG):
        code = check_edinet.run(CANARY, "2026-09-29", _client(httpx.MockTransport(boom)))

    assert code == 1
    assert CANARY not in caplog.text


def test_JSONでない200応答は失敗として終える() -> None:
    html = httpx.MockTransport(lambda _: httpx.Response(200, text="<html>maintenance</html>"))

    assert check_edinet.run(CANARY, "2026-09-29", _client(html)) == 1
