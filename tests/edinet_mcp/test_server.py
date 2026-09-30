"""MCP サーバーのテスト。メモリ内でクライアントとつなぎ、ツールを実際に呼ぶ。"""

import json

import pytest
from mcp import Client

from edinet_mcp.server import build_server
from edinet_mcp.service import EdinetService

from .conftest import CUR_DOC

pytestmark = pytest.mark.anyio

TOOLS = {"list_companies", "search_filings", "get_financials", "get_ratios", "get_page"}


@pytest.fixture
def client(service: EdinetService) -> Client:
    return Client(build_server(service))


async def test_5つのツールを公開し_どれも説明がある(client: Client) -> None:
    async with client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == TOOLS
    assert all(t.description and len(t.description) > 20 for t in tools)


async def test_ツールの説明は_融合検索や出典の使い方を伝える(client: Client) -> None:
    async with client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert "XBRL" in (tools["get_financials"].description or "")
    assert "ページ" in (tools["search_filings"].description or "")
    assert "可否" in (tools["get_ratios"].description or "")  # 判断はしない、と明示


async def test_検索を呼ぶと_出典つきの本文が構造化されて返る(client: Client) -> None:
    async with client:
        result = await client.call_tool("search_filings", {"query": "原材料", "sec_code": "9999"})
    assert not result.is_error
    items = result.structured_content["result"]  # type: ignore[index]
    assert items[0]["doc_id"] == CUR_DOC
    assert (items[0]["page_start"], items[0]["page_end"]) == (2, 3)


async def test_財務数値の金額は_桁を落とさず文字列で返る(client: Client) -> None:
    async with client:
        result = await client.call_tool("get_financials", {"sec_code": "9999"})
    data = result.structured_content
    assert data["financials"]["total_assets"] == "1000"  # type: ignore[index]
    assert data["source"] == "XBRL"  # type: ignore[index]


async def test_財務比率を呼ぶ(client: Client) -> None:
    async with client:
        result = await client.call_tool("get_ratios", {"sec_code": "9999"})
    ratios = result.structured_content["ratios"]  # type: ignore[index]
    assert ratios["equity_ratio"]["name"]
    assert ratios["equity_ratio"]["value"] == "50.0"


async def test_ページを呼ぶ(client: Client) -> None:
    async with client:
        result = await client.call_tool("get_page", {"sec_code": "9999", "page": 3})
    assert result.structured_content["text"] == "従業員の状況"  # type: ignore[index]


async def test_失敗は_エラーとして_理由つきで返る(client: Client) -> None:
    async with client:
        result = await client.call_tool("get_page", {"sec_code": "9999", "page": 99})
    assert result.is_error
    text = json.dumps([c.model_dump() for c in result.content], ensure_ascii=False)
    assert "1 から 3" in text


async def test_知らない証券コードも_エラーで返る(client: Client) -> None:
    async with client:
        result = await client.call_tool("get_ratios", {"sec_code": "1234"})
    assert result.is_error


async def test_並行して呼んでも_結果が揃う(client: Client) -> None:
    import anyio

    results: list[bool] = []

    async def call() -> None:
        r = await client.call_tool("search_filings", {"query": "原材料", "sec_code": "9999"})
        results.append(not r.is_error)

    async with client, anyio.create_task_group() as tg:
        for _ in range(8):
            tg.start_soon(call)
    assert results == [True] * 8
