import pytest

from edinet_mcp.service import EdinetService
from tests.edinet_fakes import FakeSearcher, build_service


@pytest.fixture
def anyio_backend() -> str:
    # 非同期テストは asyncio だけで回す（trio は使わない）
    return "asyncio"


@pytest.fixture
def searcher() -> FakeSearcher:
    return FakeSearcher()


@pytest.fixture
def service(searcher: FakeSearcher) -> EdinetService:
    return build_service(searcher)
