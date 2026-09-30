import pytest


@pytest.fixture
def anyio_backend() -> str:
    # 非同期テストは asyncio だけで回す（trio は使わない）
    return "asyncio"
