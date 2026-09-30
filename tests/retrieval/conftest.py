import uuid
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql

from retrieval.settings import DatabaseSettings
from retrieval.store import ChunkStore


@pytest.fixture
def store() -> Iterator[ChunkStore]:
    """テストごとに専用のスキーマを作る。本物のデータ（public）には触れない。"""
    url = DatabaseSettings(_env_file=None).database_url  # type: ignore[call-arg]
    schema = f"test_{uuid.uuid4().hex[:12]}"
    try:
        admin = psycopg.connect(url, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL に接続できません（docker compose up -d --wait）")
    admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    s = ChunkStore.connect(url, schema=schema)
    try:
        yield s
    finally:
        s.close()
        admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        admin.close()
