"""生成結果の保存: Markdown のメモと、検査・使用量を含む JSON。"""

import json
from pathlib import Path

import pytest

from agents.export import save_result
from agents.pipeline import run_baseline
from edinet_mcp.service import EdinetService
from llm.fake import ScriptedBackend

pytestmark = pytest.mark.anyio

EMPTY = json.dumps(
    {
        "overview": [{"text": "出典なし。", "evidence_ids": []}],
        "financial_findings": [],
        "business_risks": [],
        "positives": [],
        "negatives": [],
        "open_items": [],
    },
    ensure_ascii=False,
)


async def test_MarkdownとJSONを保存する(service: EdinetService, tmp_path: Path) -> None:
    result = await run_baseline(service, ScriptedBackend([EMPTY]), "9999")

    md_path, json_path = save_result(result, tmp_path, backend="scripted", model="fake")

    assert md_path.name == "9999_baseline_scripted.md"
    assert md_path.read_text(encoding="utf-8").startswith("# 与信メモ（草案）")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["mode"] == "baseline"
    assert data["usage"]["calls"] == 1
    assert data["backend"] == {"name": "scripted", "model": "fake"}
    assert data["inspection"]["rejected"] == 1  # 出典なしの主張
    assert data["evidence"]["E1"]["kind"] == "metric"
    assert data["memo"]["sec_code"] == "9999"


async def test_保存先のディレクトリを作る(service: EdinetService, tmp_path: Path) -> None:
    result = await run_baseline(service, ScriptedBackend([EMPTY]), "9999")
    md_path, _ = save_result(result, tmp_path / "a" / "b", backend="x", model="y")
    assert md_path.exists()
