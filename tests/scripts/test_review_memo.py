"""確認用の出力コマンド: 読めないファイルがあっても、残りは続ける。"""

import json
import sys
from pathlib import Path

import pytest

from agents.export import save_result
from agents.pipeline import run_baseline
from edinet_mcp.service import EdinetService
from llm.fake import ScriptedBackend
from scripts.review_memo import main

pytestmark = pytest.mark.anyio

EMPTY = json.dumps(
    {
        k: []
        for k in (
            "overview",
            "financial_findings",
            "business_risks",
            "positives",
            "negatives",
            "open_items",
        )
    }
)


async def test_読めないファイルがあっても_残りを処理し_終了コードは1(
    service: EdinetService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = await run_baseline(service, ScriptedBackend([EMPTY]), "9999")
    _, good = save_result(result, tmp_path, backend="scripted", model="fake")
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    missing = tmp_path / "missing.json"

    monkeypatch.setattr(sys, "argv", ["review_memo", str(missing), str(broken), str(good)])
    assert main() == 1

    assert good.with_suffix(".review.md").exists()  # 失敗の後ろのファイルも処理した
    assert "missing.json" in caplog.text and "broken.json" in caplog.text


async def test_すべて読めれば_終了コードは0(
    service: EdinetService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await run_baseline(service, ScriptedBackend([EMPTY]), "9999")
    _, good = save_result(result, tmp_path, backend="scripted", model="fake")
    monkeypatch.setattr(sys, "argv", ["review_memo", str(good)])
    assert main() == 0


async def test_形式の違うJSONは_どの項目が問題かを示す(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mode": "baseline"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["review_memo", str(bad)])
    assert main() == 1
    assert "memo" in caplog.text and "evidence" in caplog.text  # 足りない項目の場所


async def test_blindは_別のファイル名で書く(
    service: EdinetService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await run_baseline(service, ScriptedBackend([EMPTY]), "9999")
    _, good = save_result(result, tmp_path, backend="scripted", model="fake")
    monkeypatch.setattr(sys, "argv", ["review_memo", "--blind", str(good)])
    assert main() == 0
    assert good.with_suffix(".blind.review.md").exists()
    assert not good.with_suffix(".review.md").exists()


async def test_blindで書いたファイルには_Verifierの判定が入らない_スクリプトを通しても(
    service: EdinetService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agents.export import SavedResult, load_result
    from agents.memo import VerificationRecord

    reply = json.dumps(
        {
            **{
                k: []
                for k in ("overview", "business_risks", "positives", "negatives", "open_items")
            },
            "financial_findings": [{"text": "自己資本比率は50.0%である。", "evidence_ids": ["E1"]}],
        }
    )
    result = await run_baseline(service, ScriptedBackend([reply]), "9999")
    _, path = save_result(result, tmp_path, backend="scripted", model="fake")
    saved: SavedResult = load_result(path)
    saved.memo.verifications = [
        VerificationRecord(
            section="financial_findings",
            claim_text="自己資本比率は50.0%である。",
            verdict="partial",
            reason="言い過ぎ",
        )
    ]
    path.write_text(saved.model_dump_json(), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["review_memo", "--blind", str(path)])
    assert main() == 0
    blind = path.with_suffix(".blind.review.md").read_text(encoding="utf-8")
    assert "言い過ぎ" not in blind and "Verifier" not in blind

    monkeypatch.setattr(sys, "argv", ["review_memo", str(path)])
    assert main() == 0
    assert "言い過ぎ" in path.with_suffix(".review.md").read_text(encoding="utf-8")


async def test_検証なしのメモをblindにすると_比べる相手がないと警告する(
    service: EdinetService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = await run_baseline(service, ScriptedBackend([EMPTY]), "9999")
    _, path = save_result(result, tmp_path, backend="scripted", model="fake")
    monkeypatch.setattr(sys, "argv", ["review_memo", "--blind", str(path)])
    assert main() == 0
    assert "比べる相手がない" in caplog.text
