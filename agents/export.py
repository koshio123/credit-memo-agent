"""生成結果の保存: 読むための Markdown と、検査・使用量・証拠を含む JSON。

JSON は `SavedResult` の形で保存し、読み込むときも同じ型で検証する（確認用の出力などが、
手で書き換えられた・版の違う JSON でも、理由の分かる形で失敗するように）。
"""

from pathlib import Path

from pydantic import BaseModel, TypeAdapter, model_validator

from agents.memo import MemoDraft
from agents.pipeline import MemoResult
from agents.render import render_memo
from agents.state import CLAIM_SECTIONS, Evidence


class BackendInfo(BaseModel):
    name: str
    model: str


class UsageInfo(BaseModel):
    calls: int
    cached_calls: int
    input_tokens: int
    output_tokens: int


class InspectionInfo(BaseModel):
    rejected: int
    warnings: int
    failures: int = 0  # 古い JSON（この項目がない版）も読めるようにする


class SavedResult(BaseModel):
    mode: str
    backend: BackendInfo
    usage: UsageInfo
    inspection: InspectionInfo
    memo: MemoDraft
    evidence: dict[str, Evidence]

    @model_validator(mode="after")
    def _evidence_exists(self) -> SavedResult:
        """主張が引用する証拠が、保存された証拠にあるか検証する。

        Returns:
            検証を通った自分自身。

        Raises:
            ValueError: 保存された証拠にない出典を主張が引用しているとき。
        """
        for key, _ in CLAIM_SECTIONS:
            for claim in getattr(self.memo, key):
                for evidence_id in claim.evidence_ids:
                    if evidence_id not in self.evidence:
                        raise ValueError(
                            f"主張が、保存された証拠にない出典 {evidence_id} を引用しています"
                            f"（{key}: {claim.text[:40]}）"
                        )
        return self


_EVIDENCE_MAP = TypeAdapter(dict[str, Evidence])


def from_result(result: MemoResult, backend: str, model: str) -> SavedResult:
    """生成結果を、保存用の型にする。

    Args:
        result: 生成結果。
        backend: LLM バックエンドの名前。
        model: 使ったモデル名。

    Returns:
        保存用の結果。
    """
    return SavedResult(
        mode=result.mode,
        backend=BackendInfo(name=backend, model=model),
        usage=UsageInfo(
            calls=result.calls,
            cached_calls=result.cached_calls,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        ),
        inspection=InspectionInfo(
            rejected=len(result.memo.rejected),
            warnings=len(result.memo.warnings),
            failures=len(result.memo.failures),
        ),
        memo=result.memo,
        evidence=_EVIDENCE_MAP.validate_python(dict(result.pool.items)),
    )


def save_result(result: MemoResult, out_dir: Path, backend: str, model: str) -> tuple[Path, Path]:
    """生成結果を、読むための Markdown と JSON に保存する。

    Args:
        result: 生成結果。
        out_dir: 保存先のディレクトリ。無ければ作る。
        backend: LLM バックエンドの名前。
        model: 使ったモデル名。

    Returns:
        (Markdown のパス, JSON のパス)。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.memo.sec_code}_{result.mode}_{backend}"
    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"
    md_path.write_text(render_memo(result), encoding="utf-8")
    saved = from_result(result, backend, model)
    json_path.write_text(saved.model_dump_json(indent=2), encoding="utf-8")
    return md_path, json_path


def load_result(path: Path) -> SavedResult:
    """保存した JSON を、同じ型で検証して読み込む。

    Args:
        path: JSON のパス。

    Returns:
        保存された結果。

    Raises:
        pydantic.ValidationError: 形が合わない、または引用する証拠が欠けているとき。
    """
    return SavedResult.model_validate_json(path.read_text(encoding="utf-8"))
