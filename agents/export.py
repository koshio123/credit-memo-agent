"""生成結果の保存: 読むための Markdown と、検査・使用量・証拠を含む JSON。"""

import json
from pathlib import Path

from agents.pipeline import MemoResult
from agents.render import render_memo


def save_result(result: MemoResult, out_dir: Path, backend: str, model: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.memo.sec_code}_{result.mode}_{backend}"
    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"
    md_path.write_text(render_memo(result), encoding="utf-8")
    payload = {
        "mode": result.mode,
        "backend": {"name": backend, "model": model},
        "usage": {
            "calls": result.calls,
            "cached_calls": result.cached_calls,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        },
        "inspection": {
            "rejected": len(result.memo.rejected),
            "warnings": len(result.memo.warnings),
        },
        "memo": result.memo.model_dump(mode="json"),
        "evidence": {k: v.model_dump(mode="json") for k, v in result.pool.items.items()},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path
