# 設計判断ログ

何を選び、何を諦めたかを、その時点で書き溜める。新しいものを上に追記する。

## 2026-09-30 開発ツールと品質ゲート

- **決定**: uv / ruff / pyright(strict) / pytest / pre-commit を使い、同じ検査を Claude Code の hook・pre-commit・GitHub Actions の3箇所で回す。
- **理由**: Claude が書いたコードにも自分が書いたコードにも同じ基準をかけるため。規約を「お願い」ではなく強制にする。
- **諦めたこと**: mypy との併用。pyright の strict に一本化した（設定が少なく、速い）。
- **トレードオフ**: `agents/`, `llm/`, `api/` などの汎用的な名前をルート直下のパッケージにした（PLAN.md の構成どおり）。配布はしないアプリなので許容するが、外部ライブラリと名前が衝突したら `creditmemo/` 配下に移す。

## 2026-09-30 LLMの利用範囲

- **決定**: Claude Pro の範囲内で作る。Claude Code 経由（Agent SDK）とローカルLLM（Ollama）を切り替えられるようにする。Anthropic API は使わない。
- **理由**: 追加のAPI課金を避けるため。Pro には API 利用枠が含まれない。
- **未確認**: Agent SDK をサブスクリプション認証で使うことの規約上の扱い。W0 中に確認する。
