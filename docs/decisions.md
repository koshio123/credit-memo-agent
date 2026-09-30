# 設計判断ログ

何を選び、何を諦めたかを、その時点で書き溜める。新しいものを上に追記する。

## 2026-09-30 LLMバックエンドの抽象（`llm/`）

- **決定**: 呼び出しは `LLMBackend`（Protocol）に統一し、`complete(LLMRequest) -> LLMResponse` を非同期で提供する。モデルは呼び出し側が `fast / standard / strong` の3段階で指定し、実モデル名はバックエンドが決める。応答キャッシュは `CachedBackend` としてバックエンドを包む。
- **理由**: Claude Pro の利用枠に制約があるため、ローカルLLMとClaudeを差し替えられることが前提になる。段階で指定すれば、エージェントのコードを変えずに割当だけ変えられる。キャッシュのキーにバックエンド名とモデル名を含めるので、割当を変えたときに古い応答を使い回さない。
- **仕様**: 失敗した呼び出しはキャッシュしない。壊れたキャッシュは読み飛ばして上書きする。温度が0より大きい呼び出しも、最初に得た応答を再現する（再現性を優先）。バックエンド固有の失敗は `LLMBackendError` にそろえる。
- **未実装**: `claude_code`（Agent SDK）と `anthropic_api`。前者はProのログインで動くか、規約上問題ないかを確認してから実装する。選ぶと `NotImplementedError` になる。
- **未検証**: Ollama の実機との接続。Ollama は未導入で、現状のテストはHTTPをモックしている。Qwen3 は既定で思考過程（`<think>`）を出す可能性があるので、実機で確認して必要なら除去する。

## 2026-09-30 PostgreSQL（pgvector + pg_bigm）

- **決定**: `pgvector/pgvector:pg17` にpg_bigm（v1.2-20250903）をソースからビルドして足したイメージを、Docker Composeで動かす。日本語の全文検索は pg_bigm の2-gram（`gin_bigm_ops`）、ベクトル検索は pgvector。
- **理由**: PostgreSQL標準の全文検索（tsvector）は日本語の分かち書きができない。pg_bigmは形態素解析器が不要で、`LIKE` の部分一致がそのまま索引に乗る。RDSなどのマネージドでも使える。
- **検証**: 日本語3行のテーブルで、ベクトルの近傍検索、`likequery('原材料価格')` の部分一致、実行計画で `Bitmap Index Scan`（bigm索引）が使われることを確認した。
- **諦めたこと**: BM25。pg_bigmは2-gramの一致度で順位付けするため、BM25とは順位が異なる。W2でRecall@kを測り、足りなければ検索側でBM25相当のスコアを計算し直す。
- **注意**: ホスト側のポートは5433にした（手元のPostgreSQL 17が5432を使う可能性があるため）。拡張の作成は初回起動時だけ走るので、SQLを変えたら `docker compose down -v` でボリュームを消す。

## 2026-09-30 開発ツールと品質ゲート

- **決定**: uv / ruff / pyright(strict) / pytest / pre-commit を使い、同じ検査を Claude Code の hook・pre-commit・GitHub Actions の3箇所で回す。
- **理由**: Claude が書いたコードにも自分が書いたコードにも同じ基準をかけるため。規約を「お願い」ではなく強制にする。
- **諦めたこと**: mypy との併用。pyright の strict に一本化した（設定が少なく、速い）。
- **トレードオフ**: `agents/`, `llm/`, `api/` などの汎用的な名前をルート直下のパッケージにした（PLAN.md の構成どおり）。配布はしないアプリなので許容するが、外部ライブラリと名前が衝突したら `creditmemo/` 配下に移す。

## 2026-09-30 LLMの利用範囲

- **決定**: Claude Pro の範囲内で作る。Claude Code 経由（Agent SDK）とローカルLLM（Ollama）を切り替えられるようにする。Anthropic API は使わない。
- **理由**: 追加のAPI課金を避けるため。Pro には API 利用枠が含まれない。
- **未確認**: Agent SDK をサブスクリプション認証で使うことの規約上の扱い。W0 中に確認する。
