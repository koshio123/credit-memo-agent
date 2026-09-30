# 設計判断ログ

何を選び、何を諦めたかを、その時点で書き溜める。新しいものを上に追記する。

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
