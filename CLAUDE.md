# credit-memo-agent 開発規約

EDINETの有価証券報告書から、出典付きの与信メモ草案を生成するマルチエージェント。計画は [PLAN.md](PLAN.md)、設計判断は [docs/decisions.md](docs/decisions.md) を参照する。

## コマンド

```bash
uv sync                          # 依存のインストール
docker compose up -d --wait      # PostgreSQL（pgvector + pg_bigm）を起動
uv run pytest                    # テスト（LLMを呼ぶものは含まない）
uv run ruff check . && uv run ruff format --check .
uv run pyright                   # 型チェック（strict）
uv run pre-commit run --all-files
```

## 規約

- **型**: すべての関数に型ヒントを付ける。pyright の strict を通す。`Any` は境界（外部APIの生のJSONなど）だけに限り、Pydantic モデルで型を付けてから内部に渡す。
- **LLM呼び出し**: 必ず `llm/` のバックエンド抽象を通す。エージェントやワーカーから SDK を直接呼ばない。応答はキャッシュされる前提で、テストでは実LLMを呼ばない（実際に呼ぶテストは `@pytest.mark.llm`）。
- **秘密情報**: APIキーなどは `.env` にだけ置く（gitignore 済み）。コード・テスト・ログ・コミットに含めない。新しい変数は `.env.example` にも追記する。
- **ログ**: `print` は使わず `logging` を使う（ruff の T20 で検出される）。
- **パス**: `os.path` ではなく `pathlib` を使う。
- **日本語**: コメント・docstring・README・コミットメッセージは日本語。識別子は英語。
- **数値**: 財務数値・比率は LLM に計算させない。コードで計算し、計算式と入力値を出力に残す。
- **引用**: 生成する主張には必ず出典スパン（doc_id, page, 文字オフセット, 引用文）を付ける。出典のない主張は Verifier が差し戻す。
- **判断しない**: システムは与信の可否を出さない。論点の整理と根拠の提示までに留める。

## テスト方針

- 正解が決まる部分（パーサー、財務比率、引用照合）は、実装より先にテストを書く。
- LLM が絡む部分はキャッシュ済みの応答を使った回帰テストにする。
- CI は LLM を呼ばない。実LLMでの評価は手元で実行し、結果を `evals/reports/` に残す。

## Git

- `main` へ直接コミットしない。作業単位ごとにブランチを切り（`feat/…`, `fix/…`, `chore/…`, `docs/…`）、区切りのよいところでコミットする。
- マージ前に `/code-review` を通す。
- 設計上の判断（何を選び、何を諦めたか）は `docs/decisions.md` に追記する。完成後にまとめて書かない。

## 自動化されているもの

- `.claude/settings.json` の hook が、`.py` の編集後に `ruff check --fix` と `ruff format` を自動で走らせる。直せない違反は編集直後にエラーとして返るので、その場で直す。
- pre-commit と GitHub Actions が、lint・型チェック・テストを同じ基準で実行する。
