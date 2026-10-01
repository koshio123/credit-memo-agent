# 開発環境のセットアップ記録

この環境に対して行った初期設定をすべて記録する。何を、どこに、なぜ入れたか、どう元に戻すかを書く。設計上の判断の理由は [decisions.md](decisions.md) にあり、ここでは重複させない。

記録日: 2026-09-30 / 対象: Apple M4 Pro・24GB・macOS

## 1. 全体の一覧

| # | 内容 | 場所 | 種類 |
| --- | --- | --- | --- |
| 1 | uv と Python 3.14.7 | `~/.local/`（uv 管理） | マシンの環境 |
| 2 | Homebrew の `ollama` 0.35.0 | `/opt/homebrew/` | マシンの環境 |
| 3 | Ollama のモデル（`qwen3:8b`、取得済み） | `~/.ollama/` | マシンの環境 |
| 4 | Docker イメージ・ボリューム | Docker Desktop の内部 | マシンの環境 |
| 5 | Python の依存パッケージ | リポジトリの `.venv/` | リポジトリ（gitignore） |
| 6 | pre-commit のフック | `.git/hooks/pre-commit` | リポジトリ（コミットされない） |
| 7 | 個人メモを追跡対象から外す設定 | `.git/info/exclude` | リポジトリ（コミットされない） |
| 8 | ツール設定・CI・規約・hook | リポジトリのファイル | リポジトリ（コミットされる） |
| 9 | 起動中のバックグラウンドプロセス | 現在のセッション | 一時的 |

## 2. マシンの環境に入れたもの（リポジトリの外）

### 2.1 uv と Python 3.14.7

- uv 0.12.9 は元から入っていた。Python 3.14.7 は uv が管理する版を使う（`.python-version` で 3.14 に固定）。
- pyenv の Python 3.12.8 も入っているが、このプロジェクトでは使わない。
- 経緯: `requires-python = ">=3.12"` だけだと uv が最新の 3.14 を選び、ruff・pyright の対象（3.12）とずれていた。3.14 にそろえた。→ decisions.md「Pythonを3.14に固定」

### 2.2 Ollama（ローカル LLM の実行環境）

```bash
brew install ollama          # 0.35.0
```

- **サービスとして常駐させていない。** `brew services start ollama` は実行していない（ログイン時に自動起動するため）。必要なときだけ手動で起動する。
- 起動コマンド（brew が推奨した環境変数付き）:

  ```bash
  OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
  ```

- Ollama の既定のコンテキスト長は 4096 トークン（サーバーのログで確認）。超えた入力は黙って切り捨てられるため、`llm/local.py` で `num_ctx` を明示している（既定 16384）。→ decisions.md「Ollama特有の落とし穴への対処」
- 元に戻す: `brew uninstall ollama` と `rm -rf ~/.ollama`（モデルも消える）。

### 2.3 Ollama のモデル

| モデル | 用途 | サイズ | 状態 |
| --- | --- | --- | --- |
| `qwen3:8b` | 開発中の反復、形式・配線の確認 | 5.2GB | **取得済み**（2026-09-30。回線が遅く、途中で何度も止まったため、監視スクリプトで取得をやり直しながら進めた） |
| `qwen3:14b` | 分析・起草 | 約 9GB | **未取得**（後回しにした） |
| VLM（Qwen2.5-VL 7B 級） | PDF の図表の読み取り | 未定 | **未取得**（基準線が財務数値で飽和したため保留。本文・表の読み取りの評価を作るときに判断する） |

注意:

- 取得が止まったときは `ollama pull qwen3:8b` をもう一度実行すれば、途中から再開する。
- 進捗は `du` では測れない（Ollama が先に領域を確保するため過大に見える）。`~/.ollama/models/blobs/*-partial-*` の JSON の `Completed` / `Size` を見る。
- 8B は金融の基礎知識で誤りが出る（自己資本比率の定義を取り違えた）。反復用に限り、内容の品質はClaudeで評価する。→ decisions.md
- **設定の既定値は `qwen3:14b` を指している。** 8B しかない間は、`.env` で次のように上書きしないと、存在しないモデルを呼んで失敗する。

  ```
  LOCAL_MODEL_STANDARD=qwen3:8b
  LOCAL_MODEL_STRONG=qwen3:8b
  ```

- Claude 側のモデル（`claude-haiku-4-5` / `claude-sonnet-5-5` / `claude-opus-5-5`）はインストール不要で、Claude Code のログイン経由で使う。3 つとも使えることを実機で確認済み。

### 2.4 Docker

`docker compose up -d --wait` で次のものができる。

| 種類 | 名前 | 内容 |
| --- | --- | --- |
| イメージ | `pgvector/pgvector:pg17`（公式） | 当初は pg_bigm をビルドして足した自前のイメージ（`credit-memo-agent-db:local`）だったが、pg_bigm を取り除いて公式に替えた（decisions.md） |
| ボリューム | `credit-memo-agent_pgdata` | PostgreSQL のデータ（拡張は vector のみ。取り込み済みのチャンクを含む） |
| コンテナ | `credit-memo-agent-db-1` | 起動時は `db` |

- ホスト側のポートは **5433**（手元の PostgreSQL 17 が 5432 を使う可能性があるため）。`127.0.0.1` にだけ公開している。
- 拡張（vector）は、`ChunkStore.connect` がスキーマの作成のときに作る（IF NOT EXISTS）。初期化用の SQL は置いていない。
- **pg_bigm 入りの古いボリュームから移行する場合**（取り込み済みのデータを残すなら）: 先に、古いイメージのまま `DROP INDEX chunks_text_bigm_idx; DROP EXTENSION pg_bigm;` を実行してから、イメージを替える。この索引が残ったまま公式のイメージに替えると、`chunks` への書き込みが失敗する。データが要らなければ `docker compose down -v` で足りる。古いイメージは `docker rmi credit-memo-agent-db:local` で消せる。
- 元に戻す: `docker compose down -v`（取り込み済みのデータも消える。`scripts.ingest_index` で作り直せる）。

### 2.5 埋め込みモデル（Hugging Face のキャッシュ）

`multilingual-e5-small`（`intfloat/`）と `ruri-v3-30m`（`cl-nagoya/`）が `~/.cache/huggingface/hub/` に入っている。`scripts/ingest_index.py` の初回実行で取得される。消しても再取得される（回線が遅いので時間がかかる）。取り込み後の DB は Docker のボリュームにあり、`docker compose down -v` で消える（その場合は `uv run python -m scripts.ingest_index` で作り直す。約2分）。

### 2.6 pre-commit のキャッシュ

`pre-commit-hooks`（外部リポジトリ）の環境が `~/.cache/pre-commit/` に作られた。消しても再作成される。

## 3. リポジトリの中にだけあるもの（コミットされない）

- **`.venv/`**: `uv sync` で作る仮想環境（Python 3.14.7）。
- **`.git/hooks/pre-commit`**: `uv run pre-commit install` で入る。クローンし直したら再実行が必要。
- **`.git/info/exclude`**: 公開しない個人メモ 1 ファイルを追跡対象から外す一行（ファイル名はそちらを参照）。`.gitignore` に書くと公開リポジトリにファイル名が残るので、こちらに書いた。クローンしても引き継がれない。
- **`data/`**（gitignore）: `data/edinet/`（取得した有価証券報告書 20 書類の PDF・XBRL・CSV、約 52MB。`uv run python -m scripts.fetch_filings`）と、`data/ground_truth/`（XBRL から作った正解データ。`uv run python -m scripts.build_ground_truth`）。EDINET 由来なので公開しない。
- **`.env`**: `.env.example` からコピーして作成済み。`LOCAL_MODEL_STANDARD/STRONG` は 8B に向け、`EDINET_API_KEY` を設定した。git の追跡対象外（`.gitignore`）。キーを表示させないため、編集後は読まない。
- **リモート**: `origin` は `https://github.com/koshio123/credit-memo-agent.git`（**公開**）。2026-09-30 に作成され、先頭の 2 コミット（履歴を作り直した後のもの）がプッシュされ、その時点の GitHub Actions は成功した。以降のコミットは、この記録の後にプッシュした。

## 4. リポジトリに入れたファイル（コミットされる）

### 4.1 ツールと品質ゲート

| ファイル | 役割 |
| --- | --- |
| `pyproject.toml` | 依存、ruff（lint・整形）、pyright（strict）、pytest の設定。`[tool.uv] package = false`（配布しないアプリ） |
| `.python-version` | Python 3.14 に固定 |
| `uv.lock` | 依存の固定 |
| `.pre-commit-config.yaml` | コミット時に ruff・pyright・pytest、秘密鍵・巨大ファイル検出などを実行 |
| `.github/workflows/ci.yml` | push / PR で ruff・pyright・pytest（`-m "not llm"`）を実行。**まだ一度も動かしていない** |
| `.claude/settings.json` | Claude Code の PostToolUse hook。`Write`/`Edit` で `.py` を編集した後に呼ぶ |
| `.claude/format-python.sh` | hook の中身。`ruff check --fix` → `ruff format` を自動実行し、直せない違反は編集直後にエラーとして返す |
| `CLAUDE.md` | 開発規約（型必須、LLM 呼び出しは `llm/` 経由、秘密情報は `.env` のみ、Claude 利用は少量・手動のみ、など） |
| `.gitignore` | `.env`、`data/`、`.cache/`、`.venv/` など |
| `.env.example` | 環境変数のひな形（§7） |

### 4.2 実装・データ・文書

| ファイル | 内容 |
| --- | --- |
| `docker-compose.yml` | PostgreSQL（公式の pgvector イメージ）の環境 |
| `llm/` | LLM 呼び出しの抽象（型、キャッシュ、Ollama、Claude Code、テスト用の偽バックエンド、設定、ファクトリ） |
| `retrieval/` | 本文のチャンク分け、埋め込み、BM25、RRF、PostgreSQL への保存と検索、取り込み |
| `agents/` | メモを書くエージェント（証拠の収集、内規照合、Planner・ワーカー・パイプライン、Markdown 出力、出典と構造の検査） |
| `scripts/generate_memo.py` | メモの生成。**実際に LLM を呼ぶ**（ローカルは `LOCAL_MODEL_*`・`LOCAL_NUM_CTX=32768`、Claude は `LLM_BACKEND=claude_code` で自分の利用枠を使う。一度に3社まで）。結果は `data/memos/` |
| `edinet_mcp/` | エージェント向けの MCP サーバー（検索・XBRL の財務数値・財務比率・ページ取得）。`uv run python -m edinet_mcp`（stdio。DB の起動と取り込みが要る） |
| `scripts/check_edinet_mcp.py`、`scripts/check_mcp_agent.py` | MCP サーバーの実機確認。後者は Claude の利用枠を少し使う |
| `evals/l2.py`、`scripts/ingest_index.py`、`scripts/eval_l2.py` | 検索の評価（L2）と、取り込み・評価のコマンド。`uv run python -m scripts.<名前>` で実行 |
| `tests/` | 上記のテスト。DB が要るものは `-m db`（CI では除く） |
| `scripts/check_ollama.py`、`scripts/check_claude_code.py` | 実機との接続確認。CI では実行しない。`uv run python -m scripts.<名前>` で実行 |
| `policies/credit_policy.md` | **架空の**融資内規（条文番号付き。財務指標の算式と境界値を附則で定義） |
| `templates/credit_memo.md` | 与信メモの雛形 |
| `PLAN.md`、`docs/decisions.md`、`docs/setup.md` | 計画、設計判断ログ、この記録 |

## 5. 依存パッケージ

実行時（`uv add` したもの。他は推移的依存）:

| パッケージ | 用途 |
| --- | --- |
| `pydantic` 2.13 / `pydantic-settings` 2.15 | 型付きのデータと、環境変数・`.env` からの設定 |
| `httpx` 0.28 | Ollama への HTTP 呼び出し |
| `claude-agent-sdk` 0.2.162 | Claude Code 経由の呼び出し。Anthropic の CLI を同梱している |
| `mcp` 2.2 | MCP サーバーとクライアント（`edinet_mcp/`）。2.x では `FastMCP` が `MCPServer` に改名されている |
| `psycopg` 3.3 / `pgvector` 0.5 | PostgreSQL への接続と、ベクトル型の読み書き |
| `sentence-transformers` 6.1（PyTorch 2.14） | 埋め込みモデル（e5-small、ruri-v3-30m）の実行。Apple Silicon では MPS を使う |
| `pdfplumber` 0.11 | PDF の本文の取り出し（基準線） |

開発時（`dev` グループ）: `ruff` 0.16.9、`pyright` 1.1.414、`pytest` 9.1.1、`pre-commit` 4.6.2。

## 6. 現在起動しているプロセス

この作業セッションで起動したもの。マシンを再起動すれば消える。

| プロセス | 目的 |
| --- | --- |
| `ollama serve` | Ollama のサーバー（`127.0.0.1:11434`）。手動で起動 |

止めるとき: `pkill -f "ollama pull"`、`pkill -f "ollama serve"`。

## 7. 環境変数（`.env.example` の要点）

| 変数 | 意味 |
| --- | --- |
| `LLM_BACKEND` | `local`（既定） / `claude_code` / `anthropic_api`（未実装） |
| `OLLAMA_BASE_URL`、`LOCAL_MODEL_FAST/STANDARD/STRONG` | Ollama の接続先と、段階ごとのモデル |
| `LOCAL_NUM_CTX`（既定 16384）、`LOCAL_THINK`（既定 false） | コンテキスト長と、思考モード |
| `CLAUDE_MODEL_FAST/STANDARD/STRONG`、`CLAUDE_THINK` | Claude 側のモデル（完全な ID で固定）と思考モード |
| `LLM_CACHE_ENABLED`、`LLM_CACHE_DIR` | 応答キャッシュ（既定は有効、`.cache/llm`） |
| `POSTGRES_*`、`DATABASE_URL` | PostgreSQL の接続情報（ポートは 5433） |
| `EDINET_API_KEY` | EDINET API v2 のキー（取得・設定済み。`.env` にだけ置く） |

**`ANTHROPIC_API_KEY` と `ANTHROPIC_AUTH_TOKEN` は環境に置かない。** あると Claude Code が API 課金で動くため、`claude_code` バックエンドが作成を拒否する。

## 8. 新しいマシンで再現する手順

```bash
git clone <このリポジトリ> && cd credit-memo-agent
uv sync                                  # Python 3.14 と依存を用意
uv run pre-commit install                # コミット時の検査を有効にする
cp .env.example .env                     # 必要な値を編集（EDINET_API_KEY など）

docker compose up -d --wait              # PostgreSQL（初回はイメージの取得）

brew install ollama                      # ローカル LLM を使う場合
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &
ollama pull qwen3:8b

uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

Claude Code 経由を使う場合は、Claude Code に自分のアカウントでログインしておく（API キーは不要）。

データの取得から検索の評価まで（書類は `data/` に置かれ、コミットされない。埋め込みモデルの初回取得に時間がかかる）:

```bash
uv run python -m scripts.fetch_filings                      # 開発用10社（EDINET_API_KEY が要る）
uv run python -m scripts.fetch_filings --dataset heldout    # 保留データ10社
uv run python -m scripts.build_ground_truth  # XBRL から L1 の正解データ（開発用）
uv run python -m scripts.eval_l1_pdfplumber  # L1 評価（PDF の抽出精度）。--dataset heldout で保留データ
uv run python -m scripts.ingest_index        # チャンク分け・埋め込み・DB への保存（約2分）
uv run python -m scripts.eval_l2             # L2 評価（検索）。結果は evals/reports/
uv run python -m scripts.check_edinet_mcp    # MCP サーバーを stdio で起動して確認
# メモの生成（LLM を呼ぶ。ローカルなら Ollama を起動し LOCAL_MODEL_* を設定）
uv run python -m scripts.generate_memo --sec-code 6744 --mode multi_agent
```

## 9. 未完了・未確認

| 項目 | 状態 |
| --- | --- |
| Ollama との実機確認 | **完了**（`think`・`done_reason`・入力の切り捨て。結果は decisions.md） |
| EDINET API キー | 取得・設定済み。確認スクリプトの初回実行でキーが画面に出たが、ユーザーの判断で再発行はしていない。原因は修正済み |
| 対象企業 10 社の選定 | **完了**（`evals/datasets/companies.json`。経緯と限界は decisions.md） |
| GitHub へのプッシュ・CI | **確認済み**（2026-10-01 に W3 までを push。CI は 51 秒で成功。torch / sentence-transformers を含む依存の導入と、`db` マークのテストの除外が問題なく動く） |
| エージェントがツールを使う方法 | **未決**（W3 の最初に決める。PLAN.md 9-4。案は、Worker が決定的なコードでサービス層を呼ぶ方式） |
| L2 設問の人による確認 | 未実施（質問は Claude が根拠から作った。docs/decisions.md） |
| `edinet-mcp` と Agent SDK の接続 | **完了**（`scripts/check_mcp_agent.py`。結果は decisions.md） |
| `anthropic_api` バックエンド | 未実装（選ぶと `NotImplementedError`） |
