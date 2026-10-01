# credit-memo-agent

EDINET の有価証券報告書から、出典つきの与信メモの草案を生成するマルチエージェント。生成した主張を検証エージェントが出典と照合し、財務比率はコードで再計算する。**開発中**（W0〜W3 が完了し、W4 の検証エージェントに着手する段階）。

## 免責（必ずお読みください）

- **内規は架空です。** `policies/credit_policy.md` は、実在の金融機関の内規ではなく、本リポジトリの検証用に作成した模擬文書です。数値基準も、実務の水準を示すものではありません。
- **実在企業の信用評価ではありません。** `evals/datasets/companies.json` には実在する上場企業 10 社を載せていますが、公開されている有価証券報告書をもとに、検証用に簡易計算した数値です。「標準」「留意」「要精査」といった水準は、上の**架空の基準**に当てはめた分類にすぎず、その企業の信用力や財務の健全性についての評価・助言ではありません。融資・投資の判断には使わないでください。
- **システムは可否を判断しません。** 与信メモは判断材料の整理と根拠の提示にとどまり、融資の可否・金利・限度額・担保の要否についての結論も推奨も出さない設計です（規程 第3条）。
- 生成物は、人によるレビューを経るまで、審査資料として使えません。

## 状況

| フェーズ | 内容 | 状態 |
| --- | --- | --- |
| W0 | 開発環境、LLM バックエンド（ローカル LLM と Claude）、PostgreSQL、架空の内規、対象企業 10 社の選定 | 完了 |
| W1 | 有報の取り込み（PDF・XBRL）と、財務数値の抽出精度の評価 | 完了 |
| W2 | 本文の検索（BM25 ＋ 埋め込みの融合）と、その評価（L2）、`edinet-mcp` | 完了（リランクは未実施。`edinet-mcp` の単体公開は W5 で判断） |
| W3 | エージェント（計画・調査・起草）と、主張ごとの出典・機械的な検査 | 完了（1 社で実機確認。意味の検証は W4） |
| W4〜W5 | 検証エージェント・差し戻し・引用ビューア、L3 評価、事前生成メモ | 計画は [PLAN.md](PLAN.md) |

- 計画: [PLAN.md](PLAN.md)
- 設計判断（何を選び、何を諦めたか）: [docs/decisions.md](docs/decisions.md)
- 開発環境の記録と再現手順: [docs/setup.md](docs/setup.md)
- 開発規約: [CLAUDE.md](CLAUDE.md)

## 開発

```bash
uv sync                                   # Python 3.14 と依存を用意
uv run pre-commit install
cp .env.example .env                      # 値を編集する（キーは .env にだけ書き、コミットしない）
docker compose up -d --wait               # PostgreSQL（pgvector）

uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

LLM は `llm/` の抽象を通して呼びます。ローカル LLM（Ollama）と、Claude Code のログイン経由（Claude Agent SDK）を切り替えられます。後者は、自分のサブスクリプションで動かす少量・手動の用途に限ります。詳細は [docs/setup.md](docs/setup.md) と [docs/decisions.md](docs/decisions.md) を参照してください。

## ライセンス

コードと文書は [MIT ライセンス](LICENSE) です。

ただし、次のものは MIT の対象ではなく、それぞれの提供元の規約に従います。

- EDINET が公開している書類やデータ（有価証券報告書の PDF・XBRL、`evals/datasets/companies.json` に含まれる書類 ID などの公開メタデータ）は、EDINET の利用規約に従います。書類の本体はこのリポジトリに含めていません。
- Claude Agent SDK など、依存するライブラリは、それぞれのライセンスに従います。
