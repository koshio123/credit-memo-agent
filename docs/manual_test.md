# 手動テストの手順（W3 までの実装）

自動テストでは確かめられないこと（実データでの動き、メモの内容が出典に支えられているか、異常時の挙動）を、人が確かめる手順。所要時間の目安は、準備と A〜D で約30分、メモの突き合わせ（E）で約30分。

**この手順書に書いた期待の数値・メッセージは、2026-10-01 時点の実装での例**（実データの取り込みや文言を変えると変わる）。食い違ったら、まず期待の側が古くなっていないかを疑う。

確かめた結果は、最後の「記録」の表に書き、気づいたことは docs/decisions.md か issue に残す。

## 準備

```bash
docker compose up -d --wait                       # PostgreSQL（取り込み済みのデータが入っているはず）
docker compose exec -T db psql -U credit_memo -d credit_memo -c "select count(*) from chunks"
# → 6914（20社の当期の有報）。0 なら uv run python -m scripts.ingest_index で取り込む（約2分）
ls data/edinet | wc -l                            # → 40（20社×2期）
```

- ローカル LLM を使う場合: Ollama を起動する（`curl localhost:11434/api/tags` で `qwen3:8b` が見える）。
- Claude を使う場合: Claude Code に自分のアカウントでログインしておく。**`ANTHROPIC_API_KEY` は環境に置かない。**
- 対象にする会社の例: 6744 能美防災（PDF は `data/edinet/S100YHTN/S100YHTN.pdf`、137ページ）。

## A. 自動テストと品質ゲート

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run pyright
```

期待: すべて通る。DB が無いと db マークのテストはスキップされる。

## B. 検索

```bash
uv run python -m scripts.search_filings 6744 "原材料価格の高騰の影響" "従業員数と平均年齢" "継続企業の前提"
```

| 見るもの | 期待 |
| --- | --- |
| 「原材料価格の高騰の影響」 | 上位に、事業等のリスクの「原材料等の調達について」（p.23）が出る |
| 「従業員数と平均年齢」 | 「従業員の状況」（p.60 付近）が出る |
| 出力のページ | PDF（`open data/edinet/S100YHTN/S100YHTN.pdf`）の**そのページ**に、出力された文が実際にある |
| 「継続企業の前提」 | 監査報告書の定型文（p.132 付近）などが出る。**これは問題の記載ではない**（定型文） |

**確かめること**: 引用がページに実在するか（3件以上、PDF を開いて目で確認する）。

## C. MCP サーバー

```bash
uv run python -m scripts.check_edinet_mcp
```

期待: ツール5つ（list_companies, search_filings, get_financials, get_ratios, get_page）が見え、20社の一覧、検索の結果、自己資本比率 76.4%（標準）、1ページ目の本文が出る。最後に、範囲外のページが `is_error=True` になる。

任意（Claude の利用枠を少し使う）: `uv run python -m scripts.check_mcp_agent` で、Agent SDK からツールが呼ばれる。

## D. メモの生成

ローカル（配線の確認用。内容の品質は期待しない）:

```bash
LLM_BACKEND=local LOCAL_MODEL_FAST=qwen3:8b LOCAL_MODEL_STANDARD=qwen3:8b LOCAL_MODEL_STRONG=qwen3:8b \
  LOCAL_NUM_CTX=32768 uv run python -m scripts.generate_memo --sec-code 6744 --mode multi_agent
```

Claude（自分の利用枠を使う。少量だけ）:

```bash
LLM_BACKEND=claude_code uv run python -m scripts.generate_memo --sec-code 6744 --mode multi_agent
```

期待: `data/memos/6744_multi_agent_<backend>.md` と `.json` ができる。ログの「LLM 5 回」「除外 N・警告 N」を控える。同じ入力でもう一度実行すると、**キャッシュから返って速く、「キャッシュ 5」になる**（利用枠を使わない）。

## E. メモの確認

### E-1. 構造（目で見る。5分）

`data/memos/6744_multi_agent_<backend>.md` を開き、次を確かめる。

- [ ] 節が 0〜6 と「出典一覧」「検査の記録」の順にある
- [ ] 主張の文末に出典の番号（[3] など）があり、出典一覧の番号と対応している
- [ ] 「規程の水準」の列が、標準・留意・要精査のいずれか（または「—」）になっている
- [ ] 融資の可否・推奨を述べる文がない（ヘッダーの免責の文を除く）
- [ ] 第0節が「確認できなかった」の書き方で、「問題はない」と断定していない

### E-2. 数値をPDFと突き合わせる（10分）

財務の表（2.1 と 2.3）の数値は、コードが XBRL から算定したもの。PDF の同じ数字と合うかを確かめる。出典一覧の「PDF p.」のページを開く。

| 項目 | メモ（能美防災・直近期） | PDF の該当ページで確かめる |
| --- | --- | --- |
| 総資産 | 181,811 百万円 | p.63 の連結貸借対照表 |
| 純資産 | 138,986 百万円 | p.64 |
| 売上高 | 139,657 百万円 | p.65 の連結損益計算書 |
| 営業利益 | 18,349 百万円 | p.65 |
| 自己資本比率 | 76.4% | 138,986 ÷ 181,811 を自分で計算して一致するか |

- [ ] 5項目とも一致した（食い違ったら、どの項目か記録する）

### E-3. 主張と出典の突き合わせ（約30分。いちばん大事）

メモの主張が、出典の内容に**実際に支えられているか**は、コードでは検査できない（機械検査は、出典の有無・存在・数値の一致までで、意味は見ていない）。人が確かめる。

```bash
uv run python -m scripts.review_memo data/memos/6744_multi_agent_<backend>.json
```

できた `…review.md` を開く。主張ごとに、出典の引用文（本文）または算式（数値）が並ぶ。次の基準で、判定の欄に印を付ける。

| 判定 | 基準 |
| --- | --- |
| 支持する | 引用文・算式から、主張がそのまま言える |
| 一部だけ支持する | 主張の一部は言えるが、言い過ぎ・足りない点がある |
| 支持しない | 引用文・算式から言えない、または食い違う |
| 判断できない | 引用が長すぎる・表が崩れていて読めない |

- 全主張を見るのが難しければ、**各節から2〜3件**でよい。
- 本文の出典は、PDF のページを開いて、引用文が**実際にそこにある**ことも確かめる。
- 数えた結果（支持する／一部／しない／判断できない）を「記録」に書く。**これが、W4 の Verifier の評価の最初の基準になる。**

### E-4. 注意して見る点（既知の弱点）

- **見出しの階層は当てにならないことがある**。例: セグメント情報の表（p.103）の見出しが「連結キャッシュ・フロー計算書 > 残存履行義務に配分した取引価格」と出る。チャンク分けが見出しをページをまたいで引き継ぐため。**出典のページと引用文は正しい**が、見出しは参考程度。
- 8B（ローカル）は、否定的な要素に「水準は標準」と書く、証拠にない「詳細な情報は提供されていない」を確認事項に挙げる、などの内容の誤りがある。構造の検査では捕まらない。
- 論点（第5節）が、財務の所見（2.2）の言い換えになりやすい。
- 対象の20社は健全な会社ばかり。第10・11・13条の「該当」側は、実データでは見られない。

## F. 異常系

| やること | 期待 |
| --- | --- |
| `uv run python -m scripts.search_filings 0000 "x"` | 「失敗しました: 証券コード '0000' は対象外です。対象: 6744, 1969, …」と理由だけが出て終わる（スタックトレースは出ない） |
| DB を止める（`docker compose stop db`）→ 検索 | 「検索を使えません（PostgreSQL に接続できないか、モデルを読み込めません）。docker compose up -d --wait で DB を起動してください」。**DB が要らない財務数値・比率（get_ratios など）は使える** |
| 止めた DB を戻す（`docker compose start db`）→ 検索 | 検索が使える（MCP サーバーを動かしたままでも、次の呼び出しでつなぎ直す） |
| `ANTHROPIC_API_KEY=x LLM_BACKEND=claude_code uv run python -m scripts.generate_memo --sec-code 6744` | **LLM を呼ぶ前に**、「環境変数 ANTHROPIC_API_KEY が設定されています。… API 課金になるため、作成を中止しました」と出て終わる |
| `--sec-code` に4社以上を渡して `LLM_BACKEND=claude_code` | 「claude_code は自分の利用枠を使うため、一度に 3 社までです」で拒否される |
| Ollama に届かない状態で生成（`LLM_BACKEND=local OLLAMA_BASE_URL=http://localhost:9 LOCAL_NUM_CTX=32768 LLM_CACHE_ENABLED=false uv run python -m scripts.generate_memo --sec-code 6744 --mode baseline`） | 「失敗しました: Ollama に接続できません (http://localhost:9)…」。`LOCAL_NUM_CTX` を省くと、先に「入力が num_ctx=16384 に収まらない可能性があります」で止まる（入力が切り捨てられるのを防ぐ検査） |
| 未知の証券コードで生成 `--sec-code 0000` | 「失敗しました: 証券コード '0000' は対象外です…」 |

## 記録

| 日付 | 確かめた人 | 項目 | 結果（OK / 問題あり） | 気づいたこと |
| --- | --- | --- | --- | --- |
| | | B 検索 | | |
| | | C MCP | | |
| | | D 生成 | | |
| | | E-1 構造 | | |
| | | E-2 数値 | | |
| | | E-3 突き合わせ | 支持 ／ 一部 ／ しない ／ 判断不能 = _ ／ _ ／ _ ／ _ | |
| | | F 異常系 | | |
