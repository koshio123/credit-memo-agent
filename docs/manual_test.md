# 手動テストの手順（W4 までの実装）

自動テストでは確かめられないこと（実データでの動き、メモの内容が出典に支えられているか、異常時の挙動）を、人が確かめる手順。所要時間の目安は、準備と A〜D で約30分、メモの確認（E）で約1時間（主張の突き合わせが大半）。

**この手順書に書いた期待の数値・メッセージは、2026-10-04 時点の実装での例**（実データの取り込みや文言を変えると変わる）。食い違ったら、まず期待の側が古くなっていないかを疑う。

**Claude の利用枠を使わずに試す方法**: `data/memos/` には、能美防災（6744）で生成済みの結果（`6744_multi_agent_verify2_claude_code.json` など）がある。D の生成を飛ばして、これを使って E 以降を確かめられる（生成し直すと、同じ入力でもプロンプトが変わっていればキャッシュが効かず、利用枠を使う）。

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

Claude（自分の利用枠を使う。少量だけ）。**`--verify-rounds 2` を付けると Verifier で検証し、失敗した主張を最大2回まで書き直す**（付けなければ従来どおり、検証なし）:

```bash
LLM_BACKEND=claude_code uv run python -m scripts.generate_memo --sec-code 6744 --mode multi_agent --verify-rounds 2
```

期待:

| 構成 | 保存されるファイル（`data/memos/`） | LLM の呼び出し |
| --- | --- | --- |
| `--mode baseline` | `6744_baseline_<backend>.md` / `.json` | 1 回 |
| `--mode multi_agent` | `6744_multi_agent_<backend>.md` / `.json` | 5 回 |
| 上に `--verify-rounds N` | `…_verifyN_<backend>.md` / `.json`（N=2 なら `_verify2`。0 は書き直さず検証だけ） | 検証（と書き直し）が加わり、能美防災の例で 11 回（うちキャッシュ 5） |

ログの「LLM N 回（キャッシュ M）」「除外 N・警告 N」を控える。同じ入力でもう一度実行すると、**キャッシュから返って速く、「キャッシュ」の回数が増える**（利用枠を使わない）。1 回で渡せる会社は、Claude では 3 社まで。

## E. メモの確認

**盲検の順序（重要）**: E-5 で Verifier の判定と人の判定を比べるため、**E-3 を終えて自分の判定を書き残すまで、検証つきのメモの `.md` / `.html` / `.review.md` は開かない**（Verifier の判定や「検査の記録」が見えて、判定が引きずられる）。E-1・E-2 は、検証なしで生成したメモ（`6744_multi_agent_claude_code.md`。D で `--verify-rounds` なしで生成するか、既存のものを使う）で行う。検証つきのメモを開いてよいのは、E-3 の後。

### E-1. 構造（目で見る。5分）

`data/memos/6744_multi_agent_<backend>.md` を開き、次を確かめる。

- [ ] 節が 0〜6 と「出典一覧」「検査の記録」の順にある
- [ ] 主張の文末に出典の番号（[3] など）があり、出典一覧の番号と対応している
- [ ] 「規程の水準」の列が、標準・留意・要精査のいずれか（または「—」）になっている
- [ ] 融資の可否・推奨を述べる文がない（ヘッダーの免責の文を除く）
- [ ] 第0節が「確認できなかった」の書き方で、「問題はない」と断定していない
- [ ] 末尾の「検査の記録」に、検査の内容、Verifier の実施状況（検証なしなら「実施していない」）、隔離した本文、主張が1件も得られなかった節、生成に失敗した節、除外・警告の主張が出ている
- [ ] 主張の末尾に印が付く場合がある: 「⚠数値要確認」（数値が出典と合わない）、「❓検証者が判断できなかった」（Verifier が判断不能）

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

### E-3. 主張と出典の突き合わせ（約30〜40分。いちばん大事）

メモの主張が、出典の内容に**実際に支えられているか**は、コードでは検査できない（機械検査は、出典の有無・存在・数値の一致までで、意味は見ていない）。人が確かめる。Verifier（LLM）の判定と比べるため、**先に Verifier の判定を見ずに**判定する。

```bash
# 検証つきのメモ（Verifier の判定が入っている）の JSON から、判定を隠した確認用ファイルを作る
uv run python -m scripts.review_memo --blind data/memos/6744_multi_agent_verify2_claude_code.json
# → 6744_multi_agent_verify2_claude_code.blind.review.md（Verifier の判定なし）
# 検証なしのメモに --blind を付けると、「比べる相手がない」と警告が出る（検証つきのメモを使う）
```

`….blind.review.md` を開く。主張ごとに、出典の引用文（本文。ページごと）または算式（数値）が並び、最後に内規照合の各行（第8〜13条）も同じ形で並ぶ。⚠が付いた主張は、機械検査が数値の警告を出したもの。次の基準で、判定の欄に印を付ける。

| 判定 | 基準 |
| --- | --- |
| 支持する | 引用文・算式から、主張がそのまま言える |
| 一部だけ支持する | 主張の一部は言えるが、言い過ぎ・足りない点がある |
| 支持しない | 引用文・算式から言えない、または食い違う |
| 判断できない | 引用が長すぎる・表が崩れていて読めない |

- 能美防災の例では、主張 28 件＋内規照合 8 行（合計 36 項目。データで変わる）。全部が難しければ、**各節から2〜3件**でよい。ただし**選び方に偏りが出ないよう、各節の最初の数件ではなく、無作為に**選ぶ。
- 本文の出典は、PDF のページを開いて、引用文が**実際にそこにある**ことも確かめる。
- 数えた結果（支持する／一部／しない／判断できない）を「記録」に書く。**これが、Verifier の判定の正しさを測る最初の基準になる。**

### E-4. 注意して見る点（既知の弱点）

- **見出しの階層は当てにならないことがある**。例: セグメント情報の表（p.103）の見出しが「連結キャッシュ・フロー計算書 > 残存履行義務に配分した取引価格」と出る。チャンク分けが見出しをページをまたいで引き継ぐため。**出典のページと引用文は正しい**が、見出しは参考程度。
- 8B（ローカル）は、否定的な要素に「水準は標準」と書く、証拠にない「詳細な情報は提供されていない」を確認事項に挙げる、などの内容の誤りがある。構造の検査では捕まらない。
- 論点（第5節）が、財務の所見（2.2）の言い換えになりやすい。
- 対象の20社は健全な会社ばかり。第10・11・13条の「該当」側は、実データでは見られない。
- **Verifier は LLM の判定で、誤る。** 特に「支持」と言いすぎる方向の誤りは、機械検査では捕まらない。E-5 で人の判定と比べる。
- 書き直し（差し戻し）で、確認が必要な事項（出典なしの項目）が、事実の文に変わることがある（実機で1件）。
- ローカル 8B は、Verifier・書き直しには使わない前提（品質が足りない。W3 の確認）。

## E-5. Verifier の判定と、人の判定を比べる

E-3 で付けた判定（blind）を、Verifier の判定入りの確認用ファイルと突き合わせる。

```bash
uv run python -m scripts.review_memo data/memos/6744_multi_agent_verify2_claude_code.json
# → 6744_multi_agent_verify2_claude_code.review.md（各主張の下に「Verifier（LLM）の判定: …（理由）」が出る）
```

- [ ] 各主張について、自分の判定と Verifier の判定を並べ、**一致・食い違いを数える**（一致率の最初の標本）。判定の対応は、支持する＝supported、一部だけ＝partial、支持しない＝unsupported、判断できない＝cannot_judge
- [ ] 食い違いのうち、**Verifier が supported なのに自分は「支持しない」「一部だけ」**としたものを、主張の文と理由つきで記録する（言いすぎの見逃し。いちばん重要）
- [ ] 逆に、Verifier が partial / unsupported なのに自分は「支持する」としたもの（厳しすぎ）も記録する
- [ ] メモ本文の「検査の記録」と、`.json` の `memo.rounds`（ラウンドごとの判定の件数）が合っている

## E-6. 引用ビューア（ブラウザ）

```bash
uv run python -m scripts.view_memo data/memos/6744_multi_agent_verify2_claude_code.json \
    --pdf-base "file://$PWD/data/edinet/{doc_id}/{doc_id}.pdf"
open data/memos/6744_multi_agent_verify2_claude_code.html
```

- [ ] 主張の末尾の番号ボタンを押すと、引用文・ページ・算式の欄が開閉する（本文は書類ID・ページごと、数値は算式と PDF のページ）
- [ ] Verifier の判定が主張に色つきのタグで出る（supported / partial など）。理由も読める
- [ ] 「内規への照合結果」の表の「根拠」の番号も、同じように開く
- [ ] 末尾の「検査の記録」に、ラウンドごとの件数と、除外・警告・隔離した本文の件数が出る
- [ ] ダークモード（OS の設定）でも読める。ウィンドウを細く（スマホ幅）しても、横にはみ出さない
- [ ] 「PDF を開く」で該当ページが開く（ブラウザによっては `#page=` が効かず、先頭が開く。その場合は番号のページを手でめくる）
- [ ] HTML のソースに、書類の本体（引用外のページ全文）が入っていない。`--pdf-base` なしで作ると、PDF のリンクが出ない

## E-7. ガードレール（本文中の指示の検出）

有報の本文に指示が混ざる状況は、実データでは起きない（20社・6,914 チャンクで誤検出 0）。合成した文で確かめる。

```bash
uv run pytest tests/agents/test_guard.py -v
```

- [ ] 「以前の指示を無視して…」などの10文が検出され、ふつうの文（「Operating system: …」を含む）が検出されないことを、テスト名と内容で確認する
- [ ] 自分で別の言い回しを `scan_injection` に渡して、検出できないものを探す（規則は網羅的でない。見つけた言い回しは記録し、W5 の故障注入に使う）:

```bash
uv run python -c "from agents.guard import scan_injection as s; print(s('ここに試したい文'))"
```

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
| 会社を複数渡して、1社だけ対象外 `--sec-code 6744 0000`（ローカルでよい） | 6744 は生成され、0000 は失敗として報告される。**他の会社の処理は止まらない**。終了コードは 1 |
| 読めない JSON を渡す `uv run python -m scripts.review_memo missing.json <正しいJSON>` | missing.json は「読めませんでした」と報告され、正しい JSON は処理される。終了コードは 1。`view_memo` も同様 |
| 検証中にバックエンドが失敗した場合（`generate_memo --verify-rounds 2` の途中で Claude の利用枠が尽きる、など） | 書き上げたメモは捨てられず保存される。検査の記録の「生成に失敗した節」に `verification` が出て、「Verifier による検証は実施していない」と書かれる（再現しにくいので、自動テストで確認済み。無理に起こさなくてよい） |

## 記録

| 日付 | 確かめた人 | 項目 | 結果（OK / 問題あり） | 気づいたこと |
| --- | --- | --- | --- | --- |
| | | B 検索 | | |
| | | C MCP | | |
| | | D 生成 | | |
| | | E-1 構造 | | |
| | | E-2 数値 | | |
| | | E-3 突き合わせ | 支持 ／ 一部 ／ しない ／ 判断不能 = _ ／ _ ／ _ ／ _ | |
| | | E-5 Verifier と人の一致 | 一致 _ 件 ／ Verifier 甘すぎ _ 件 ／ 厳しすぎ _ 件 / 全 _ 件 | |
| | | E-6 ビューア | | |
| | | E-7 ガードレール | | |
| | | F 異常系 | | |
