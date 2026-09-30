#!/usr/bin/env bash
# PostToolUse hook: 編集された .py を ruff で自動修正・整形し、直せない違反はClaudeに返す。
set -u
f=$(jq -r '.tool_response.filePath // .tool_input.file_path // empty')
case "$f" in
  *.py) ;;
  *) exit 0 ;;
esac
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
uv run --quiet ruff check --fix --quiet "$f" >/dev/null 2>&1
uv run --quiet ruff format --quiet "$f"
# 自動修正できなかった違反が残っていれば、exit 2 でClaudeに伝える
uv run --quiet ruff check --quiet "$f" >&2 || exit 2
