#!/usr/bin/env bash
# Build the pinned environment for the harness:
#   .venv/              Python 3.13, requirements.lock.txt, microimpute editable
#   microimpute/        microimpute main at e4cbe02 (the experiment's code)
#   microimpute-pr219/  PR 219's head, 296e47d (harness/pr219.py imports it)
set -euo pipefail
cd "$(dirname "$0")/.."

MAIN=e4cbe025dcea8a8b2733d9562f2fa04e50719d5b
PR219=296e47dc1d49b8f8be940dd915a24efc2a7b091d

if [ ! -d microimpute/.git ]; then
  git clone --quiet https://github.com/PolicyEngine/microimpute.git microimpute
fi
git -C microimpute fetch --quiet origin "$MAIN" "$PR219"
git -C microimpute checkout --quiet --detach "$MAIN"
if [ ! -e microimpute-pr219 ]; then
  git -C microimpute worktree add --quiet --detach "$PWD/microimpute-pr219" "$PR219"
fi
test "$(git -C microimpute rev-parse HEAD)" = "$MAIN"
test "$(git -C microimpute-pr219 rev-parse HEAD)" = "$PR219"

uv venv --quiet --python 3.13 .venv
uv pip install --quiet --python .venv/bin/python -r requirements.lock.txt
uv pip install --quiet --python .venv/bin/python --no-deps -e ./microimpute
echo "setup done: .venv, microimpute@${MAIN:0:7}, microimpute-pr219@${PR219:0:7}"
