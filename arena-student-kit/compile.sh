#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT_DIR/out"

javac --release 17 \
  -cp "$ROOT_DIR/lib/arena-framework.jar" \
  -d "$ROOT_DIR/out" \
  "$ROOT_DIR/src/student/StudentBotImpl.java"

echo "Compiled StudentBotImpl.java"
