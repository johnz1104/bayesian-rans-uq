#!/usr/bin/env bash
# Every tracked C++ source and header in the native components is formatted.
set -euo pipefail
cd "$(dirname "$0")/../.."
formatter=${CFDR_CLANG_FORMAT:-clang-format}
command -v "$formatter" >/dev/null || { echo "Required formatter missing: $formatter" >&2; exit 2; }
"$formatter" --version
files=()
while IFS= read -r -d '' file; do
    files+=("$file")
done < <(git ls-files -z -- core rans-uq extras tests/tools | grep -z -E '\.(cpp|hpp)$')
"$formatter" --dry-run --Werror --style=file "${files[@]}"
