#!/usr/bin/env bash
# Record golden reference numbers with the repro preset.
#
#   ./tests/tools/record_goldens.sh [out_dir] [case ...]
#
# out_dir defaults to tests/golden. Cases run as separate processes, at most
# five at a time. The code
# identity written into every file is the current HEAD plus a dirty flag for
# uncommitted changes to native sources or the build files.
set -euo pipefail
cd "$(dirname "$0")/../.."

out="${1:-tests/golden}"
shift || true
mkdir -p "$out"

cmake --preset repro > /dev/null
cmake --build --preset repro --target record_goldens > /dev/null

sha="$(git rev-parse HEAD)"
dirty=0
if [[ -n "$(git status --porcelain -- core rans-uq extras tests/tools \
    tests/CMakeLists.txt cmake CMakeLists.txt CMakePresets.json)" ]]; then
    dirty=1
fi

cases=("$@")
if [[ ${#cases[@]} -eq 0 ]]; then
    cases=(periodic_channel_flat couette_re171_ny96 couette_re171_ny384
           developing_channel_re180 periodic_channel_re180 bfs_lemoin_uniform_inlet hills_ercoftac_60x40
           cosine_bump_periodic dbns_shock_tubes dbns_mms sensitivity_couette_re171_ny96)
fi

printf '%s\n' "${cases[@]}" \
    | xargs -P 5 -I{} ./build/repro/record_goldens "$out" "$sha" "$dirty" {}
