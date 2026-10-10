#!/usr/bin/env bash
# Verify relocatable installed native targets from a separate consumer build.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
build=${1:-"$root/build/dev"}
case "$build" in /*) ;; *) build="$root/$build";; esac
scratch=$(mktemp -d "${TMPDIR:-/tmp}/cfdr-install.XXXXXX")
cleanup() {
    status=$?
    if [[ $status -ne 0 ]]; then
        for log in "$scratch"/*.log; do [[ ! -f "$log" ]] || cat "$log"; done
    fi
    rm -rf "$scratch"
    exit "$status"
}
trap cleanup EXIT
cmake --install "$build" --prefix "$scratch/prefix" > "$scratch/install.log"
# Relocate the prefix before consumption, and copy only the consumer itself.
mv "$scratch/prefix" "$scratch/relocated"
cp -R "$root/tests/install-consumer" "$scratch/consumer"
cmake -S "$scratch/consumer" -B "$scratch/build" \
    -DCMAKE_PREFIX_PATH="$scratch/relocated" > "$scratch/configure.log"
cmake --build "$scratch/build" --parallel 2 > "$scratch/build.log" 2>&1
ctest --test-dir "$scratch/build" --output-on-failure
cp -R "$root/rans-uq/hills" "$scratch/hills"
cmake -S "$scratch/hills" -B "$scratch/hills-build" \
    -DCMAKE_PREFIX_PATH="$scratch/relocated" > "$scratch/hills-configure.log"
cmake --build "$scratch/hills-build" --parallel 2 > "$scratch/hills-build.log" 2>&1
ctest --test-dir "$scratch/hills-build" --output-on-failure
if [[ -n "${CFDR_PYTHON:-}" ]]; then
    py_version=$("$CFDR_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    site_packages="$scratch/relocated/lib/python$py_version/site-packages"
    if [[ ! -d "$site_packages/cfdr" ]]; then
        echo "Requested installed Python package is absent: $site_packages/cfdr" >&2
        exit 1
    fi
    if [[ -d "$site_packages/cfdr" ]]; then
        (cd "$scratch" && PYTHONPATH="$site_packages" \
            "$CFDR_PYTHON" -c 'import cfdr; assert len(cfdr._core.build_info()["source_digest"]) == 64; print(cfdr._core.__file__)')
    fi
fi
