#!/usr/bin/env bash
# Default is strict native, Python, typing, lint and formatting verification.
# --native explicitly selects native build/tests/installed consumers only.
set -euo pipefail
cd "$(dirname "$0")/../.."
run_slow=0
native_only=0
distribution=0
preflight=0
for arg in "$@"; do
    case "$arg" in
        --all) run_slow=1 ;;
        --native) native_only=1 ;;
        --distribution) distribution=1 ;;
        --preflight) preflight=1 ;;
        *) echo "Unknown check option: $arg" >&2; exit 2 ;;
    esac
done
if [[ $native_only -eq 1 && $distribution -eq 1 ]]; then
    echo "--native and --distribution are mutually exclusive" >&2; exit 2
fi
for tool in cmake c++; do
    command -v "$tool" >/dev/null || { echo "Required tool missing: $tool" >&2; exit 2; }
done
python=${CFDR_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.11/bin/python3}
with_python=OFF
if [[ $native_only -eq 1 && $run_slow -eq 1 ]]; then
    # The golden comparator uses only the selected interpreter's standard library.
    [[ -x "$python" ]] || { echo "Required interpreter unavailable: $python" >&2; exit 2; }
    "$python" -c 'import sys; print("Golden-check interpreter:", sys.executable, sys.version)'
fi
if [[ $native_only -eq 0 ]]; then
    [[ -x "$python" ]] || { echo "Required interpreter unavailable: $python" >&2; exit 2; }
    "$python" - "$distribution" <<'PY'
import importlib.util
import sys
required = ['pybind11', 'numpy', 'pytest', 'mypy', 'ruff']
if sys.argv[1] == '1':
    required += ['build', 'scikit_build_core']
missing = [name for name in required
           if (spec := importlib.util.find_spec(name)) is None or spec.origin is None]
print('check.sh interpreter:', sys.executable, sys.version)
if missing:
    sys.exit('Required verification dependencies missing: ' + ', '.join(missing))
PY
    command -v "${CFDR_CLANG_FORMAT:-clang-format}" >/dev/null || {
        echo "Required C++ formatter unavailable: ${CFDR_CLANG_FORMAT:-clang-format}" >&2; exit 2;
    }
    with_python=ON
fi
if [[ $preflight -eq 1 ]]; then
    echo "Requested verification dependencies are available."
    exit 0
fi
cmake_args=(--preset dev -DCFDR_BUILD_PYTHON="$with_python")
if [[ "$with_python" == ON ]]; then cmake_args+=(-DPython_EXECUTABLE="$python"); fi
cmake "${cmake_args[@]}" >/dev/null
cmake --build --preset dev --parallel 2
if [[ $run_slow -eq 1 ]]; then
    ctest --preset dev --parallel 2
    "$python" tests/tools/check_goldens.py --recorder build/dev/record_goldens
else
    ctest --preset dev --parallel 2 --label-exclude slow
fi
if [[ "$with_python" == ON ]]; then
    CFDR_PYTHON="$python" ./tests/tools/check_install.sh build/dev
    PYTHONPATH="$PWD/build/dev/python" "$python" -m pytest -q rans-uq/python/tests tests/tools/tests
    "$python" -m mypy rans-uq/python/cfdr
    "$python" -m ruff check rans-uq/python tests/tools
    "$python" -m ruff format --check rans-uq/python tests/tools
    CFDR_PYTHON="$python" ./tests/tools/check_format.sh
    if [[ $distribution -eq 1 ]]; then
        "$python" tests/tools/check_distribution.py
    fi
    echo "Strict requested checks passed."
else
    env -u CFDR_PYTHON ./tests/tools/check_install.sh build/dev
    echo "Native build, tests and installed consumers passed (explicit --native scope)."
fi
