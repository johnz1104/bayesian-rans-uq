# Installing and using cfdr

The native C++17 libraries and optional Python package can be installed
independently. They have been tested on macOS (Apple silicon) with Apple Clang 14
and Python 3.11.
The package requires Python 3.11 or later, NumPy, CMake 3.21 or later and a C++17 compiler.

## Native consumers

Build into a separate install prefix:

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DCFDR_BUILD_PYTHON=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/prefix"
cmake --build build/native --parallel 2
cmake --install build/native
```

Installed targets carry C++17, thread linkage and include directories:

| Target | Library |
|---|---|
| `cfdr::core` | Mesh, fields, finite-volume and linear-algebra infrastructure. |
| `cfdr::incompressible` | SIMPLE, SST and flow observations/sensitivities. |
| `cfdr::dbns` | Density-based compressible solver. |
| `cfdr::compressible` | Experimental pressure-based low-Mach solver. |

The [complete installed consumer](../tests/install-consumer/CMakeLists.txt)
uses `find_package(cfdr 0.1 CONFIG REQUIRED)` and links the required targets.
Configure it with `-DCMAKE_PREFIX_PATH="$PWD/build/prefix"`.

Native reference-taking objects require their mesh to outlive them. Compute
wall distances before constructing SST-based solvers. Keep shared meshes
immutable and give concurrent evaluations separate solver/model instances.
One model's mutable cache and workspace are not safe for concurrent calls.
Check the explicit solve status before accepting predictions; finite capped
outputs remain diagnostic.

## Python

Select the interpreter explicitly and install from the checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install .
python -c 'import cfdr; print(cfdr.__file__)'
```

The build uses scikit-build-core and pybind11. An extension built for one
Python ABI cannot be reused by another interpreter. Missing native support
raises an import error; there is no silent pure-Python solver fallback.

`cfdr` supplies frozen configuration types and typed statistical utilities.
The low-level solver interface is `cfdr._core`. Field exports own their NumPy
arrays; deleting a solver or taking another snapshot does not invalidate old
exports. `FlowFields.snapshot()` includes interior and boundary arrays.
`ForwardModel.evaluate_with_diagnostics(..., snapshot=True).snapshot` also
includes corrected face fluxes and source identity. Neither is a versioned
restart-file format.

`ForwardBatch` copies coefficient rows, releases the GIL for synchronous native
work and returns ordered records with explicit statuses. Each worker starts
cold with private solver storage. Other solver/model methods retain their
individual GIL behavior. Use the convergence mask before accepting predictions.
A Python source-observer callback receives owned arrays and is retained only
for the duration of its solve call. Native observer views are transient and
must be copied if retained.

Record configuration, method choices and `cfdr._core.build_info()` with numerical
results. Its source digest identifies the C++ source/header snapshot recorded
by the build, including test sources that are not compiled into the libraries. Compiler flags
are reported separately; a revision alone does not identify dirty sources or
stale binaries.

## Verification and distribution checks

Install verification dependencies in the chosen interpreter and run:

```sh
python -m pip install '.[dev,distribution]'
CFDR_PYTHON="$(command -v python)" ./tests/tools/check.sh --all
```

The strict gate requires native/Python tests, mypy, Ruff and both language format
checks. Missing tools fail explicitly. `--all` adds slow native tests and the
six-case golden check. `--native` explicitly selects native build, tests and
installed consumers; `--preflight` only checks tool availability.

The wheel acceptance script currently requires Framework Python 3.11 on macOS.
It builds a wheel, installs it into a fresh environment, runs copied package
tests outside the checkout, and checks typing, installed origins, system-library
linkage and source identity:

```sh
python tests/tools/check_distribution.py --output /tmp/cfdr-wheel-check
```

The output directory must be new and outside the source tree. `--wheelhouse`
selects an offline dependency directory; build tools must already be installed
in the calling interpreter, and without Ninja an offline build needs
`CMAKE_GENERATOR="Unix Makefiles"`. A build-tree Python test pass does not
replace this installed-wheel check. Current limitations are listed in the
[README](../README.md).
