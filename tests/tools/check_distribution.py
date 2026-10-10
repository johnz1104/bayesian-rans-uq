"""Build and verify a wheel in a clean Framework Python 3.11 environment.

No inherited PYTHONPATH or user site is used. Raw files stay outside the repo.
--wheelhouse supports fully offline build/runtime dependencies supplied by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    args = parser.parse_args()
    required = ("build", "scikit_build_core", "pybind11")
    missing = [
        name
        for name in required
        if (spec := importlib.util.find_spec(name)) is None or spec.origin is None
    ]
    if missing:
        raise SystemExit("Required wheel build dependencies missing: " + ", ".join(missing))
    if sys.version_info[:2] != (3, 11) or not sys.base_prefix.startswith(
        "/Library/Frameworks/Python.framework/"
    ):
        raise SystemExit("Use Framework Python 3.11 for this distribution acceptance check")
    for tool in ("cmake", "c++", "otool"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Required distribution tool missing: {tool}")
    root = Path(__file__).resolve().parents[2]
    if args.output:
        out = args.output.resolve()
        if out == root or root in out.parents:
            raise SystemExit("Wheel acceptance output must be outside the source tree")
        out.mkdir(parents=True, exist_ok=False)
    else:
        out = Path(tempfile.mkdtemp(prefix="cfdr-wheel-"))
    env = dict(
        os.environ,
        PYTHONNOUSERSITE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        CMAKE_BUILD_PARALLEL_LEVEL="2",
    )
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    commands: list[list[str]] = []
    receipt: dict[str, object] = {
        "schema": "cfdr-wheel-check-v1",
        "passed": False,
        "python": sys.executable,
        "python_version": sys.version,
        "build_tool_versions": {
            name: version(name) for name in ("build", "scikit-build-core", "pybind11")
        },
        "cmake": subprocess.check_output(["cmake", "--version"], text=True).strip(),
        "compiler": subprocess.check_output(["c++", "--version"], text=True).strip(),
        "commands": commands,
        "data": "synthetic tests",
        "seed": None,
        "provisional": "package acceptance, no research result",
    }

    def run(argv: list[str], name: str, cwd: Path = out) -> None:
        commands.append(argv)
        with (out / name).open("w") as stream:
            subprocess.run(
                argv, cwd=cwd, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT
            )

    try:
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        receipt["source_sha256"] = {
            name: sha(root / name)
            for name in tracked
            if name
            and (
                name.startswith(("core/", "rans-uq/", "extras/", "tests/tools/", "cmake/"))
                or name in ("pyproject.toml", "CMakeLists.txt")
            )
        }
        receipt["revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(out / "wheels"),
                str(root),
            ],
            "build.log",
        )
        wheels = list((out / "wheels").glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("expected exactly one wheel")
        wheel = wheels[0]
        receipt["wheel_sha256"] = sha(wheel)
        run([sys.executable, "-m", "venv", str(out / "venv")], "venv.log")
        python = str(out / "venv/bin/python")
        pip = [python, "-m", "pip", "install", "--no-cache-dir"]
        if args.wheelhouse:
            pip += ["--no-index", "--find-links", str(args.wheelhouse.resolve())]
        run([*pip, f"{wheel}[dev]"], "install.log")
        copied = out / "tests"
        copied.mkdir()
        for test in (root / "rans-uq" / "python" / "tests").glob("test_*.py"):
            shutil.copyfile(test, copied / test.name)
        verify = out / "verify_install.py"
        verify.write_text("""from importlib import metadata, resources
from pathlib import Path
import json
import sys
import cfdr
package = Path(cfdr.__file__).resolve()
core = Path(cfdr._core.__file__).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve())
assert core.is_relative_to(Path(sys.prefix).resolve())
assert resources.files(cfdr).joinpath("py.typed").is_file()
info = cfdr._core.build_info()
assert len(info["source_digest"]) == 64
assert info["revision"] != "unknown"
print(json.dumps({"package": str(package), "extension": str(core), "build": info,
                  "versions": {name: metadata.version(name) for name in
                               ("cfdr", "numpy", "pytest", "mypy", "ruff")}}))
""")
        run([python, str(verify)], "identity.json")
        identity = json.loads((out / "identity.json").read_text())
        receipt["installed_identity"] = identity
        # Independently reproduce CMake's sorted C++ source-byte identity.
        sources = sorted(
            path
            for directory in ("core", "rans-uq", "extras", "tests/tools")
            for pattern in ("**/*.cpp", "**/*.hpp")
            for path in (root / directory).glob(pattern)
        )
        digest = hashlib.sha256(
            "".join(f"{p.relative_to(root)}:{sha(p)}\n" for p in sources).encode()
        ).hexdigest()
        if identity["build"]["source_digest"] != digest:
            raise RuntimeError("wheel source identity differs from source bytes")
        run(["otool", "-L", identity["extension"]], "linkage.log")
        linkage = (out / "linkage.log").read_text()
        for line in linkage.splitlines()[1:]:
            dependency = line.strip().split(" (", 1)[0]
            if not dependency.startswith(("/usr/lib/", "/System/Library/")):
                raise RuntimeError(f"unexpected non-system extension linkage: {dependency}")
        run([python, "-m", "pytest", "-q", str(copied)], "pytest.log")
        # Resolve the installed package through this clean interpreter. A filesystem
        # target under site-packages makes mypy treat dependencies as user modules.
        run([python, "-m", "mypy", "--strict", "-p", "cfdr"], "mypy.log")
        receipt["passed"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["log_sha256"] = {p.name: sha(p) for p in out.glob("*.log")}
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        sys.stdout.write(f"Distribution receipt: {out / 'receipt.json'}\n")


if __name__ == "__main__":
    main()
