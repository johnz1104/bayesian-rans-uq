#!/usr/bin/env python3
"""Record and compare the six current-default golden cases without rebuilding.

The recorder must come from the current source, built with the dev or repro
preset. Both use -O2 without fast-math. Default candidates are temporary;
--output keeps evidence in a new directory and refuses to reuse existing files.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    "constrained_omega": (
        "couette_re171_ny96",
        "periodic_channel_re180",
        "bfs_lemoin_uniform_inlet",
        "cosine_bump_periodic",
    ),
    "dbns_stage_consistent": ("dbns_shock_tubes", "dbns_mms"),
}
# A change to any of these marks recorded references as coming from dirty source.
NATIVE_PATHS = (
    "core",
    "rans-uq",
    "extras",
    "tests/tools",
    "tests/CMakeLists.txt",
    "cmake",
    "CMakeLists.txt",
    "CMakePresets.json",
)


def check(recorder: Path, references: Path, output: Path) -> None:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--", *NATIVE_PATHS],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    for group, cases in GROUPS.items():
        candidate = output / group
        candidate.mkdir()
        # Run one process at a time so verification does not oversubscribe the host.
        for case in cases:
            subprocess.run(
                [str(recorder), str(candidate), revision, str(int(dirty)), case], check=True
            )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tests" / "tools" / "compare_goldens.py"),
                str(references / group),
                str(candidate),
                "--rtol",
                "1e-12",
                "--atol",
                "1e-14",
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", type=Path, default=ROOT / "build/dev/record_goldens")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "tests/golden")
    parser.add_argument("--output", type=Path, help="keep candidates in this new directory")
    args = parser.parse_args()
    recorder = args.recorder.resolve()
    references = args.reference_root.resolve()
    if not recorder.is_file() or not os.access(recorder, os.X_OK):
        parser.error(f"required executable recorder is unavailable: {recorder}")
    for group, cases in GROUPS.items():
        actual = {path.name for path in (references / group).glob("*.json") if path.is_file()}
        expected = {f"{case}.json" for case in cases}
        if actual != expected:
            parser.error(
                f"reference inventory differs for {group}: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )
    try:
        if args.output is not None:
            output = args.output.resolve()
            output.mkdir(parents=True, exist_ok=False)
            check(recorder, references, output)
        else:
            with tempfile.TemporaryDirectory(prefix="cfdr-goldens-") as temporary:
                check(recorder, references, Path(temporary))
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Golden check failed: {error}", file=sys.stderr)
        return 1
    print("All six current-default golden cases passed (rtol=1e-12, atol=1e-14).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
