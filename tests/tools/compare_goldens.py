#!/usr/bin/env python3
"""Compare two directories of golden JSON files number by number.

    python3 tests/tools/compare_goldens.py OLD_DIR NEW_DIR [--rtol 1e-12] [--atol 1e-14]

Every numeric leaf must be finite and satisfy |new - old| <= atol + rtol*|old|.
Null leaves are invalid outside the ignored metadata fields: the legacy writer
used null to mask nonfinite numerical results.
Timing and code-identity fields are skipped. A missing or empty reference
cannot certify a comparison. Standard library only.
"""

import argparse
import json
import math
import sys
from pathlib import Path

SKIP_KEYS = {"wall_seconds", "code", "build", "notes"}


def walk(old, new, path, out):
    if old is None or new is None:
        out.append((path, "null value outside ignored metadata", None, None))
    elif isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key in SKIP_KEYS:
                continue
            if key not in old or key not in new:
                out.append(
                    (f"{path}.{key}", "missing in " + ("new" if key in old else "old"), None, None)
                )
                continue
            walk(old[key], new[key], f"{path}.{key}", out)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append((path, f"length {len(old)} -> {len(new)}", None, None))
            return
        for i, (a, b) in enumerate(zip(old, new)):
            walk(a, b, f"{path}[{i}]", out)
    elif type(old) in (int, float) and type(new) in (int, float):
        out.append((path, "num", float(old), float(new)))
    elif type(old) is not type(new) or old != new:
        out.append((path, f"{old!r} -> {new!r}", None, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-14)
    args = ap.parse_args()

    for name in ("rtol", "atol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            ap.error(f"--{name} must be finite and nonnegative")

    n_bad = 0
    old_dir, new_dir = Path(args.old), Path(args.new)
    for directory in (old_dir, new_dir):
        if not directory.is_dir():
            ap.error(f"comparison directory does not exist: {directory}")
    old_files = sorted(old_dir.glob("*.json"))
    if not old_files:
        ap.error(f"reference directory contains no JSON files: {old_dir}")
    old_names = {file.name for file in old_files}
    for new_file in sorted(new_dir.glob("*.json")):
        if new_file.name not in old_names:
            print(f"{new_file.name}: missing reference in {old_dir}")
            n_bad += 1
    for old_file in old_files:
        new_file = new_dir / old_file.name
        if not new_file.is_file():
            print(f"{old_file.name}: missing in {new_dir}")
            n_bad += 1
            continue
        diffs = []
        walk(json.loads(old_file.read_text()), json.loads(new_file.read_text()), "", diffs)
        bad = []
        for path, kind, a, b in diffs:
            if kind != "num":
                bad.append(f"  {path}: {kind}")
                continue
            if not math.isfinite(a) or not math.isfinite(b):
                bad.append(f"  {path}: non-finite value {a!r} -> {b!r}")
                continue
            if abs(b - a) > args.atol + args.rtol * abs(a):
                rel = abs(b - a) / abs(a) if a != 0 else math.inf
                bad.append(f"  {path}: {a:.10g} -> {b:.10g}  (rel {rel:.3g})")
        status = "OK" if not bad else f"{len(bad)} differences"
        print(f"{old_file.name}: {status}")
        for line in bad[:40]:
            print(line)
        if len(bad) > 40:
            print(f"  ... {len(bad) - 40} more")
        n_bad += len(bad)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
