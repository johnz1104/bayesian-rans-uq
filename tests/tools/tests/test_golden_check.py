"""Test the release golden gate through its CLI and the real comparator."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tests" / "tools" / "check_goldens.py"
GROUPS = {
    "constrained_omega": (
        "couette_re171_ny96",
        "periodic_channel_re180",
        "bfs_lemoin_uniform_inlet",
        "cosine_bump_periodic",
    ),
    "dbns_stage_consistent": ("dbns_shock_tubes", "dbns_mms"),
}


@pytest.fixture
def gate(tmp_path):
    references = tmp_path / "references"
    for group, cases in GROUPS.items():
        directory = references / group
        directory.mkdir(parents=True)
        for case in cases:
            (directory / f"{case}.json").write_text(json.dumps({"case": case, "value": 1.25}))
    recorder = tmp_path / "recorder"
    recorder.write_text(
        f"#!{sys.executable}\n"
        """import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1])
case = sys.argv[4]
mode = os.environ.get("FAKE_GOLDEN_MODE", "pass")
with open(os.environ["FAKE_GOLDEN_LOG"], "a") as stream:
    stream.write(json.dumps({"out": str(out), "case": case}) + "\\n")
assert not (out / (case + ".json")).exists(), "candidate was reused"
if mode != "missing":
    value = 1.26 if mode == "perturb" else 1.25
    (out / (case + ".json")).write_text(json.dumps({"case": case, "value": value}))
if mode == "extra":
    (out / "unrequested.json").write_text("1")
if mode == "exit":
    sys.exit(9)
"""
    )
    recorder.chmod(0o755)
    log = tmp_path / "recorded.jsonl"

    def run(*options, mode="pass"):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--recorder",
                str(recorder),
                "--reference-root",
                str(references),
                *map(str, options),
            ],
            env={**os.environ, "FAKE_GOLDEN_LOG": str(log), "FAKE_GOLDEN_MODE": mode},
            capture_output=True,
            text=True,
            check=False,
        )

    return run, references, log


def test_six_current_cases_use_separate_fresh_candidate_directories(gate, tmp_path):
    run, _, log = gate
    for repeat in range(2):
        output = tmp_path / f"candidate-{repeat}"
        result = run("--output", output)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.count(": OK") == 6
        for group, cases in GROUPS.items():
            assert {path.stem for path in (output / group).glob("*.json")} == set(cases)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 12
    assert {entry["case"] for entry in records} == {
        case for cases in GROUPS.values() for case in cases
    }
    assert len({entry["out"] for entry in records}) == 4


@pytest.mark.parametrize("mode", ["perturb", "missing", "extra", "exit"])
def test_recorder_or_comparison_failure_fails_gate(gate, mode):
    run, _, log = gate
    result = run(mode=mode)
    assert result.returncode != 0, result.stdout + result.stderr
    assert log.exists(), "the failing recorder control did not execute"


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_reference_inventory_cannot_silently_shrink_or_expand(gate, change):
    run, references, log = gate
    directory = references / "constrained_omega"
    if change == "missing":
        (directory / "couette_re171_ny96.json").unlink()
    else:
        (directory / "unrequested.json").write_text("1")
    result = run()
    assert result.returncode != 0, result.stdout + result.stderr
    assert "reference inventory" in result.stderr
    assert not log.exists(), "invalid reference inventory must fail before a solve"


def test_existing_output_is_rejected_without_modification(gate, tmp_path):
    run, _, log = gate
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "preserved.json"
    marker.write_text("42")
    result = run("--output", output)
    assert result.returncode != 0, result.stdout + result.stderr
    assert marker.read_text() == "42"
    assert not log.exists()


def test_default_temporary_candidates_are_cleaned_after_success(gate):
    run, _, log = gate
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 6
    assert all(not Path(entry["out"]).exists() for entry in records)


def test_missing_recorder_fails_before_a_solve(gate, tmp_path):
    run, _, log = gate
    result = run("--recorder", tmp_path / "missing-recorder")
    assert result.returncode != 0, result.stdout + result.stderr
    assert "recorder" in result.stderr
    assert not log.exists()


def test_native_all_preflight_requires_selected_python(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "tests" / "tools" / "check.sh"), "--native", "--all", "--preflight"],
        env={**os.environ, "CFDR_PYTHON": str(tmp_path / "missing-python")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "Required interpreter unavailable" in result.stderr


def test_recorder_cli_reports_missing_output_directory(tmp_path):
    recorder = Path(os.environ.get("CFDR_GOLDEN_RECORDER", ROOT / "build/dev/record_goldens"))
    assert recorder.is_file(), f"Build record_goldens or set CFDR_GOLDEN_RECORDER: {recorder}"
    output = tmp_path / "missing-parent" / "output"
    result = subprocess.run(
        [str(recorder), str(output), "synthetic-cli-io-control", "0", "dbns_shock_tubes"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "failed to record dbns_shock_tubes" in result.stderr
    assert "golden output directory does not exist" in result.stderr
    assert not output.exists()


def test_recorder_metadata_round_trips_json_control_characters(tmp_path):
    recorder = Path(os.environ.get("CFDR_GOLDEN_RECORDER", ROOT / "build/dev/record_goldens"))
    assert recorder.is_file(), f"Build record_goldens or set CFDR_GOLDEN_RECORDER: {recorder}"
    # Process arguments cannot contain NUL; the native writer test covers that byte.
    metadata = "".join(chr(byte) for byte in range(1, 32)) + '"\\ω'
    result = subprocess.run(
        [str(recorder), str(tmp_path), metadata, "0", "dbns_shock_tubes"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = json.loads((tmp_path / "dbns_shock_tubes.json").read_text())
    assert recorded["code"]["git_sha"] == metadata
