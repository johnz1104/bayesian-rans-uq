"""Exercise the actual golden CLI with independent finite and malformed fixtures."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "tests" / "tools" / "compare_goldens.py"


def compare(tmp_path, old, new, *options):
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "sample.json").write_text(old)
    (new_dir / "sample.json").write_text(new)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(old_dir), str(new_dir), *options],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
@pytest.mark.parametrize("side", ["old", "new", "both"])
def test_nonfinite_numbers_fail(tmp_path, value, side):
    old = value if side in ("old", "both") else "2"
    new = value if side in ("new", "both") else "2"
    result = compare(tmp_path, '{"field": [' + old + "]}", '{"field": [' + new + "]}")
    assert result.returncode != 0, result.stdout + result.stderr


@pytest.mark.parametrize("old,new", [("true", "1"), ("1", "true"), ("false", "0")])
def test_boolean_numeric_type_changes_fail(tmp_path, old, new):
    result = compare(tmp_path, old, new)
    assert result.returncode != 0, result.stdout + result.stderr


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_empty_reference_directory_fails(tmp_path, missing):
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    if not missing:
        old_dir.mkdir()
    new_dir.mkdir()
    (new_dir / "sample.json").write_text("1")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(old_dir), str(new_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "old,new",
    [
        ('{"field": [1, 2]}', '{"field": [1]}'),
        ('{"field": [[1, 2]]}', '{"field": [[1], [2]]}'),
        ('{"field": [1]}', '{"field": {"0": 1}}'),
        ('{"field": 1}', "{}"),
        ('{"field": 1}', '{"field": 1.01}'),
    ],
)
def test_shape_keys_and_out_of_tolerance_changes_fail(tmp_path, old, new):
    result = compare(tmp_path, old, new)
    assert result.returncode != 0, result.stdout + result.stderr


def test_missing_candidate_file_fails(tmp_path):
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "sample.json").write_text("1")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(old_dir), str(new_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr


def test_finite_tolerance_and_metadata_policy_pass(tmp_path):
    result = compare(
        tmp_path,
        '{"field": [2, -2, 0], "flag": true, "notes": "old", "wall_seconds": 1}',
        '{"field": [2.000000000001, -2.000000000001, 1e-15], "flag": true, '
        '"notes": "new", "wall_seconds": 2}',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "sample.json: OK" in result.stdout


@pytest.mark.parametrize("option", ["--rtol", "--atol"])
@pytest.mark.parametrize("value", ["nan", "inf"])
def test_nonfinite_tolerances_fail(tmp_path, option, value):
    result = compare(tmp_path, "1", "100", option, value)
    assert result.returncode != 0, result.stdout + result.stderr


def test_candidate_without_reference_fails(tmp_path):
    compare(tmp_path, "1", "1")
    (tmp_path / "new" / "unreferenced.json").write_text("900")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "old"), str(tmp_path / "new")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr


@pytest.mark.parametrize("value", ["null", "[null]", "[[null]]"])
@pytest.mark.parametrize("side", ["old", "new", "both"])
def test_null_numerical_leaves_fail(tmp_path, value, side):
    old = value if side in ("old", "both") else "2"
    new = value if side in ("new", "both") else "2"
    result = compare(tmp_path, '{"field": ' + old + "}", '{"field": ' + new + "}")
    assert result.returncode != 0, result.stdout + result.stderr


@pytest.mark.parametrize("key", ["wall_seconds", "code", "build", "notes"])
def test_null_in_ignored_metadata_passes(tmp_path, key):
    result = compare(tmp_path, '{"field": 2, "' + key + '": null}', '{"field": 2}')
    assert result.returncode == 0, result.stdout + result.stderr
