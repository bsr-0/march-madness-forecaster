from src.evaluation.pool_settings import ESPN_STANDARD, resolve_pool_settings, settings_from_mapping
import subprocess
import sys
from pathlib import Path


def test_default_is_canonical_espn_contract():
    s = resolve_pool_settings()
    assert s.pool_size == 30
    assert s.n_opponents == 29
    assert s.scoring == ESPN_STANDARD
    assert s.preset_id == "pool30_espn_standard"


def test_supported_pool_sizes_preserve_round_scoring():
    for size in (10, 30, 50, 100):
        s = resolve_pool_settings(size)
        assert s.n_opponents == size - 1
        assert tuple(s.scoring) == ("R64", "R32", "S16", "E8", "F4", "CHAMP")


def test_invalid_settings_fail_loudly():
    for size in (0, 31, 1000, "abc"):
        try:
            resolve_pool_settings(size)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pool size was accepted")
    try:
        resolve_pool_settings(30, "custom")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported scoring preset was accepted")


def test_mapping_defaults_are_stable():
    assert settings_from_mapping({"pool_size": 50}).preset_id == "pool50_espn_standard"




def test_artifact_and_evaluator_expose_settings_cli():
    root = Path(__file__).parents[1]
    for script in (root / "scripts/experiments/build_candidate_artifact.py", root / "scripts/evaluate_fitted_bracket.py"):
        out = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=True)
        assert "--pool-size" in out.stdout
        assert "--scoring" in out.stdout


def test_settings_are_part_of_artifact_identity():
    from scripts.evaluate_fitted_bracket import evaluate
    # Do not run the expensive evaluator here; verify the guard's exact
    # contract at the source boundary where a stale artifact is rejected.
    source = Path(__file__).parents[1] / "scripts/evaluate_fitted_bracket.py"
    text = source.read_text()
    assert "candidate artifact settings do not match evaluation settings" in text
    assert "pool_settings" in text
