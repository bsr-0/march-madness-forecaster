"""Parity for the evaluated fitted-model bracket (scripts/evaluate_fitted_bracket.py).

The site shows three brackets with a P(1st) and an expected-points figure each.
Those numbers are only comparable if all three come from ONE scorer under ONE
configuration. Two of the brackets are scored inside the candidate-artifact
pipeline; the third is scored afterwards by a replay of that pipeline's
referee. These tests are the proof that the replay IS that scorer -- at the
level of function identity, of configuration, and of reproduced numbers --
and that the evaluation changed nothing about production selection.

Nothing here runs the replay (50,000 simulations); the evaluator runs it and
RECORDS its parity check. These tests hold that record to the artifact that
is actually on disk now, so a candidate artifact rebuilt after the evaluation
fails here rather than shipping numbers from two different builds.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import evaluate_fitted_bracket as ev  # noqa: E402
from scripts.experiments import build_candidate_artifact as bca  # noqa: E402
from src.simulation import bracket_topology as _bt  # noqa: E402

EVALS = sorted((REPO / "artifacts" / "fitted_eval").glob("fitted_eval_*.json"))


def _year(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def _artifact(e: dict) -> dict:
    """The candidate artifact an evaluation was scored against.

    artifacts/candidates/*.json is gitignored (the real artifacts are built,
    not committed), so on a fresh checkout the parity record has nothing to be
    held against. Skip, as tests/test_training_matrix.py does for unbuilt
    payloads, rather than fail on a missing file that is not a defect.
    """
    path = REPO / e["inputs"]["candidates_artifact"]
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} not built on this checkout")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- same scorer


def test_evaluator_binds_the_artifact_builders_scoring_functions_by_identity():
    """Not "equivalent" functions: the same objects, imported from the same modules.

    A reimplementation of expected_scores or pool_p_first -- however faithful
    -- is exactly what must never exist; identity is the check that cannot be
    satisfied by a copy.
    """
    for name in (
        "expected_scores", "round_marginals",            # EV
        "pool_p_first", "draw_selection_trials",        # P(1st)
        "build_seed_probabilities", "build_espn_pick_distribution",  # referee + opponents
        "simulate_bracket_outcomes", "PairwiseProbabilities",        # tournament draws
        "build_bracket_order", "ESPN_SCORING", "DEFAULT_POOL_SIZE",  # topology, rules, pool
        "_encode_rows", "_rating_sources", "resolve_field", "assert_pretournament_inputs",
    ):
        assert getattr(ev, name) is getattr(bca, name), name


def test_evaluator_scores_with_the_named_strategies_two_lines():
    """score() is the EV line and the P(1st) line of _champion_equity_strategy()."""
    src = (REPO / "scripts" / "evaluate_fitted_bracket.py").read_text()
    assert "expected_scores([winners], ref[\"marg\"], ESPN_SCORING)[0]" in src
    assert "pool_p_first(row, ref[\"p1_trials\"], ref[\"first_round\"])[0]" in src


def test_evaluator_never_imports_into_the_production_pipeline():
    """Selection cannot be influenced by something it does not know exists."""
    for p in (
        REPO / "scripts" / "experiments" / "build_candidate_artifact.py",
        REPO / "src" / "product" / "selection.py",
        REPO / "src" / "optimization" / "bracket_construction.py",
    ):
        if p.exists():
            assert "evaluate_fitted_bracket" not in p.read_text(), p
            assert "fitted_eval" not in p.read_text(), p


def test_evaluator_only_reads_the_candidate_artifact():
    src = (REPO / "scripts" / "evaluate_fitted_bracket.py").read_text()
    assert "candidates_{year}.json\"" in src
    # The only write is to its own directory.
    assert src.count("write_text(") == 1
    assert 'OUT_DIR / f"fitted_eval_{a.year}.json"' in src


# ------------------------------------------------------- recorded parity holds


@pytest.mark.parametrize("path", EVALS, ids=[p.stem for p in EVALS])
def test_recorded_parity_reproduced_the_shipped_numbers_exactly(path: Path):
    e = json.loads(path.read_text())
    assert e["kind"] == ev.KIND
    assert e["parity"], "an evaluation with no parity record is not evidence"
    for k, v in e["parity"].items():
        assert v["shipped"] == v["replayed"], f"{path.name}: {k} {v}"


@pytest.mark.parametrize("path", EVALS, ids=[p.stem for p in EVALS])
def test_parity_record_matches_the_candidate_artifact_on_disk_now(path: Path):
    """A rebuilt artifact invalidates the evaluation; it must not ship beside it."""
    e = json.loads(path.read_text())
    art = _artifact(e)
    assert art["meta"].get("generated_at") == e["inputs"]["candidates_generated_at"], (
        f"{path.name}: candidate artifact was rebuilt after this evaluation; re-run the evaluator"
    )
    for name, v in art["named_strategies"].items():
        rec = e["parity"][f"named:{name}"]
        assert rec["shipped"] == {"ev": v["ev"], "p1": v["p1"]}
        assert rec["replayed"] == {"ev": v["ev"], "p1": v["p1"]}
    for j, c in enumerate(art["candidates"][:5]):
        rec = e["parity"][f"candidate:{j}"]
        assert rec["replayed"] == {"ev": c["ev"], "p1": c["p1"]}


@pytest.mark.parametrize("path", EVALS, ids=[p.stem for p in EVALS])
def test_configuration_matches_the_artifact(path: Path):
    e = json.loads(path.read_text())
    art = _artifact(e)
    sc = e["scorer"]
    assert sc["n_sims_total"] == art["meta"]["n_sims"]
    assert sc["p1_trials"] == art["meta"]["p1_trials"]
    assert sc["pool_size"] == art["meta"]["p1_pool_size"] == bca.DEFAULT_POOL_SIZE
    assert sc["n_opponents"] == bca.DEFAULT_POOL_SIZE - 1
    assert sc["trials_seed"] == sc["seed"] + 7
    assert sc["torvik_sims_replayed"] == max(1, art["meta"]["n_sims"] // sc["n_rating_sources"])
    assert e["p1_assumption"] == art["meta"]["p1_assumption"]


# ------------------------------------------------- the bracket is well-formed,
# is the browser's, and is not a candidate


@pytest.mark.parametrize("path", EVALS, ids=[p.stem for p in EVALS])
def test_evaluated_bracket_is_one_bracket_on_the_artifacts_tree(path: Path):
    e = json.loads(path.read_text())
    art = _artifact(e)
    ids = [t["id"] for t in art["teams"]]
    fr = [ids[i] for i in art["first_round"]]
    winners = [[ids[i] for i in r] for r in e["w"]]
    assert [len(r) for r in winners] == [32, 16, 8, 4, 2, 1]
    _bt.winner_sets_to_bool_vector(winners, fr)   # strict; raises if not a bracket


@pytest.mark.parametrize("path", EVALS, ids=[p.stem for p in EVALS])
def test_evaluated_bracket_is_not_in_the_bank_and_selection_is_untouched(path: Path):
    e = json.loads(path.read_text())
    art = _artifact(e)
    assert "fitted" not in json.dumps(art["named_strategies"]).lower()
    assert all(c.get("src", "") != ev.KIND for c in art["candidates"])
    assert e["not_a_candidate"]


@pytest.mark.parametrize("path", EVALS, ids=[p.stem for p in EVALS])
def test_embedded_copy_in_the_payload_equals_the_file_or_is_absent(path: Path):
    """build_ui_payload embeds only when the fit inputs still match; if it did
    embed, it must have embedded THIS evaluation, unchanged."""
    e = json.loads(path.read_text())
    season_path = REPO / "docs" / "data" / f"season_{_year(path)}.json"
    if not season_path.exists():
        pytest.skip("payload not built")
    season = json.loads(season_path.read_text())
    fe = season.get("fitted_eval")
    if fe is None:
        return
    for k in ("kind", "w", "ev", "p1", "not_a_candidate", "p1_meaning", "scorer", "inputs", "generated_at"):
        assert fe[k] == e[k], k
    assert e["inputs"]["fit_inputs_hash"] == ev.fit_inputs_hash(season)
    assert e["inputs"]["training_sha256"] == ev.sha256_file(REPO / "docs" / "data" / "training.json")


def test_browser_still_produces_the_evaluated_bracket():
    """The strongest check: run docs/fit.js + docs/app.js now, compare picks.

    One season is enough to catch a drift in the JS or the payload; the hash
    checks above cover the rest per season without executing Node."""
    if not EVALS:
        pytest.skip("no evaluations on disk")
    if shutil.which("node") is None:
        pytest.skip("node not available; the JS harness cannot run here")
    path = EVALS[-1]
    e = json.loads(path.read_text())
    proc = subprocess.run(
        ["node", str(REPO / "scripts" / "fitted_bracket_js.js"), str(_year(path)), str(REPO / "docs")],
        capture_output=True, text=True, check=True,
    )
    js = json.loads(proc.stdout)
    assert js["w"] == e["w"], f"{path.name}: the browser's fitted bracket has changed; re-run the evaluator"
    assert js["keys"] == e["fit"]["keys"]


# ------------------------------------------- provenance says what the code does


def test_seed_referee_as_of_excludes_the_target_season():
    """as_of=Y tallies only seasons < Y. Behavioural, against the Kaggle file."""
    import csv

    from src.data.seed_pick_model import _win_rate

    kaggle_max = max(
        int(r["Season"]) for r in csv.DictReader(open(REPO / "data" / "kaggle" / "MNCAATourneyCompactResults.csv"))
    )
    # Dropping the last covered season must move at least one common cell.
    y = kaggle_max
    moved = any(_win_rate(a, b, "recent", y) != _win_rate(a, b, "recent", y + 1) for a, b in ((1, 8), (5, 12), (2, 7), (3, 6)))
    assert moved, f"as_of={y} vs {y + 1} changed nothing; as_of is not excluding season {y}"
    # And a season the file does not contain excludes nothing extra.
    assert _win_rate(1, 8, "recent", kaggle_max + 1) == _win_rate(1, 8, "recent", None)


def test_candidate_builder_passes_as_of_year_to_the_referee():
    src = (REPO / "scripts" / "experiments" / "build_candidate_artifact.py").read_text()
    assert "build_seed_probabilities(seeds, as_of=year)" in src


def test_provenance_caveat_describes_point_in_time_construction():
    """The shipped artifacts' caveat used to claim the referee table 'includes the
    target season's own results' while build() passed as_of=year, which
    excludes them. Provenance text must track the implementation."""
    prov = bca.assert_pretournament_inputs(2026)["seed_head_to_head"]
    assert prov.get("point_in_time") is True
    cav = prov["caveat"]
    assert "as_of=2026" in cav and "strictly before 2026" in cav
    assert "includes the target season" not in cav


@pytest.mark.parametrize("path", sorted((REPO / "artifacts" / "candidates").glob("candidates_*.json")),
                         ids=lambda p: p.stem)
def test_artifacts_built_after_the_correction_carry_the_corrected_caveat(path: Path):
    """Frozen artifacts predate the fix and are not rewritten (checkpoint 2);
    anything built from now on must carry the corrected text."""
    art = json.loads(path.read_text())
    prov = art.get("provenance", {}).get("seed_head_to_head", {})
    if prov.get("point_in_time") is not True:
        pytest.skip("frozen artifact from before the provenance correction (2026-09-16)")
    assert "includes the target season" not in prov["caveat"]
    assert f"as_of={art['year']}" in prov["caveat"]


def test_ev_and_p1_table_distinction_is_documented_not_hidden():
    """Displayed EV and P(1st) reproduce the production definitions, which use
    DIFFERENT probability tables (Torvik log5 marginals; seed-rate referee).
    That is a known characteristic of the framework, recorded in every
    evaluation and in the evaluator's own docstring -- not an assertion that
    the two come from one unified model."""
    src = (REPO / "scripts" / "evaluate_fitted_bracket.py").read_text()
    assert "different probability tables" in src
    for path in EVALS:
        e = json.loads(path.read_text())
        assert "different probability tables" in e["scorer"]["note"], path.name
        assert "Torvik" in e["scorer"]["ev"] and "seed-rate" in e["scorer"]["p1"], path.name
