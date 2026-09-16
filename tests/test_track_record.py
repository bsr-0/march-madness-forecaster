"""Parity for the track record (scripts/build_track_record.py).

"Would have won X% of pools" is only honest if the field it was measured in
is the field P(1st) was measured in, scored by the same rules. These tests
hold the artifact to that: same function objects, same configuration as the
candidate artifact, recorded parity against the shipped numbers, the
outcome equal to the canonical results, and the payload's embedded copy equal
to the file for exactly the brackets the payload carries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import build_track_record as tr  # noqa: E402
from scripts.experiments import build_candidate_artifact as bca  # noqa: E402
from scripts.experiments import objective_diversity_matrix as odm  # noqa: E402

RECORDS = sorted((REPO / "artifacts" / "track_record").glob("track_record_*.json"))


def _year(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def _artifact(rec: dict) -> dict:
    path = REPO / rec["inputs"]["candidates_artifact"]
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} not built on this checkout")
    return json.loads(path.read_text())


def test_track_record_scores_with_the_p1_referees_own_functions():
    """pool_p_first scores opponents with score_brackets_team_identity and
    splits ties with first_place_shares; the track record must use the same
    objects, not equivalents."""
    from src.optimization.payout import first_place_shares
    from src.simulation.pool_competition import score_brackets_team_identity

    assert tr.score_brackets_team_identity is score_brackets_team_identity
    assert tr.first_place_shares is first_place_shares
    assert odm.score_brackets_team_identity is score_brackets_team_identity
    assert tr.ESPN_SCORING is bca.ESPN_SCORING
    assert tr._encode_rows is bca._encode_rows
    # and the field comes from the same replay the fitted evaluation proved
    from scripts import evaluate_fitted_bracket as ev

    assert tr.rebuild_referee is ev.rebuild_referee
    assert tr.parity_check is ev.parity_check


@pytest.mark.parametrize("path", RECORDS, ids=[p.stem for p in RECORDS])
def test_recorded_parity_holds_against_the_artifact_on_disk(path: Path):
    rec = json.loads(path.read_text())
    assert rec["kind"] == tr.KIND
    art = _artifact(rec)
    assert art["meta"].get("generated_at") == rec["inputs"]["candidates_generated_at"], (
        f"{path.name}: candidate artifact rebuilt after this record; re-run build_track_record"
    )
    for k, v in rec["parity"].items():
        assert v["shipped"] == v["replayed"], f"{path.name}: {k}"
    for name, v in art["named_strategies"].items():
        assert rec["parity"][f"named:{name}"]["replayed"] == {"ev": v["ev"], "p1": v["p1"]}
    assert rec["scorer"]["p1_trials"] == art["meta"]["p1_trials"]
    assert rec["scorer"]["pool_size"] == art["meta"]["p1_pool_size"]
    assert rec["scorer"]["trials_seed"] == rec["scorer"]["seed"] + 7


@pytest.mark.parametrize("path", RECORDS, ids=[p.stem for p in RECORDS])
def test_strategy_picks_are_the_artifacts_and_the_fitted_evaluations(path: Path):
    rec = json.loads(path.read_text())
    art = _artifact(rec)
    for sid, named in tr.NAMED.items():
        assert rec["strategies"][sid]["w"] == art["named_strategies"][named]["w"], sid
    if "model" in rec["strategies"]:
        fe = json.loads((REPO / rec["strategies"]["model"]["source"]).read_text())
        assert rec["strategies"]["model"]["w"] == fe["w"]
        assert rec["strategies"]["model"]["fitted_eval_generated_at"] == fe["generated_at"]


@pytest.mark.parametrize("path", RECORDS, ids=[p.stem for p in RECORDS])
def test_outcome_is_the_canonical_tournament_result(path: Path):
    from scripts._common import load_tournament_results
    from src.simulation.pool_competition import actual_winners_by_round

    rec = json.loads(path.read_text())
    actual = actual_winners_by_round(load_tournament_results(_year(path)))
    assert tr.outcome_hash(actual) == rec["inputs"]["outcome_sha256"]


@pytest.mark.parametrize("path", RECORDS, ids=[p.stem for p in RECORDS])
def test_numbers_are_internally_consistent(path: Path):
    rec = json.loads(path.read_text())
    pool = rec["scorer"]["pool_size"]
    for sid, r in rec["strategies"].items():
        assert 0 <= r["won_share"] <= 1, sid
        assert 1 <= r["median_rank"] <= pool, sid
        assert r["points"] % 10 == 0 and 0 <= r["points"] <= 1920, sid   # ESPN totals are multiples of 10
        assert r["n_trials"] == rec["scorer"]["p1_trials"]
        # Winning most fields implies a top median rank, and vice versa.
        if r["won_share"] > 0.5:
            assert r["median_rank"] == 1, sid
        if r["median_rank"] == 1:
            assert r["won_share"] >= 0.25, sid


@pytest.mark.parametrize("path", RECORDS, ids=[p.stem for p in RECORDS])
def test_embedded_copy_equals_the_file_for_exactly_the_payloads_brackets(path: Path):
    season_path = REPO / "docs" / "data" / f"season_{_year(path)}.json"
    if not season_path.exists():
        pytest.skip("payload not built")
    season = json.loads(season_path.read_text())
    emb = season.get("track_record")
    if emb is None:
        return
    rec = json.loads(path.read_text())
    for sid, r in emb["strategies"].items():
        src = rec["strategies"][sid]
        for k in ("points", "won_share", "median_rank", "pool_median_points", "pool_best_points"):
            assert r[k] == src[k], (sid, k)
        if sid == "model":
            assert season["fitted_eval"]["w"] == src["w"]
        else:
            st = next(s for s in season["strategies"] if s["id"] == sid)
            assert st["picks"] == src["w"]
    assert emb["generated_at"] == rec["generated_at"]
