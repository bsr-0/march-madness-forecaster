"""Tests for public picks pipeline and leverage optimization."""

import json
import os

import pytest

from src.data.scrapers.espn_picks import ConsensusData, ESPNPicksScraper, aggregate_consensus
from src.optimization.leverage import analyze_pool, TeamMetadata
from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig


FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "public_picks_2026.json")


def _load_fixture() -> dict:
    with open(FIXTURE_PATH, "r") as f:
        return json.load(f)


def _normalize_public(consensus) -> dict:
    public = {}
    for team_id, picks in consensus.teams.items():
        row = {}
        for round_name, value in picks.as_dict.items():
            v = float(value)
            if v > 1.0:
                v = v / 100.0
            row[round_name] = v
        public[team_id] = row
    return public


def test_public_picks_end_to_end_aggregate_and_leverage():
    payload = _load_fixture()
    scraper = ESPNPicksScraper()

    espn = scraper._dict_to_consensus(payload)
    yahoo = scraper._dict_to_consensus(payload)
    cbs = scraper._dict_to_consensus(payload)

    consensus = aggregate_consensus(espn, yahoo, cbs)
    assert set(consensus.sources) == {"espn", "yahoo", "cbs"}

    # Regression: aggregate_consensus must handle subset weights
    # (e.g., when CBS scraper fails, only ESPN and Yahoo have weight).
    subset_weights = {"espn": 0.625, "yahoo": 0.375}
    empty_cbs = ConsensusData(sources=["cbs"])
    consensus_partial = aggregate_consensus(espn, yahoo, empty_cbs, weights=subset_weights)
    assert "espn" in consensus_partial.sources
    assert "yahoo" in consensus_partial.sources
    assert len(consensus_partial.teams) > 0

    public_picks = _normalize_public(consensus)

    model_probs = {
        "duke": {"R64": 0.97, "R32": 0.90, "S16": 0.78, "E8": 0.60, "F4": 0.42, "CHAMP": 0.28},
        "unc": {"R64": 0.93, "R32": 0.83, "S16": 0.58, "E8": 0.38, "F4": 0.26, "CHAMP": 0.14},
        "gonzaga": {"R64": 0.88, "R32": 0.69, "S16": 0.44, "E8": 0.27, "F4": 0.18, "CHAMP": 0.10},
        "midmajor": {"R64": 0.70, "R32": 0.22, "S16": 0.10, "E8": 0.05, "F4": 0.03, "CHAMP": 0.02},
    }

    team_metadata = {
        team_id: TeamMetadata(team_name=picks.team_name, seed=picks.seed, region=picks.region)
        for team_id, picks in consensus.teams.items()
    }

    analysis = analyze_pool(
        pool_size=100,
        model_probs=model_probs,
        public_picks=public_picks,
        team_metadata=team_metadata,
    )

    assert analysis.leverage_picks, "Expected leverage picks from aggregated public picks"
    assert analysis.pareto_brackets, "Expected pareto brackets from pool analysis"


def test_public_picks_fallback_to_model_probs():
    config = SOTAPipelineConfig(year=2026, scrape_live=False)
    pipeline = SOTAPipeline(config)

    model_probs = {
        "alpha": {"R64": 0.75, "R32": 0.55, "S16": 0.30, "E8": 0.15, "F4": 0.08, "CHAMP": 0.04},
        "beta": {"R64": 0.65, "R32": 0.45, "S16": 0.25, "E8": 0.12, "F4": 0.06, "CHAMP": 0.03},
    }

    public_picks = pipeline._load_public_picks(model_probs)
    assert public_picks == model_probs

    team_metadata = {
        "alpha": TeamMetadata(team_name="Alpha", seed=1, region="East"),
        "beta": TeamMetadata(team_name="Beta", seed=2, region="West"),
    }

    analysis = analyze_pool(
        pool_size=50,
        model_probs=model_probs,
        public_picks=public_picks,
        team_metadata=team_metadata,
    )

    assert analysis.recommended_strategy == "chalk"
    assert analysis.leverage_picks == []


# ---------------------------------------------------------------------------
# Leverage/Pareto optimizer test (merged from test_leverage_optimizer)
# ---------------------------------------------------------------------------


def _build_full_bracket_probs():
    model_probs = {}
    public_probs = {}
    metadata = {}

    for region in ("East", "West", "South", "Midwest"):
        for seed in range(1, 17):
            team_id = f"{region.lower()}_{seed}"
            seed_strength = (17 - seed) / 16.0
            model_probs[team_id] = {
                "R64": min(0.99, 0.45 + 0.50 * seed_strength),
                "R32": min(0.95, 0.20 + 0.60 * seed_strength),
                "S16": min(0.85, 0.10 + 0.55 * seed_strength),
                "E8": min(0.70, 0.05 + 0.45 * seed_strength),
                "F4": min(0.50, 0.03 + 0.35 * seed_strength),
                "CHAMP": min(0.30, 0.01 + 0.22 * seed_strength),
            }
            public_probs[team_id] = {
                "R64": min(0.99, 0.50 + 0.45 * seed_strength),
                "R32": min(0.97, 0.25 + 0.55 * seed_strength),
                "S16": min(0.90, 0.12 + 0.48 * seed_strength),
                "E8": min(0.75, 0.06 + 0.36 * seed_strength),
                "F4": min(0.55, 0.03 + 0.28 * seed_strength),
                "CHAMP": min(0.32, 0.01 + 0.18 * seed_strength),
            }
            metadata[team_id] = TeamMetadata(
                team_name=team_id.upper(),
                seed=seed,
                region=region,
            )

    return model_probs, public_probs, metadata


def test_pareto_optimizer_generates_full_63_game_bracket():
    model_probs, public_probs, metadata = _build_full_bracket_probs()
    analysis = analyze_pool(
        pool_size=600,
        model_probs=model_probs,
        public_picks=public_probs,
        team_metadata=metadata,
    )
    assert analysis.pareto_brackets

    bracket = analysis.pareto_brackets[-1]
    assert bracket.champion
    assert len(bracket.final_four) == 4
    assert "CHAMP" in bracket.picks
    assert len(bracket.picks) >= 63
    assert any(k.startswith("R64_") for k in bracket.picks)
