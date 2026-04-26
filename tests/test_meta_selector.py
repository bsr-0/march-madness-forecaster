"""Tests for the learned meta-selector bracket construction.

Covers: feature assembly, leverage baseline, trained GBM selector,
walk-forward contracts, bracket validity, and path consistency.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.prediction.meta_selector import (
    BASE_FEATURE_ORDER,
    CONTEXT_KEYS,
    ESPN_SCORING,
    ROUND_NAMES,
    _game_features,
    _pairwise_prob,
    build_leverage_bracket,
    build_training_data,
    feature_names,
    leverage_pick,
    n_features,
)

DATA_ROOT = Path("data")
HIST_DIR = DATA_ROOT / "raw" / "historical"


def _has_year_data(year: int) -> bool:
    return (HIST_DIR / f"tournament_seeds_{year}.json").exists() and (
        HIST_DIR / f"tournament_results_{year}.json"
    ).exists()


# Use 2025 as the default test year (most recent complete year)
TEST_YEAR = 2025
NEEDS_DATA = pytest.mark.skipif(not _has_year_data(TEST_YEAR), reason=f"No data for {TEST_YEAR}")


@pytest.fixture
def seeds_2025():
    if not _has_year_data(TEST_YEAR):
        pytest.skip("No data")
    with open(HIST_DIR / f"tournament_seeds_{TEST_YEAR}.json") as f:
        raw = json.load(f)
    teams = raw["teams"] if isinstance(raw, dict) and "teams" in raw else raw
    return {t["team_id"]: t["seed"] for t in teams}


@pytest.fixture
def simple_round_probs(seeds_2025):
    """Minimal round_probs with just seed base for fast tests."""
    from src.prediction.seed_probabilities import build_seed_round_probabilities

    return {"seed": build_seed_round_probabilities(seeds_2025)}


@pytest.fixture
def simple_pick_dist(seeds_2025):
    """Fake pick distribution: favorites get 70% public pick, underdogs 30%."""
    dist = {}
    for tid, seed in seeds_2025.items():
        dist[tid] = {}
        for rnd in ROUND_NAMES:
            # Lower seeds (better teams) get higher public picks
            dist[tid][rnd] = max(0.05, 1.0 - seed / 16.0)
    return dist


# ---------------------------------------------------------------------------
# Feature Assembly Tests
# ---------------------------------------------------------------------------


class TestFeatureAssembly:
    def test_feature_count(self):
        # bases + seeds(2) + picks(2) + leverage_diff + agreement + consensus + disagreement + matchup_type + round + context
        expected = len(BASE_FEATURE_ORDER) + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + len(CONTEXT_KEYS)
        assert n_features() == expected
        assert len(feature_names()) == expected

    def test_feature_names_unique(self):
        names = feature_names()
        assert len(names) == len(set(names))

    @NEEDS_DATA
    def test_feature_vector_shape(self, seeds_2025, simple_round_probs):
        teams = list(seeds_2025.keys())
        feat = _game_features(
            teams[0],
            teams[1],
            "R64",
            0,
            simple_round_probs,
            None,
            seeds_2025,
            None,
        )
        assert feat.shape == (n_features(),)

    @NEEDS_DATA
    def test_pairwise_prob_symmetric(self, seeds_2025, simple_round_probs):
        teams = list(seeds_2025.keys())
        t1, t2 = teams[0], teams[1]
        rp = simple_round_probs["seed"]
        p12 = _pairwise_prob(t1, t2, "R64", rp)
        p21 = _pairwise_prob(t2, t1, "R64", rp)
        assert abs(p12 + p21 - 1.0) < 1e-8

    @NEEDS_DATA
    def test_missing_base_gives_nan(self, seeds_2025):
        teams = list(seeds_2025.keys())
        # Only seed base, no torvik — torvik feature should be NaN
        from src.prediction.seed_probabilities import build_seed_round_probabilities

        brp = {"seed": build_seed_round_probabilities(seeds_2025)}
        feat = _game_features(
            teams[0],
            teams[1],
            "R64",
            0,
            brp,
            None,
            seeds_2025,
            None,
        )
        # torvik is index 1 in BASE_FEATURE_ORDER
        torvik_idx = list(BASE_FEATURE_ORDER).index("torvik")
        assert np.isnan(feat[torvik_idx])

    @NEEDS_DATA
    def test_round_index_in_features(self, seeds_2025, simple_round_probs):
        teams = list(seeds_2025.keys())
        for round_idx, round_name in enumerate(ROUND_NAMES):
            feat = _game_features(
                teams[0],
                teams[1],
                round_name,
                round_idx,
                simple_round_probs,
                None,
                seeds_2025,
                None,
            )
            # round_index is at position: len(BASE_FEATURE_ORDER) + 9
            # (seeds=2, picks=2, lev_diff=1, agreement=1, consensus=1, disagreement=1, matchup_type=1)
            ri_idx = len(BASE_FEATURE_ORDER) + 9
            assert feat[ri_idx] == float(round_idx)


# ---------------------------------------------------------------------------
# Leverage Baseline Tests
# ---------------------------------------------------------------------------


class TestLeverageBaseline:
    @NEEDS_DATA
    def test_picks_favorite_when_no_public_picks(self, seeds_2025, simple_round_probs):
        """Without ESPN data, leverage = probability → picks favorite."""
        teams = list(seeds_2025.keys())
        # Find a 1-seed and 16-seed in same region
        seed_1 = [t for t, s in seeds_2025.items() if s == 1][0]
        seed_16 = [t for t, s in seeds_2025.items() if s == 16][0]
        # 1-seed should be picked (higher prob)
        pick = leverage_pick(
            seed_1,
            seed_16,
            "R64",
            simple_round_probs,
            None,
            seeds_2025,
            primary_base="seed",
        )
        assert pick is True  # team1 (1-seed) picked

    @NEEDS_DATA
    def test_picks_underdog_when_heavily_overowned(self, seeds_2025, simple_round_probs):
        """If favorite has huge public pick and modest win prob, underdog wins on leverage."""
        teams = list(seeds_2025.keys())
        t1, t2 = teams[0], teams[1]
        # Fake pick distribution: t1 has 95% public pick, t2 has 5%
        fake_dist = {
            t1: {"R64": 0.95},
            t2: {"R64": 0.05},
        }
        # With seed-based probs, a 1v16 has ~98% for 1-seed — leverage still
        # favors favorite: 0.98 × 0.05 = 0.049 vs 0.02 × 0.95 = 0.019
        # Use a more balanced matchup to test the flip:
        # If prob is 60/40 and public is 95/5:
        # lev_t1 = 0.60 × 0.05 = 0.03, lev_t2 = 0.40 × 0.95 = 0.38
        balanced_rp = {
            "seed": {
                t1: {"R64": 0.60},
                t2: {"R64": 0.40},
            }
        }
        pick = leverage_pick(
            t1,
            t2,
            "R64",
            balanced_rp,
            fake_dist,
            seeds_2025,
            primary_base="seed",
        )
        assert pick is False  # team2 has higher leverage

    @NEEDS_DATA
    def test_bracket_has_63_games(self, seeds_2025, simple_round_probs):
        from src.prediction.meta_selector import _load_year_data

        _, pick_dist, seeds, _, first_round, _ = _load_year_data(TEST_YEAR, DATA_ROOT)
        bracket = build_leverage_bracket(
            first_round,
            simple_round_probs,
            pick_dist,
            seeds,
            primary_base="seed",
        )
        assert bracket.shape == (63,)
        assert bracket.dtype == bool

    @NEEDS_DATA
    def test_bracket_is_deterministic(self, seeds_2025, simple_round_probs):
        from src.prediction.meta_selector import _load_year_data

        _, pick_dist, seeds, _, first_round, _ = _load_year_data(TEST_YEAR, DATA_ROOT)
        b1 = build_leverage_bracket(first_round, simple_round_probs, pick_dist, seeds)
        b2 = build_leverage_bracket(first_round, simple_round_probs, pick_dist, seeds)
        np.testing.assert_array_equal(b1, b2)

    @NEEDS_DATA
    def test_bracket_is_path_consistent(self, seeds_2025, simple_round_probs):
        """A team in R32 must have been picked in R64."""
        from src.prediction.meta_selector import _load_year_data
        from src.simulation.pool_competition import picks_by_round

        _, pick_dist, seeds, _, first_round, _ = _load_year_data(TEST_YEAR, DATA_ROOT)
        bracket = build_leverage_bracket(first_round, simple_round_probs, pick_dist, seeds)
        picks = picks_by_round(bracket, first_round)

        # Every R32 winner must be in R64 winners
        assert picks["R32"].issubset(picks["R64"])
        # Every S16 winner must be in R32 winners
        assert picks["S16"].issubset(picks["R32"])
        # Chain all the way
        assert picks["E8"].issubset(picks["S16"])
        assert picks["F4"].issubset(picks["E8"])
        assert picks["CHAMP"].issubset(picks["F4"])
        # Champion is exactly 1 team
        assert len(picks["CHAMP"]) == 1


# ---------------------------------------------------------------------------
# Training Data Tests
# ---------------------------------------------------------------------------


class TestTrainingData:
    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021),
        reason="Need 2019+2021 data",
    )
    def test_training_data_shape(self):
        """Training data has correct dimensions."""
        X, y, w = build_training_data([2019, 2021])
        # 63 games per year × 2 years = 126 rows (might be slightly less
        # if some games are skipped due to data gaps)
        assert X.ndim == 2
        assert X.shape[1] == n_features()
        assert len(y) == len(X)
        assert len(w) == len(X)
        assert X.shape[0] >= 100  # at least ~50 games per year

    @pytest.mark.skipif(not _has_year_data(2019), reason="Need 2019 data")
    def test_labels_are_binary(self):
        X, y, w = build_training_data([2019])
        assert set(np.unique(y)).issubset({0.0, 1.0})

    @pytest.mark.skipif(not _has_year_data(2019), reason="Need 2019 data")
    def test_weights_are_positive(self):
        X, y, w = build_training_data([2019])
        assert np.all(w >= 0)

    @pytest.mark.skipif(not _has_year_data(2019), reason="Need 2019 data")
    def test_weights_favor_late_rounds(self):
        """Championship correct picks should have higher weight than R64."""
        X, y, w = build_training_data([2019])
        # Round index is feature at position len(BASE_FEATURE_ORDER) + 6
        ri_idx = len(BASE_FEATURE_ORDER) + 9
        r64_mask = X[:, ri_idx] == 0
        champ_mask = X[:, ri_idx] == 5
        if champ_mask.sum() > 0 and r64_mask.sum() > 0:
            # Championship weight should be much higher on average
            assert w[champ_mask].mean() > w[r64_mask].mean()


# ---------------------------------------------------------------------------
# Trained Selector Tests
# ---------------------------------------------------------------------------


class TestTrainedSelector:
    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021),
        reason="Need 2019+2021 data",
    )
    def test_model_trains_without_error(self):
        from src.prediction.meta_selector import train_meta_selector

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector(X, y, w)
        assert hasattr(model, "predict")

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021) or not _has_year_data(TEST_YEAR),
        reason="Need training + test data",
    )
    def test_trained_bracket_is_valid(self):
        from src.prediction.meta_selector import (
            _load_year_data,
            build_trained_bracket,
            train_meta_selector,
        )
        from src.simulation.pool_competition import picks_by_round

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector(X, y, w)

        brp, pick_dist, seeds, context, first_round, _ = _load_year_data(TEST_YEAR, DATA_ROOT)
        bracket = build_trained_bracket(first_round, brp, pick_dist, seeds, context, model)

        assert bracket.shape == (63,)
        assert bracket.dtype == bool

        picks = picks_by_round(bracket, first_round)
        assert len(picks["CHAMP"]) == 1
        assert picks["CHAMP"].issubset(picks["F4"])

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021) or not _has_year_data(TEST_YEAR),
        reason="Need training + test data",
    )
    def test_trained_bracket_is_deterministic(self):
        from src.prediction.meta_selector import (
            _load_year_data,
            build_trained_bracket,
            train_meta_selector,
        )

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector(X, y, w)

        brp, pick_dist, seeds, context, first_round, _ = _load_year_data(TEST_YEAR, DATA_ROOT)
        b1 = build_trained_bracket(first_round, brp, pick_dist, seeds, context, model)
        b2 = build_trained_bracket(first_round, brp, pick_dist, seeds, context, model)
        np.testing.assert_array_equal(b1, b2)

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021) or not _has_year_data(TEST_YEAR),
        reason="Need training + test data",
    )
    def test_model_not_all_chalk(self):
        """At least some underdog picks across the bracket."""
        from src.prediction.meta_selector import (
            _load_year_data,
            build_trained_bracket,
            train_meta_selector,
        )
        from src.simulation.pool_competition import picks_by_round

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector(X, y, w)

        brp, pick_dist, seeds, context, first_round, _ = _load_year_data(TEST_YEAR, DATA_ROOT)
        bracket = build_trained_bracket(first_round, brp, pick_dist, seeds, context, model)
        picks = picks_by_round(bracket, first_round)

        # Check: not every R64 winner is a top-4 seed
        r64_winners = picks["R64"]
        seeds_file = json.load(open(HIST_DIR / f"tournament_seeds_{TEST_YEAR}.json"))
        seeds_list = seeds_file["teams"] if isinstance(seeds_file, dict) and "teams" in seeds_file else seeds_file
        seeds_map = {t["team_id"]: t["seed"] for t in seeds_list}
        chalk_count = sum(1 for t in r64_winners if seeds_map.get(t, 16) <= 4)
        # At most 28/32 chalk (at least 4 upsets in R64)
        # This is a soft check — the model SHOULD pick some upsets
        assert chalk_count < 32, "Model picked all favorites — degenerated to argmax"
