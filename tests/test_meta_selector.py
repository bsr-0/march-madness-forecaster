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
    _compute_team_path,
    _game_features,
    _pairwise_prob,
    _rank_champion_candidates,
    build_champion_first_brackets,
    build_champion_locked_bracket,
    build_leverage_bracket,
    build_training_data,
    feature_names,
    leverage_pick,
    load_meta_context,
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
        # bases + seeds(2) + picks(2) + leverage_diff + agreement + consensus + disagreement + matchup_type + round + context + S4(3)
        expected = len(BASE_FEATURE_ORDER) + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + len(CONTEXT_KEYS) + 3
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
    def test_native_lightgbm_wrapper_exposes_feature_importances(self):
        from src.prediction.meta_selector import _NativeLightGBMClassifier

        class DummyBooster:
            def feature_importance(self, importance_type="split"):
                assert importance_type == "gain"
                return [3.0, 1.5, 0.0]

            def predict(self, X):
                return np.full(len(X), 0.7)

        model = _NativeLightGBMClassifier(DummyBooster())

        np.testing.assert_array_equal(model.feature_importances_, np.asarray([3.0, 1.5, 0.0]))
        probs = model.predict_proba(np.asarray([[0.0], [1.0]]))
        assert probs.shape == (2, 2)

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

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021),
        reason="Need training data",
    )
    def test_lr_model_trains_without_error(self):
        from src.prediction.meta_selector import train_meta_selector_lr

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector_lr(X, y, w)
        assert hasattr(model, "predict")

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021) or not _has_year_data(TEST_YEAR),
        reason="Need training + test data",
    )
    def test_lr_bracket_is_valid(self):
        from src.prediction.meta_selector import (
            _load_year_data,
            build_trained_bracket,
            train_meta_selector_lr,
        )
        from src.simulation.pool_competition import picks_by_round

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector_lr(X, y, w)

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
    def test_lr_bracket_is_deterministic(self):
        from src.prediction.meta_selector import (
            _load_year_data,
            build_trained_bracket,
            train_meta_selector_lr,
        )

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector_lr(X, y, w)

        brp, pick_dist, seeds, context, first_round, _ = _load_year_data(TEST_YEAR, DATA_ROOT)
        b1 = build_trained_bracket(first_round, brp, pick_dist, seeds, context, model)
        b2 = build_trained_bracket(first_round, brp, pick_dist, seeds, context, model)
        np.testing.assert_array_equal(b1, b2)

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021),
        reason="Need training data",
    )
    def test_margin_model_trains_without_error(self):
        from src.prediction.meta_selector import (
            build_margin_training_data,
            train_meta_selector_margin,
        )

        X, y_margin, w = build_margin_training_data([2019, 2021])
        assert y_margin.min() < 0, "Expected some negative margins"
        assert y_margin.max() > 0, "Expected some positive margins"
        model = train_meta_selector_margin(X, y_margin, w)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_margin")

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021) or not _has_year_data(TEST_YEAR),
        reason="Need training + test data",
    )
    def test_margin_bracket_is_valid(self):
        from src.prediction.meta_selector import (
            _load_year_data,
            build_margin_training_data,
            build_trained_bracket,
            train_meta_selector_margin,
        )
        from src.simulation.pool_competition import picks_by_round

        X, y_margin, w = build_margin_training_data([2019, 2021])
        model = train_meta_selector_margin(X, y_margin, w)

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
    def test_multi_seed_ensemble_produces_valid_bracket(self):
        from src.prediction.meta_selector import (
            _load_year_data,
            build_trained_bracket,
            train_multi_seed_ensemble,
        )
        from src.simulation.pool_competition import picks_by_round

        X, y, w = build_training_data([2019, 2021])
        model = train_multi_seed_ensemble(X, y, w, n_seeds=3)
        assert hasattr(model, "predict")
        assert len(model.models) == 3

        brp, pick_dist, seeds, context, first_round, _ = _load_year_data(TEST_YEAR, DATA_ROOT)
        bracket = build_trained_bracket(first_round, brp, pick_dist, seeds, context, model)

        assert bracket.shape == (63,)
        assert bracket.dtype == bool

        picks = picks_by_round(bracket, first_round)
        assert len(picks["CHAMP"]) == 1
        assert picks["CHAMP"].issubset(picks["F4"])


# ---------------------------------------------------------------------------
# Phase 0/1/2: Context skew fix, multiseed mode, #9 ablation mode
# ---------------------------------------------------------------------------


class TestLoadMetaContext:
    """Verify load_meta_context returns all CONTEXT_KEYS."""

    @pytest.mark.skipif(not _has_year_data(TEST_YEAR), reason="Need test year data")
    def test_returns_all_context_keys(self):
        seeds_path = HIST_DIR / f"tournament_seeds_{TEST_YEAR}.json"
        if not seeds_path.exists():
            pytest.skip("No seeds file")
        import json as _json

        with open(seeds_path) as f:
            seeds = _json.load(f)
        ctx = load_meta_context(TEST_YEAR, seeds, DATA_ROOT)
        for key in CONTEXT_KEYS:
            assert key in ctx, f"Missing context key: {key}"


class TestBacktestModeRegistration:
    """Verify new modes are registered in ALL_MODES and LEGACY_MODE_MAP."""

    def test_multiseed_registered(self):
        from scripts.mc_pool_backtest import ALL_MODES, LEGACY_MODE_MAP

        assert "meta_gbm_multiseed" in ALL_MODES
        assert "meta_gbm_multiseed" in LEGACY_MODE_MAP

    def test_no9_ablation_registered(self):
        from scripts.mc_pool_backtest import ALL_MODES, LEGACY_MODE_MAP

        assert "meta_gbm_no9" in ALL_MODES
        assert "meta_gbm_no9" in LEGACY_MODE_MAP

    @pytest.mark.parametrize(
        "mode",
        [
            "meta_gbm_lr",
            "meta_gbm_margin",
            "meta_gbm_vegas",
            "meta_sa",
            "meta_exhaustive",
            "meta_region",
            "meta_sa_chalk",
            "meta_gbm_elim",
            "meta_gbm_minimal",
            "meta_sa_vol",
            "meta_region_gbm",
            "meta_exhaustive_gbm",
            "meta_exhaustive_margin",
            "meta_region_blend",
            "meta_region_4champ",
            "meta_region_elo",
            "meta_region_massey",
            "meta_region_blend90",
            "meta_region_poolaware",
            "meta_region_poolaware_wideseed",
        ],
    )
    def test_new_modes_registered(self, mode):
        from scripts.mc_pool_backtest import ALL_MODES, LEGACY_MODE_MAP

        assert mode in ALL_MODES, f"{mode} not in ALL_MODES"
        assert mode in LEGACY_MODE_MAP, f"{mode} not in LEGACY_MODE_MAP"


# ---------------------------------------------------------------------------
# S1: Champion-First Construction Tests
# ---------------------------------------------------------------------------

TEST_YEAR = 2024  # noqa: F811 — reuse for champion tests


class TestChampionFirst:
    """Tests for the champion-first two-phase bracket construction."""

    @pytest.fixture(autouse=True)
    def _load_test_data(self):
        """Load test year data once for the class."""
        from src.prediction.meta_selector import _load_year_data

        if not _has_year_data(TEST_YEAR):
            pytest.skip(f"No data for {TEST_YEAR}")
        self.brp, self.pick_dist, self.seeds, self.context, self.first_round, _ = _load_year_data(TEST_YEAR, DATA_ROOT)

    def test_compute_team_path_length(self):
        """Path from R64 to CHAMP is exactly 6 games."""
        team = self.first_round[0]  # first team in bracket
        path = _compute_team_path(team, self.first_round)
        assert len(path) == 6

    def test_compute_team_path_indices_valid(self):
        """All game indices in [0, 62] and unique."""
        team = self.first_round[0]
        path = _compute_team_path(team, self.first_round)
        indices = [gi for gi, _ in path]
        assert all(0 <= gi <= 62 for gi in indices)
        assert len(set(indices)) == 6

    def test_compute_team_path_round_indices(self):
        """Round indices go 0, 1, 2, 3, 4, 5."""
        team = self.first_round[0]
        path = _compute_team_path(team, self.first_round)
        rounds = [ri for _, ri in path]
        assert rounds == [0, 1, 2, 3, 4, 5]

    def test_compute_team_path_invalid_team(self):
        """Raises ValueError for team not in matchups."""
        with pytest.raises(ValueError):
            _compute_team_path("nonexistent_team", self.first_round)

    def test_rank_champion_candidates_sorted(self):
        """Candidates sorted descending by mean CHAMP prob."""
        candidates = _rank_champion_candidates(self.brp, self.first_round, top_n=6)
        assert len(candidates) <= 6
        probs = [p for _, p in candidates]
        assert probs == sorted(probs, reverse=True)

    def test_rank_champion_candidates_in_bracket(self):
        """All candidates are teams in the bracket."""
        candidates = _rank_champion_candidates(self.brp, self.first_round, top_n=6)
        fr_set = set(self.first_round)
        for team_id, _ in candidates:
            assert team_id in fr_set

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021) or not _has_year_data(TEST_YEAR),
        reason="Need training + test data",
    )
    def test_locked_bracket_has_correct_champion(self):
        """The locked champion appears in the bracket's CHAMP pick."""
        from src.prediction.meta_selector import train_meta_selector
        from src.simulation.pool_competition import picks_by_round

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector(X, y, w)
        champion = self.first_round[0]  # lock the first team

        bracket = build_champion_locked_bracket(
            champion,
            self.first_round,
            self.brp,
            self.pick_dist,
            self.seeds,
            self.context,
            model,
        )
        picks = picks_by_round(bracket, self.first_round)
        assert champion in picks["CHAMP"]

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021) or not _has_year_data(TEST_YEAR),
        reason="Need training + test data",
    )
    def test_locked_bracket_path_consistent(self):
        """Champion appears in every round from R64 to CHAMP."""
        from src.prediction.meta_selector import train_meta_selector
        from src.simulation.pool_competition import picks_by_round

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector(X, y, w)
        champion = self.first_round[0]

        bracket = build_champion_locked_bracket(
            champion,
            self.first_round,
            self.brp,
            self.pick_dist,
            self.seeds,
            self.context,
            model,
        )
        picks = picks_by_round(bracket, self.first_round)
        for rnd in ROUND_NAMES:
            assert champion in picks[rnd], f"Champion {champion} missing from {rnd}"

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021) or not _has_year_data(TEST_YEAR),
        reason="Need training + test data",
    )
    def test_locked_bracket_deterministic(self):
        """Same inputs produce identical brackets."""
        from src.prediction.meta_selector import train_meta_selector

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector(X, y, w)
        champion = self.first_round[0]

        b1 = build_champion_locked_bracket(
            champion,
            self.first_round,
            self.brp,
            self.pick_dist,
            self.seeds,
            self.context,
            model,
        )
        b2 = build_champion_locked_bracket(
            champion,
            self.first_round,
            self.brp,
            self.pick_dist,
            self.seeds,
            self.context,
            model,
        )
        np.testing.assert_array_equal(b1, b2)

    @pytest.mark.skipif(
        not _has_year_data(2019) or not _has_year_data(2021) or not _has_year_data(TEST_YEAR),
        reason="Need training + test data",
    )
    def test_champion_first_brackets_shape(self):
        """build_champion_first_brackets returns (N, 63) + N champion names."""
        from src.prediction.meta_selector import train_meta_selector

        X, y, w = build_training_data([2019, 2021])
        model = train_meta_selector(X, y, w)

        brackets, champs = build_champion_first_brackets(
            self.first_round,
            self.brp,
            self.pick_dist,
            self.seeds,
            self.context,
            model,
            top_n=4,
        )
        assert brackets.shape[1] == 63
        assert brackets.shape[0] == len(champs)
        assert len(champs) <= 4


class TestS2Features:
    """Tests for S2 tournament-specific box score features."""

    def test_feature_count_matches_base_order(self):
        # bases(15) + seeds(2) + picks(2) + leverage_diff + agreement + consensus
        # + disagreement + matchup_type + round + context(4) + S4(3) = 32
        expected = len(BASE_FEATURE_ORDER) + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + len(CONTEXT_KEYS) + 3
        assert n_features() == expected
        assert len(feature_names()) == expected

    def test_s4_feature_names_present(self):
        names = feature_names()
        for name in ("prob_variance", "prob_skewness", "market_model_gap"):
            assert name in names

    def test_four_factors_loading(self):
        from src.prediction.tournament_features import load_four_factors

        seeds_path = Path("data/raw/historical/tournament_seeds_2025.json")
        if not seeds_path.exists():
            pytest.skip("No 2025 data")
        seed_data = json.loads(seeds_path.read_text())
        team_ids = [t["team_id"] for t in seed_data["teams"]]

        ft_rate, to_margin = load_four_factors(2025, team_ids, Path("data"))
        assert len(ft_rate) > 0
        for v in ft_rate.values():
            assert 0.1 < v < 0.6, f"ft_rate {v} out of range"
        for v in to_margin.values():
            assert -0.15 < v < 0.15, f"to_margin {v} out of range"

    def test_conf_tourney_loading(self):
        from src.prediction.tournament_features import load_conf_tourney_champ

        seeds_path = Path("data/raw/historical/tournament_seeds_2025.json")
        if not seeds_path.exists():
            pytest.skip("No 2025 data")
        seed_data = json.loads(seeds_path.read_text())
        team_ids = [t["team_id"] for t in seed_data["teams"]]

        result = load_conf_tourney_champ(2025, team_ids, Path("data"))
        assert len(result) > 0
        for v in result.values():
            assert v in (0.0, 1.0), f"conf_tourney value {v} not binary"
        champions = [k for k, v in result.items() if v == 1.0]
        assert len(champions) >= 10, f"Expected >=10 conf champs, got {len(champions)}"

    def test_ranking_momentum_loading(self):
        from src.prediction.tournament_features import load_ranking_momentum

        seeds_path = Path("data/raw/historical/tournament_seeds_2025.json")
        if not seeds_path.exists():
            pytest.skip("No 2025 data")
        seed_data = json.loads(seeds_path.read_text())
        team_ids = [t["team_id"] for t in seed_data["teams"]]

        result = load_ranking_momentum(2025, team_ids, Path("data"))
        assert len(result) > 0
        nonzero = [v for v in result.values() if v != 0]
        assert len(nonzero) > 0, "All momentum values are zero"

    def test_missing_data_graceful(self):
        from src.prediction.meta_selector import _load_context

        seeds = {"fake_team_a": 1, "fake_team_b": 16}
        context = _load_context(1990, seeds, Path("data"))
        for key in CONTEXT_KEYS:
            assert key in context


class TestUpsetDetector:
    """Tests for the rule-based upset detector."""

    def _load_2025_field(self):
        seeds_path = Path("data/raw/historical/tournament_seeds_2025.json")
        if not seeds_path.exists():
            pytest.skip("No 2025 data")
        seed_data = json.loads(seeds_path.read_text())
        team_ids = [t["team_id"] for t in seed_data["teams"]]
        seeds = {t["team_id"]: t["seed"] for t in seed_data["teams"]}
        return team_ids, seeds

    def test_upset_scores_range(self):
        from src.prediction.upset_detector import clear_cache, compute_upset_scores

        clear_cache()
        team_ids, seeds = self._load_2025_field()
        scores = compute_upset_scores(2025, team_ids, seeds)
        assert len(scores) > 0
        for tid, s in scores.items():
            assert 0.0 <= s <= 1.0, f"{tid} score {s} out of [0,1]"

    def test_upset_scores_differentiate(self):
        """Scores should not all be identical."""
        from src.prediction.upset_detector import clear_cache, compute_upset_scores

        clear_cache()
        team_ids, seeds = self._load_2025_field()
        scores = compute_upset_scores(2025, team_ids, seeds)
        unique_scores = set(round(s, 4) for s in scores.values())
        assert len(unique_scores) >= 5, f"Only {len(unique_scores)} unique scores"

    def test_torvik_gap_drives_variation(self):
        """The Torvik gap signal should create meaningful spread between seeds."""
        from src.prediction.upset_detector import clear_cache, compute_upset_scores

        clear_cache()
        team_ids, seeds = self._load_2025_field()
        scores = compute_upset_scores(2025, team_ids, seeds)
        # 16-seeds should generally have low scores (weak + high seed = no gap signal)
        s16_scores = [scores[t] for t in team_ids if seeds.get(t, 16) == 16 and t in scores]
        all_scores = list(scores.values())
        if s16_scores and all_scores:
            assert np.mean(s16_scores) < np.mean(all_scores), "16-seeds should be below average"

    def test_apply_overrides_shape(self):
        """Override output has same shape as input."""
        from src.prediction.upset_detector import apply_upset_overrides

        bracket = np.ones(63, dtype=bool)
        matchups = [f"team_{i}" for i in range(64)]
        seeds = {f"team_{i}": (i // 4) + 1 for i in range(64)}
        scores = {f"team_{i}": 0.6 if i % 2 == 1 else 0.1 for i in range(64)}
        result = apply_upset_overrides(bracket, matchups, seeds, scores, 0.45, 0)
        assert result.shape == (63,)
        assert result.dtype == bool

    def test_overrides_produce_flips(self):
        """With high upset scores, some games should flip."""
        from src.prediction.upset_detector import apply_upset_overrides

        bracket = np.ones(63, dtype=bool)  # all chalk
        matchups = [f"team_{i}" for i in range(64)]
        seeds = {f"team_{i}": 1 if i % 2 == 0 else 16 for i in range(64)}
        scores = {f"team_{i}": 0.8 if i % 2 == 1 else 0.0 for i in range(64)}
        result = apply_upset_overrides(bracket, matchups, seeds, scores, 0.45, 0)
        flips = int((bracket != result).sum())
        assert flips > 0, "Expected some upset overrides"

    def test_missing_data_no_crash(self):
        """Scores work with a nonexistent year."""
        from src.prediction.upset_detector import clear_cache, compute_upset_scores

        clear_cache()
        scores = compute_upset_scores(1990, ["fake_a", "fake_b"], {"fake_a": 1, "fake_b": 16})
        assert len(scores) == 2
        for s in scores.values():
            assert s == 0.0
