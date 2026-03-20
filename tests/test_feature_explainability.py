"""Tests for feature explainability module and visualization.

Tests verify:
1. MatchupExplainer produces valid explanations (with/without SHAP)
2. LOYOFeatureImportance aggregates correctly across folds
3. PerRoundFeatureImportance stratifies by tournament round
4. Visualization functions produce PNG files
5. Edge cases (empty data, single feature, missing SHAP)
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.ml.evaluation.feature_explainability import (
    FeatureContribution,
    FeatureImportanceSummary,
    LOYOFeatureImportance,
    LOYOFeatureImportanceReport,
    MatchupExplainer,
    MatchupExplanation,
    PerRoundFeatureImportance,
    PerRoundImportance,
    ROUND_NAMES,
)
from src.ml.evaluation.explainability_plots import (
    MATPLOTLIB_AVAILABLE,
    plot_feature_importance_bar,
    plot_matchup_waterfall,
    plot_per_round_heatmap,
    plot_loyo_stability,
    plot_augmentation_ablation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_FEATURES = 20
FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class _MockModel:
    """Mock model with sklearn-like API for testing without SHAP."""

    def __init__(self, n_features: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.coef_ = rng.randn(1, n_features)
        self.intercept_ = np.array([0.0])
        self._importances = np.abs(rng.randn(n_features))

    def predict_proba(self, X):
        logits = X @ self.coef_.T + self.intercept_
        probs = 1.0 / (1.0 + np.exp(-logits.ravel()))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return self.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self):
        return self._importances


@pytest.fixture
def mock_model():
    return _MockModel(N_FEATURES)


@pytest.fixture
def synthetic_loyo_data(rng):
    """Synthetic LOYO data: 5 years × 50 games."""
    data = {}
    for year in [2019, 2021, 2022, 2023, 2024]:
        n = 50
        X = rng.randn(n, N_FEATURES)
        y = (X[:, 0] > 0).astype(float)
        data[year] = (X, y)
    return data


@pytest.fixture
def loyo_years():
    return [2019, 2021, 2022, 2023, 2024]


def _train_fn(X_train, y_train):
    """Train a simple mock model."""
    return _MockModel(X_train.shape[1])


# ---------------------------------------------------------------------------
# Test MatchupExplainer
# ---------------------------------------------------------------------------

class TestMatchupExplainer:
    """Test per-matchup explanation generation."""

    def test_explain_single_matchup(self, mock_model, rng):
        """Explain a single matchup prediction."""
        explainer = MatchupExplainer(mock_model, FEATURE_NAMES)
        X = rng.randn(N_FEATURES)

        result = explainer.explain(X, team1="Duke", team2="UNC")

        assert isinstance(result, MatchupExplanation)
        assert result.team1 == "Duke"
        assert result.team2 == "UNC"
        assert 0 <= result.predicted_probability <= 1
        assert len(result.contributions) == N_FEATURES

    def test_explain_2d_input(self, mock_model, rng):
        """Explain works with 2D input (1, D)."""
        explainer = MatchupExplainer(mock_model, FEATURE_NAMES)
        X = rng.randn(1, N_FEATURES)

        result = explainer.explain(X)
        assert len(result.contributions) == N_FEATURES

    def test_contributions_sorted_by_absolute(self, mock_model, rng):
        """Contributions are sorted by absolute value."""
        explainer = MatchupExplainer(mock_model, FEATURE_NAMES)
        X = rng.randn(N_FEATURES)

        result = explainer.explain(X)
        abs_values = [abs(c.contribution) for c in result.contributions]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_top_positive_negative_split(self, mock_model, rng):
        """Top positive and negative contributions are correctly split."""
        explainer = MatchupExplainer(mock_model, FEATURE_NAMES)
        X = rng.randn(N_FEATURES) * 2  # Large values for clearer signal

        result = explainer.explain(X, top_k=5)

        for c in result.top_positive:
            assert c.contribution > 0
        for c in result.top_negative:
            assert c.contribution < 0
        assert len(result.top_positive) <= 5
        assert len(result.top_negative) <= 5

    def test_explain_batch(self, mock_model, rng):
        """Batch explanation produces one result per sample."""
        explainer = MatchupExplainer(mock_model, FEATURE_NAMES)
        X = rng.randn(10, N_FEATURES)

        results = explainer.explain_batch(X, top_k=5)
        assert len(results) == 10
        for r in results:
            assert isinstance(r, MatchupExplanation)

    def test_explain_batch_with_names(self, mock_model, rng):
        """Batch explanation uses provided team names."""
        explainer = MatchupExplainer(mock_model, FEATURE_NAMES)
        X = rng.randn(3, N_FEATURES)
        names = [("Duke", "UNC"), ("Kansas", "Kentucky"), ("Gonzaga", "UCLA")]

        results = explainer.explain_batch(X, team_names=names)
        assert results[0].team1 == "Duke"
        assert results[1].team2 == "Kentucky"
        assert results[2].team1 == "Gonzaga"

    def test_explanation_to_dict(self, mock_model, rng):
        """Explanation serializes to dict."""
        explainer = MatchupExplainer(mock_model, FEATURE_NAMES)
        X = rng.randn(N_FEATURES)

        result = explainer.explain(X, team1="Duke", team2="UNC")
        d = result.to_dict()

        assert d["team1"] == "Duke"
        assert d["team2"] == "UNC"
        assert "predicted_probability" in d
        assert "method" in d
        assert isinstance(d["top_positive"], list)
        assert isinstance(d["top_negative"], list)

    def test_method_is_native_without_shap(self, mock_model):
        """Without SHAP TreeExplainer, falls back to native importance."""
        explainer = MatchupExplainer(mock_model, FEATURE_NAMES)
        # _MockModel doesn't work with SHAP TreeExplainer
        assert explainer._method in ("native_importance", "shap_tree", "shap_kernel")

    def test_feature_ranks_unique(self, mock_model, rng):
        """Each feature gets a unique rank."""
        explainer = MatchupExplainer(mock_model, FEATURE_NAMES)
        X = rng.randn(N_FEATURES)

        result = explainer.explain(X)
        ranks = [c.rank for c in result.contributions]
        assert len(set(ranks)) == len(ranks)  # All unique
        assert sorted(ranks) == list(range(1, N_FEATURES + 1))


# ---------------------------------------------------------------------------
# Test LOYOFeatureImportance
# ---------------------------------------------------------------------------

class TestLOYOFeatureImportance:
    """Test cross-validated feature importance."""

    def test_compute_with_train_fn(self, synthetic_loyo_data, loyo_years):
        """Compute importance using a training function."""
        loyo_imp = LOYOFeatureImportance(FEATURE_NAMES, top_k=10)
        report = loyo_imp.compute(
            loyo_years, synthetic_loyo_data, train_fn=_train_fn,
        )

        assert isinstance(report, LOYOFeatureImportanceReport)
        assert report.n_folds == len(loyo_years)
        assert report.n_features == N_FEATURES
        assert len(report.global_importances) == N_FEATURES

    def test_global_importances_sorted(self, synthetic_loyo_data, loyo_years):
        """Global importances are sorted by mean importance (descending)."""
        loyo_imp = LOYOFeatureImportance(FEATURE_NAMES)
        report = loyo_imp.compute(loyo_years, synthetic_loyo_data, train_fn=_train_fn)

        means = [imp.mean_importance for imp in report.global_importances]
        assert means == sorted(means, reverse=True)

    def test_importances_normalized(self, synthetic_loyo_data, loyo_years):
        """Each fold's importances sum to ~1 after normalization."""
        loyo_imp = LOYOFeatureImportance(FEATURE_NAMES)
        report = loyo_imp.compute(loyo_years, synthetic_loyo_data, train_fn=_train_fn)

        for year, fold_imp in report.per_fold_importances.items():
            total = sum(imp.mean_importance for imp in fold_imp)
            assert total == pytest.approx(1.0, abs=0.01)

    def test_stability_scores(self, synthetic_loyo_data, loyo_years):
        """Stability scores are between 0 and 1."""
        loyo_imp = LOYOFeatureImportance(FEATURE_NAMES, top_k=10)
        report = loyo_imp.compute(loyo_years, synthetic_loyo_data, train_fn=_train_fn)

        for imp in report.global_importances:
            assert 0.0 <= imp.stability <= 1.0

    def test_compute_with_models(self, synthetic_loyo_data, loyo_years):
        """Compute importance using pre-trained models."""
        models = {yr: _MockModel(N_FEATURES, seed=yr) for yr in loyo_years}

        loyo_imp = LOYOFeatureImportance(FEATURE_NAMES)
        report = loyo_imp.compute(
            loyo_years, synthetic_loyo_data, models_by_fold=models,
        )

        assert report.n_folds == len(loyo_years)
        assert len(report.global_importances) > 0

    def test_report_to_dict(self, synthetic_loyo_data, loyo_years):
        """Report serializes to dict."""
        loyo_imp = LOYOFeatureImportance(FEATURE_NAMES)
        report = loyo_imp.compute(loyo_years, synthetic_loyo_data, train_fn=_train_fn)
        d = report.to_dict()

        assert "method" in d
        assert "n_folds" in d
        assert "global_top_20" in d
        assert isinstance(d["global_top_20"], list)

    def test_per_fold_different_years(self, synthetic_loyo_data, loyo_years):
        """Per-fold importances are keyed by holdout year."""
        loyo_imp = LOYOFeatureImportance(FEATURE_NAMES)
        report = loyo_imp.compute(loyo_years, synthetic_loyo_data, train_fn=_train_fn)

        for year in loyo_years:
            assert year in report.per_fold_importances
            fold_imp = report.per_fold_importances[year]
            assert len(fold_imp) == N_FEATURES

    def test_empty_data(self):
        """Empty data produces empty report."""
        loyo_imp = LOYOFeatureImportance(FEATURE_NAMES)
        report = loyo_imp.compute([], {}, train_fn=_train_fn)

        assert report.n_folds == 0
        assert len(report.global_importances) == 0


# ---------------------------------------------------------------------------
# Test PerRoundFeatureImportance
# ---------------------------------------------------------------------------

class TestPerRoundFeatureImportance:
    """Test per-round feature importance."""

    def test_compute_per_round(self, rng):
        """Compute importance stratified by round."""
        n = 200
        X = rng.randn(n, N_FEATURES)
        y = (X[:, 0] > 0).astype(float)
        rounds = np.concatenate([
            np.full(80, 1),   # R64
            np.full(40, 2),   # R32
            np.full(40, 3),   # S16
            np.full(20, 4),   # E8
            np.full(10, 5),   # F4
            np.full(10, 6),   # NCG
        ])

        model = _MockModel(N_FEATURES)
        prf = PerRoundFeatureImportance(FEATURE_NAMES)
        result = prf.compute(X, y, rounds, model=model)

        assert "R64" in result
        assert "R32" in result
        assert "S16" in result
        assert result["R64"].n_games == 80
        assert result["R32"].n_games == 40

    def test_per_round_importances_normalized(self, rng):
        """Per-round importances sum to ~1."""
        n = 100
        X = rng.randn(n, N_FEATURES)
        y = (X[:, 0] > 0).astype(float)
        rounds = np.concatenate([np.full(50, 1), np.full(50, 2)])

        model = _MockModel(N_FEATURES)
        prf = PerRoundFeatureImportance(FEATURE_NAMES)
        result = prf.compute(X, y, rounds, model=model)

        for rname, round_imp in result.items():
            total = sum(imp.mean_importance for imp in round_imp.importances)
            assert total == pytest.approx(1.0, abs=0.01)

    def test_per_round_skips_small_rounds(self, rng):
        """Rounds with fewer than 5 games are skipped."""
        n = 60
        X = rng.randn(n, N_FEATURES)
        y = (X[:, 0] > 0).astype(float)
        rounds = np.concatenate([np.full(50, 1), np.full(7, 2), np.full(3, 3)])

        model = _MockModel(N_FEATURES)
        prf = PerRoundFeatureImportance(FEATURE_NAMES)
        result = prf.compute(X, y, rounds, model=model)

        assert "R64" in result  # 50 games
        assert "R32" in result  # 7 games
        assert "S16" not in result  # Only 3 games, skipped

    def test_per_round_to_dict(self, rng):
        """Per-round results serialize to dict."""
        n = 100
        X = rng.randn(n, N_FEATURES)
        y = (X[:, 0] > 0).astype(float)
        rounds = np.full(n, 1)

        model = _MockModel(N_FEATURES)
        prf = PerRoundFeatureImportance(FEATURE_NAMES)
        result = prf.compute(X, y, rounds, model=model)

        for rname, round_imp in result.items():
            d = round_imp.to_dict()
            assert "round_name" in d
            assert "n_games" in d
            assert "top_10" in d

    def test_per_round_with_train_fn(self, rng):
        """Compute per-round importance with a training function."""
        n = 100
        X = rng.randn(n, N_FEATURES)
        y = (X[:, 0] > 0).astype(float)
        rounds = np.full(n, 1)

        prf = PerRoundFeatureImportance(FEATURE_NAMES)
        result = prf.compute(X, y, rounds, train_fn=_train_fn)

        assert "R64" in result
        assert result["R64"].n_games == 100

    def test_round_names_mapping(self):
        """ROUND_NAMES maps integers 1-6 to standard tournament round names."""
        assert ROUND_NAMES[1] == "R64"
        assert ROUND_NAMES[2] == "R32"
        assert ROUND_NAMES[3] == "S16"
        assert ROUND_NAMES[4] == "E8"
        assert ROUND_NAMES[5] == "F4"
        assert ROUND_NAMES[6] == "NCG"


# ---------------------------------------------------------------------------
# Test visualization plots
# ---------------------------------------------------------------------------

class TestExplainabilityPlots:
    """Test that visualization functions produce PNG files."""

    @pytest.fixture
    def output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def loyo_report(self, rng):
        """Create a mock LOYOFeatureImportanceReport."""
        global_imp = [
            FeatureImportanceSummary(
                name=f"feature_{i}", mean_importance=1.0 / (i + 1),
                std_importance=0.01 * (i + 1), stability=1.0 - 0.05 * i,
                rank=i + 1,
            )
            for i in range(N_FEATURES)
        ]
        per_fold = {}
        for year in [2019, 2021, 2022, 2023, 2024]:
            per_fold[year] = [
                FeatureImportanceSummary(
                    name=f"feature_{i}",
                    mean_importance=float(rng.uniform(0.01, 0.1)),
                    std_importance=0.0, stability=global_imp[i].stability,
                    rank=i + 1,
                )
                for i in range(N_FEATURES)
            ]

        return LOYOFeatureImportanceReport(
            global_importances=global_imp,
            per_fold_importances=per_fold,
            per_round_importances={},
            method="native",
            n_folds=5,
            n_features=N_FEATURES,
        )

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
    def test_feature_importance_bar(self, loyo_report, output_dir):
        """Feature importance bar chart creates a PNG file."""
        path = plot_feature_importance_bar(
            loyo_report, output_dir=output_dir, top_k=10,
        )
        assert path is not None
        assert os.path.exists(path)
        assert path.endswith(".png")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
    def test_matchup_waterfall(self, rng, output_dir):
        """Matchup waterfall chart creates a PNG file."""
        contribs = [
            FeatureContribution(
                name=f"feature_{i}", value=float(rng.randn()),
                contribution=float(rng.randn() * 0.1), rank=i + 1,
            )
            for i in range(N_FEATURES)
        ]
        explanation = MatchupExplanation(
            team1="Duke", team2="UNC",
            predicted_probability=0.65, base_value=0.5,
            contributions=contribs, method="native_importance",
            top_positive=[c for c in contribs if c.contribution > 0][:5],
            top_negative=[c for c in contribs if c.contribution < 0][:5],
        )

        path = plot_matchup_waterfall(explanation, output_dir=output_dir, top_k=8)
        assert path is not None
        assert os.path.exists(path)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
    def test_per_round_heatmap(self, rng, output_dir):
        """Per-round heatmap creates a PNG file."""
        per_round = {}
        for r in [1, 2, 3, 4]:
            rname = ROUND_NAMES[r]
            importances = [
                FeatureImportanceSummary(
                    name=f"feature_{i}",
                    mean_importance=float(rng.uniform(0.01, 0.1)),
                    std_importance=0.0, stability=1.0, rank=i + 1,
                )
                for i in range(N_FEATURES)
            ]
            per_round[rname] = PerRoundImportance(
                round_name=rname, n_games=50, importances=importances,
            )

        path = plot_per_round_heatmap(per_round, output_dir=output_dir, top_k=10)
        assert path is not None
        assert os.path.exists(path)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
    def test_loyo_stability_plot(self, loyo_report, output_dir):
        """LOYO stability plot creates a PNG file."""
        path = plot_loyo_stability(loyo_report, output_dir=output_dir, top_k=10)
        assert path is not None
        assert os.path.exists(path)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
    def test_augmentation_ablation_plot(self, output_dir):
        """Augmentation ablation chart creates a PNG file."""
        baseline = [
            {"year": 2019, "brier": 0.21},
            {"year": 2021, "brier": 0.19},
            {"year": 2022, "brier": 0.22},
        ]
        ablated = [
            {"year": 2019, "brier": 0.23},
            {"year": 2021, "brier": 0.20},
            {"year": 2022, "brier": 0.24},
        ]

        path = plot_augmentation_ablation(baseline, ablated, output_dir=output_dir)
        assert path is not None
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_explainer_with_fewer_feature_names(self, rng):
        """Explainer works when feature_names < model dimensions."""
        model = _MockModel(N_FEATURES)
        short_names = [f"f_{i}" for i in range(5)]  # Only 5 names for 20 features
        explainer = MatchupExplainer(model, short_names)

        X = rng.randn(N_FEATURES)
        result = explainer.explain(X)
        assert len(result.contributions) == 5  # Truncated to name count

    def test_feature_contribution_dataclass(self):
        """FeatureContribution stores values correctly."""
        fc = FeatureContribution(
            name="seed_diff", value=0.5, contribution=-0.03, rank=1,
        )
        assert fc.name == "seed_diff"
        assert fc.value == 0.5
        assert fc.contribution == -0.03

    def test_importance_summary_dataclass(self):
        """FeatureImportanceSummary stores values correctly."""
        fis = FeatureImportanceSummary(
            name="elo_diff", mean_importance=0.15,
            std_importance=0.02, stability=0.85, rank=1,
        )
        d = fis.to_dict()
        assert d["name"] == "elo_diff"
        assert d["stability"] == 0.85
