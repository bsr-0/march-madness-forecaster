"""Tests for feature-group ablation flags."""

import numpy as np
from types import SimpleNamespace
from src.data.features.ablation import apply_ablation


class TestApplyAblation:
    def test_no_ablation_leaves_vector_unchanged(self):
        config = SimpleNamespace(ablate_elo=False, ablate_conf_tourney=False,
                                 ablate_late_season=False, ablate_market=False)
        v = np.ones(54)
        apply_ablation(v, config)
        assert np.all(v == 1.0)

    def test_ablate_elo_zeros_index_29(self):
        config = SimpleNamespace(ablate_elo=True)
        v = np.ones(54)
        apply_ablation(v, config)
        assert v[29] == 0.0
        assert v[28] == 1.0
        assert v[30] == 1.0

    def test_ablate_conf_tourney(self):
        config = SimpleNamespace(ablate_conf_tourney=True)
        v = np.ones(54)
        apply_ablation(v, config)
        assert v[46] == 0.0 and v[47] == 0.0 and v[48] == 0.0
        assert v[45] == 1.0 and v[49] == 1.0

    def test_ablate_late_season(self):
        config = SimpleNamespace(ablate_late_season=True)
        v = np.ones(54)
        apply_ablation(v, config)
        assert v[49] == 0.0 and v[50] == 0.0 and v[51] == 0.0

    def test_ablate_market(self):
        config = SimpleNamespace(ablate_market=True)
        v = np.ones(54)
        apply_ablation(v, config)
        assert v[52] == 0.0 and v[53] == 0.0

    def test_multiple_ablations(self):
        config = SimpleNamespace(ablate_elo=True, ablate_market=True)
        v = np.ones(54)
        apply_ablation(v, config)
        assert v[29] == 0.0
        assert v[52] == 0.0 and v[53] == 0.0
        assert v[0] == 1.0

    def test_missing_flags_default_to_no_ablation(self):
        config = SimpleNamespace()
        v = np.ones(54)
        apply_ablation(v, config)
        assert np.all(v == 1.0)
