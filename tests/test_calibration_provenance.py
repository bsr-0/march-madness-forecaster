from types import SimpleNamespace

from src.pipeline.stages.calibration import _fit_calibration


class _StubGame:
    def __init__(self, team1_id, team2_id, outcome):
        self.team1_id = team1_id
        self.team2_id = team2_id
        self._outcome = outcome
        self.game_id = f"{team1_id}_{team2_id}"
        self.game_date = "2026-02-01"


def test_calibration_stats_include_provenance_and_feature_manifest_hash():
    games = [_StubGame("a", "b", 1) for _ in range(20)]

    pipeline = SimpleNamespace()
    pipeline.config = SimpleNamespace(
        year=2026,
        pre_calibration_clip_lo=0.001,
        pre_calibration_clip_hi=0.999,
        min_calibration_samples_hard=5,
        min_calibration_samples=5,
        calibration_method="none",
        probability_profile="production",
        enable_multi_year_calibration=False,
        multi_year_games_dir=None,
        strict_leakage_mode=True,
        training_years=[2016, 2017],
        dev_years=[2016, 2017],
        holdout_years=[2025],
        resolve_calibration_years=lambda: [2025],
    )
    pipeline.feature_engineer = SimpleNamespace(team_features={"a": 1, "b": 1})
    pipeline._feature_manifest = {"manifest_hash": "abc123"}
    pipeline._get_validation_era_games = lambda game_flows: games
    pipeline._unique_games = lambda game_flows: games
    pipeline._game_sort_key = lambda date: 0
    pipeline._is_tournament_game = lambda date: False
    pipeline._raw_fusion_probability = lambda t1, t2: 0.7
    pipeline._game_outcome = lambda g: g._outcome
    pipeline.calibration_pipeline = None

    result = _fit_calibration(pipeline, {"a": games})

    assert result["feature_manifest_hash"] == "abc123"
    assert result["provenance"]["artifact_kind"] == "calibration_report"
    assert result["provenance"]["calibration_years"] == [2025]
