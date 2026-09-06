"""The inference path must refuse an unusable stats payload.

FINDINGS.md 6c was a train/serve skew that produced ~0.5 for every matchup and
surfaced nowhere: a zero feature differential is a perfectly plausible value, so
nothing downstream could tell a confident prediction from an empty one. The fix
added ``validate_stats_payload``, but wired it into ``train_noseed_model`` only
-- not into the inference path where the skew occurred.

This file covers that path. The reachable case is a new season:
``noseed_model._load_team_stats`` probes two directories for
``torvik_{year}.json`` and used to return ``{}`` if neither had it, which is the
normal state of a season whose pre-tournament snapshot has not been scraped yet.
Every team then fell to per-key defaults and the bracket was built from nothing.

The guard is asserted *at the loader*, not at any one of its callers, and the
callers are enumerated below. Placing it on a single caller is the mistake 6c-ii
records: there are six routes into this loader, and the first fix guarded the
one that happened to be in the traceback.
"""

from __future__ import annotations

import pytest

from scripts.mc_pool_backtest import _load_team_stats
from src.prediction.noseed_model import FeatureSkewError

# Far enough back that no snapshot will ever exist, so this stays deterministic
# regardless of which seasons are scraped into data/raw.
UNSCRAPED_SEASON = 1901

# Present in both data/raw and data/raw/historical, and in TRAIN_YEARS.
KNOWN_GOOD_SEASON = 2025


class TestMissingSnapshotIsLoud:
    def test_unscraped_season_raises_rather_than_returning_empty(self):
        with pytest.raises(FileNotFoundError) as exc:
            _load_team_stats(UNSCRAPED_SEASON)
        assert str(UNSCRAPED_SEASON) in str(exc.value)

    def test_error_names_the_file_and_the_remedy(self):
        """A March-2027 operator should not have to read source to fix this."""
        with pytest.raises(FileNotFoundError) as exc:
            _load_team_stats(UNSCRAPED_SEASON)
        message = str(exc.value)
        assert f"torvik_{UNSCRAPED_SEASON}.json" in message
        assert "rescrape_pretournament_torvik.py" in message

    def test_the_loader_itself_raises_not_just_the_backtest_wrapper(self):
        """The guard must sit at the loader, or callers can bypass it.

        ``cli/pool_cmds.py`` imports this function directly for both the
        ``noseed`` and ``blend`` pool modes, so a guard living only in
        ``mc_pool_backtest`` would leave the production CLI on the silent path.
        """
        from src.prediction.noseed_model import _load_team_stats as raw_loader

        with pytest.raises(FileNotFoundError):
            raw_loader(UNSCRAPED_SEASON)

    @pytest.mark.parametrize(
        "module_path",
        [
            "scripts.mc_pool_backtest",
            "src.optimization.recency_hparam_fitter",
        ],
    )
    def test_re_exporting_callers_inherit_the_guard(self, module_path):
        """Modules that import the name under their own namespace get it too."""
        import importlib

        loader = getattr(importlib.import_module(module_path), "_load_team_stats")
        with pytest.raises(FileNotFoundError):
            loader(UNSCRAPED_SEASON)


class TestSkewedPayloadIsLoud:
    def test_payload_missing_the_model_features_is_rejected(self, tmp_path, monkeypatch):
        """The original 6c shape: teams present, required feature keys absent.

        Written as a real snapshot on disk rather than a monkeypatched loader,
        because the loader *is* the unit under test now -- stubbing it out would
        test nothing but the stub.
        """
        import json

        import src.prediction.noseed_model as nm

        snapshot = {
            "data_type": "pre_tournament",
            # Four-factors keys only: the exact 8-of-12 payload of 6c.
            "teams": [
                {
                    "team_id": f"team-{i}",
                    "effective_fg_pct": 0.52,
                    "turnover_rate": 0.18,
                    "offensive_reb_rate": 0.30,
                    "free_throw_rate": 0.32,
                }
                for i in range(64)
            ],
        }
        (tmp_path / f"torvik_{UNSCRAPED_SEASON}.json").write_text(json.dumps(snapshot))
        monkeypatch.setattr(nm, "HIST_DIR", tmp_path)
        monkeypatch.setattr(nm, "DATA_DIR", tmp_path)

        with pytest.raises(FeatureSkewError) as exc:
            _load_team_stats(UNSCRAPED_SEASON)
        # Names the absent keys, so the operator does not have to diff payloads.
        assert "barthag" in str(exc.value)


class TestKnownGoodSeasonStillLoads:
    def test_real_season_passes_the_guard(self):
        stats = _load_team_stats(KNOWN_GOOD_SEASON)
        assert stats, "2025 snapshot should load"
        assert len(stats) > 300, "expected a full-season team payload"
