"""Closure / lock test for COUNCIL_LESSONS §2 O19 — pre-2011 ESPN picks.

Decision (from artifacts/o19_pre_2011_picks_audit_2026-04-14.md): keep
pre-2011 archived files for single-year historical fidelity, exclude
from any aggregated calibration window via MIN_PICKS_CALIBRATION_YEAR
+ strict_post_2011 kwarg.

This test fails if any of those invariants drift — e.g., the constant
moves, the strict kwarg silently accepts pre-2011 years, pre-2011
files disappear from disk, the production backtest starts including
pre-2011, or SEED_PICK_RATES calibration expands backwards in time.

DO NOT relax a bound to make this pass. Any widening of the calibration
window needs a new dated audit artifact and a fresh closure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PICKS_DIR = REPO_ROOT / "data" / "raw" / "historical_public_picks"


# RETIRED 2026-08-19: test_audit_artifact_present.
#
# It asserted the existence of artifacts/o19_pre_2011_picks_audit_2026-04-14.md,
# which commit 6393ef0 ("Consolidate historical docs into FINDINGS.md")
# deleted deliberately along with ~65 other historical documents. The file
# was not lost, so restoring it purely to satisfy an assertion would be
# cargo-culting a green suite.
#
# Nothing else in this closure depended on it: every remaining test below
# checks live code and live data (the MIN_PICKS_CALIBRATION_YEAR constant,
# the strict_post_2011 kwarg's behaviour, the pre-2011 files on disk, and
# the production backtest's year window). Those are the invariants that
# actually prevent the pre-2011 chalky bias from re-entering calibration.


def test_min_calibration_year_constant_locked() -> None:
    """MIN_PICKS_CALIBRATION_YEAR must be 2011.

    Moving this constant without a fresh audit would silently widen
    the aggregation window and re-introduce the pre-2011 chalky bias
    into opponent-model calibration."""
    from src.data.historical_picks import MIN_PICKS_CALIBRATION_YEAR

    assert MIN_PICKS_CALIBRATION_YEAR == 2011, (
        f"MIN_PICKS_CALIBRATION_YEAR={MIN_PICKS_CALIBRATION_YEAR}; audit "
        f"locks 2011. Re-audit before shifting the boundary."
    )


def test_strict_post_2011_rejects_pre_2011_years() -> None:
    """`load_historical_public_picks(..., strict_post_2011=True)` must
    raise on 2008/2009/2010 with a message pointing back to O19.

    This is the runtime enforcement that aggregators MUST use — if it
    silently accepts pre-2011, the boundary is advisory-only."""
    from src.data.historical_picks import load_historical_public_picks

    bracket_teams = {"duke": 1}
    for year in (2008, 2009, 2010):
        with pytest.raises(ValueError, match="MIN_PICKS_CALIBRATION_YEAR"):
            load_historical_public_picks(
                year,
                bracket_teams,
                strict_post_2011=True,
            )


def test_strict_post_2011_accepts_2011_and_later() -> None:
    """Boundary is >=, not >. 2011 itself must be accepted under strict
    mode (field-expansion year, analytics-culture onset). This also
    implicitly checks that the constant is 2011, not 2010 or 2012."""
    from src.data.historical_picks import load_historical_public_picks

    bracket_teams = {"duke": 1}
    # 2011 should NOT raise on the strict gate — but the file loader
    # will still run and may raise FileNotFoundError on the dummy
    # bracket_teams. That's acceptable — we just want the O19 gate
    # not to fire at this year.
    try:
        load_historical_public_picks(
            2011,
            bracket_teams,
            strict_post_2011=True,
            require_archived=False,
        )
    except ValueError as e:
        assert "MIN_PICKS_CALIBRATION_YEAR" not in str(e), (
            "strict_post_2011 rejected 2011, but 2011 is the first accepted year — boundary is off by one."
        )


def test_single_year_loader_still_works_for_pre_2011() -> None:
    """Single-year simulation (default `strict_post_2011=False`) must
    still accept pre-2011 years. A 2008 pool sim needs the 2008 public's
    actual picks for historical fidelity — the archived files exist,
    they're valid ESPN WPW data, and the loader must consume them."""
    from src.data.historical_picks import load_historical_public_picks

    # Minimal bracket_teams — just enough to invoke the loader path.
    # The archived file has its own team set; we just need the loader
    # not to blow up on the year itself.
    bracket_teams = {"kansas": 1, "duke": 1, "north_carolina": 1}
    for year in (2008, 2009, 2010):
        # Should NOT raise the O19 gate. Other errors (file format etc.)
        # would be a different bug and not what this test guards.
        result = load_historical_public_picks(year, bracket_teams)
        assert isinstance(result, dict)


def test_pre_2011_archives_exist_on_disk() -> None:
    """Archived ESPN WPW files for 2008, 2009, 2010 must stay on disk.

    Deleting them would destroy historical fidelity for single-year
    simulation of those tournaments — the whole reason we kept them
    rather than `exclude`-ing them outright."""
    for year in (2008, 2009, 2010):
        path = PICKS_DIR / f"espn_picks_{year}.json"
        assert path.exists(), (
            f"{path} missing. Pre-2011 ESPN archives are kept on disk "
            f"per O19 (single-year fidelity). If the file was intentionally "
            f"removed, re-issue the O19 audit with the new decision."
        )
        # Sanity: the file must still be a populated picks dict.
        data = json.loads(path.read_text())
        teams = data.get("teams", {})
        assert len(teams) >= 60, f"espn_picks_{year}.json has only {len(teams)} teams; expected 64."


def test_seed_pick_rates_calibration_window_excludes_pre_2011() -> None:
    """`SEED_PICK_RATES` in src/data/seed_pick_model.py is the canonical
    public-picks table. Its module docstring must continue to document
    a 2015-2024 (or later-starting) calibration window — NOT a window
    that includes pre-2011. If someone widens the window backwards,
    this test fires."""
    seed_pick_source = (REPO_ROOT / "src" / "data" / "seed_pick_model.py").read_text()

    # The docstring explicitly mentions the calibration window; the
    # literal strings "2008", "2009", or "2010" appearing as a window
    # boundary would be a red flag. We allow mentions in commentary
    # (e.g. "2008-2026" in a range label) but the specific calibration
    # anchor text should cite 2015-2024 or similar post-2011 window.
    assert "2015" in seed_pick_source, (
        "SEED_PICK_RATES module should document a calibration window "
        "starting >= 2015. If the anchor year changed, update both "
        "the module docstring and this test."
    )
    # Quick guard: the module should not start its calibration window
    # at a pre-2011 year in the anchored-points commentary block.
    assert "2008-2024" not in seed_pick_source, (
        "SEED_PICK_RATES calibration window appears to include pre-2011 years. This violates O19. Re-audit."
    )


def test_production_pool_backtest_starts_at_2011() -> None:
    """`scripts/mc_pool_backtest.py` is the production pool-EV backtest.
    Its BACKTEST_YEARS list must not include pre-2011, because the
    backtest aggregates metrics across years — including 2008-2010
    would re-introduce the chalky regime into a pooled metric."""
    source = (REPO_ROOT / "scripts" / "mc_pool_backtest.py").read_text()
    # Strict check: the BACKTEST_YEARS comprehension range start must
    # be >= 2011. Historic code used `range(2011, 2026)`; if someone
    # widens it to `range(2008, 2026)`, this fires.
    assert "range(2011" in source or "range(2012" in source, (
        "scripts/mc_pool_backtest.py BACKTEST_YEARS range doesn't start "
        "at 2011+. Production pool backtest should honor the O19 "
        "boundary on aggregated calibration."
    )
    # And literal pre-2011 years appearing as BACKTEST_YEARS members
    # would also be a violation.
    assert "BACKTEST_YEARS = [2008" not in source
    assert "BACKTEST_YEARS = [2009" not in source
    assert "BACKTEST_YEARS = [2010" not in source
