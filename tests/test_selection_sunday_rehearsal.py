"""Rehearse the Selection Sunday 2027 path against a season that does not exist.

THE FAILURE THIS EXISTS TO CATCH. Every season currently on the site was built
from data that was already on disk when the code was written. The 2027 path --
a brand new season appearing for the first time, flipping from "not_started" to
"ready" -- has never been executed. It gets exactly one chance, in the ~96 hours
after 2027-03-14, and the people running it will be in a hurry.

So the rehearsal runs the REAL builder against a synthetic 2027, assembled by
relabelling the most recent real season. It asserts the two things that would
be discovered too late: that a new season's payload comes out "ready" with
every key the page reads, and that the page would OPEN on it rather than on
last year's bracket.

Everything is redirected to tmp_path. Nothing here writes into docs/data or
artifacts/, and no synthetic 2027 file is ever left behind.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DONOR = 2026  # the newest season with a real artifact on disk
REHEARSAL = 2027


@pytest.fixture
def rehearsal(tmp_path, monkeypatch):
    """A world where 2027 has just been scraped, built entirely in tmp_path."""
    import scripts.build_ui_payload as builder

    donor_art = REPO / "artifacts" / "candidates" / f"candidates_{DONOR}.json"
    if not donor_art.exists():
        pytest.skip(f"no {DONOR} artifact on disk to build a synthetic season from")

    candidates = tmp_path / "candidates"
    candidates.mkdir()
    art = json.loads(donor_art.read_text())
    art.setdefault("meta", {})["year"] = REHEARSAL
    (candidates / f"candidates_{REHEARSAL}.json").write_text(json.dumps(art))

    stats = json.loads((REPO / "docs" / "data" / "team_stats_by_year.json").read_text())
    stats["stats_by_year"][str(REHEARSAL)] = stats["stats_by_year"][str(DONOR)]
    stats_path = tmp_path / "team_stats_by_year.json"
    stats_path.write_text(json.dumps(stats))

    out = tmp_path / "data"
    out.mkdir()
    monkeypatch.setattr(builder, "CANDIDATES_DIR", candidates)
    monkeypatch.setattr(builder, "STATS_PATH", stats_path)
    monkeypatch.setattr(builder, "OUT_DIR", out)
    monkeypatch.setattr(builder, "REPO", REPO)  # picks archive lookup stays real

    assert builder.main() == 0
    return out


class TestANewSeasonBuildsReady:
    def test_the_2027_payload_is_ready(self, rehearsal):
        """The flip from not_started to ready is the whole event."""
        payload = json.loads((rehearsal / f"season_{REHEARSAL}.json").read_text())
        assert payload["status"] == "ready", payload.get("detail")

    def test_it_carries_every_key_the_page_reads(self, rehearsal):
        payload = json.loads((rehearsal / f"season_{REHEARSAL}.json").read_text())
        for key in ("teams", "first_round", "strategies", "filters", "pool_optimized", "z", "raw"):
            assert key in payload, f"season payload is missing {key!r}"
        assert len(payload["teams"]) >= 64
        assert payload["strategies"], "no strategies to show"

    def test_the_disclosure_travels_with_the_numbers(self, rehearsal):
        """Mandatory under product.v3, and it reached the browser not at all."""
        payload = json.loads((rehearsal / f"season_{REHEARSAL}.json").read_text())
        assert payload.get("p1_assumption"), "P(1st) disclosure missing from the payload"
        assert "30" in str(payload.get("p1_pool_size"))

    def test_the_index_lists_it_as_ready(self, rehearsal):
        index = json.loads((rehearsal / "seasons.json").read_text())
        entry = next(s for s in index["seasons"] if s["year"] == REHEARSAL)
        assert entry["status"] == "ready"

    def test_an_unbuilt_season_is_not_called_not_started(self, rehearsal):
        """A season that was played but cannot be built must not claim it hasn't happened."""
        index = json.loads((rehearsal / "seasons.json").read_text())
        for season in index["seasons"]:
            if season["year"] < REHEARSAL and season["status"] != "ready":
                payload = json.loads((rehearsal / f"season_{season['year']}.json").read_text())
                assert payload["status"] == "unavailable", (
                    f"{season['year']} was played but is labelled {payload['status']!r}"
                )


class TestThePageWouldOpenOnIt:
    """The launch-day failure: the site opening on last season's bracket.

    ``state.year`` was a hardcoded 2026 with nothing to advance it, so on
    Selection Sunday 2027 a visitor would have landed on the 2026 bracket with
    2027 greyed out beside it. Asserted against the source because there is no
    JS test harness in this repo and this is one line whose absence is fatal.
    """

    def test_init_selects_the_newest_ready_season(self):
        src = (REPO / "docs" / "app.js").read_text()
        assert re.search(r"filter\(\s*s\s*=>\s*s\.status\s*===\s*'ready'\s*\)", src), (
            "init() must choose the newest season whose status is 'ready'"
        )
        assert "Math.max(...ready)" in src

    def test_the_newest_listed_season_is_not_used_directly(self):
        """max(year) would open on 2027's empty state for most of the year."""
        src = (REPO / "docs" / "app.js").read_text()
        assert "Math.max(...idx.seasons" not in src
