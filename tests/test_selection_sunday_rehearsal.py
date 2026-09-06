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

    The RULE is tested behaviourally in tests/test_picks_export.js
    (``pickDefaultSeason``), which is worth more than the source grep that used
    to live here: a regex proves a line exists and nothing about what it does.
    What remains here is the wiring -- that init() actually calls the rule.
    """

    def test_init_uses_the_season_picker(self):
        src = (REPO / "docs" / "app.js").read_text()
        assert "pickDefaultSeason(idx.seasons)" in src, (
            "init() must choose its season through pickDefaultSeason()"
        )

    def test_the_newest_listed_season_is_not_used_directly(self):
        """max(year) would open on 2027's empty state for most of the year."""
        src = (REPO / "docs" / "app.js").read_text()
        assert "Math.max(...idx.seasons" not in src


class TestTheUserFacingTextIsForUsers:
    """Strings a stranger reads on a phone on Selection Sunday.

    The council's Outsider read the page cold and reported a dozen terms they
    would have had to search for, plus a source-file path rendered to people who
    do not have the source. These are cheap to reintroduce and invisible in any
    behavioural test, so they are asserted directly.
    """

    @staticmethod
    def _user_text() -> str:
        """Rendered strings only -- comments are allowed to be technical."""
        import re

        src = (REPO / "docs" / "app.js").read_text()
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)          # block comments
        src = re.sub(r"^\s*//.*$", "", src, flags=re.M)           # line comments
        return src + (REPO / "docs" / "index.html").read_text()

    def test_no_repo_paths_are_rendered(self):
        """`app.js` shipped "Bracket shapes ... from src/product/selection.py"."""
        assert "src/product/selection.py" not in self._user_text()

    @pytest.mark.parametrize(
        "term",
        ["LOYO", "Brier", "RMSE", "Student-t", "kNN", "LightGBM", "walk-forward"],
    )
    def test_no_unexplained_jargon(self, term):
        assert term not in self._user_text(), (
            f"{term!r} is shown to users and defined nowhere on the page"
        )

    def test_the_two_final_four_panels_ask_different_questions(self):
        """One label needed a correction printed underneath it, which is a label
        admitting it failed: "Depth says how far DOWN you reach; this says how
        much of the top you keep"."""
        html = (REPO / "docs" / "index.html").read_text()
        assert "How deep does your Final Four reach" not in html
        assert "The biggest upset in your Final Four" in html

    def test_the_filters_do_not_sit_on_top_of_the_answer(self):
        """The board was the seventh thing on the page, under ~3,800px of controls."""
        html = (REPO / "docs" / "index.html").read_text()
        assert '<details class="tune"' in html, "filter panels must be collapsed by default"
        assert "<details" in html.split('id="champions"')[0], "the panels must be INSIDE the details"


class TestTheDisclosureCannotVanishQuietly:
    """A mandatory disclosure whose absence is survivable is not mandatory.

    The browser reads ``p1_assumption || ''`` and renders nothing when it is
    missing, and a missing ``p1_trials`` silently reverts the standard error to
    the scalar computed at a fixed p=0.05. Both revert a fix without failing
    anything, which is how the original defect survived in the first place.
    """

    def _artifact_without(self, tmp_path, *dropped):
        import scripts.build_ui_payload as builder

        donor = REPO / "artifacts" / "candidates" / f"candidates_{DONOR}.json"
        if not donor.exists():
            pytest.skip("no donor artifact on disk")
        art = json.loads(donor.read_text())
        for key in dropped:
            art["meta"].pop(key, None)

        cand = tmp_path / "candidates"
        cand.mkdir()
        (cand / f"candidates_{DONOR}.json").write_text(json.dumps(art))
        stats = tmp_path / "stats.json"
        stats.write_text((REPO / "docs" / "data" / "team_stats_by_year.json").read_text())
        out = tmp_path / "out"
        out.mkdir()
        return builder, cand, stats, out

    @pytest.mark.parametrize("missing", ["p1_assumption", "p1_trials"])
    def test_a_season_without_it_refuses_to_build(self, tmp_path, monkeypatch, missing):
        builder, cand, stats, out = self._artifact_without(tmp_path, missing)
        monkeypatch.setattr(builder, "CANDIDATES_DIR", cand)
        monkeypatch.setattr(builder, "STATS_PATH", stats)
        monkeypatch.setattr(builder, "OUT_DIR", out)
        with pytest.raises(ValueError, match=missing):
            builder.main()

    def test_every_shipped_season_actually_carries_them(self):
        """The check above is worthless if what is on disk predates it."""
        for path in sorted((REPO / "docs" / "data").glob("season_*.json")):
            payload = json.loads(path.read_text())
            if payload.get("status") != "ready":
                continue
            assert payload.get("p1_assumption"), f"{path.name} ships without the disclosure"
            assert payload["filters"].get("p1_trials"), f"{path.name} ships without p1_trials"

    def test_every_predicate_has_a_calibrated_probability(self):
        """The chips fall back to showing nothing, silently, if one is absent."""
        for path in sorted((REPO / "docs" / "data").glob("season_*.json")):
            payload = json.loads(path.read_text())
            if payload.get("status") != "ready":
                continue
            probs = payload["filters"].get("predicate_probabilities") or {}
            missing = [p["key"] for p in payload["filters"].get("predicates", []) if p["key"] not in probs]
            assert not missing, f"{path.name} has predicates with no probability: {missing}"


class TestNothingFromABracketSurvivesOntoAnEmptySeason:
    """2027 before Selection Sunday is the most-visited state of the year.

    The empty-season path cleared the board and the mode note but not the P(1st)
    disclosure or the "your filter was dropped" message, so both sat under "the
    2027 season hasn't started yet".
    """

    def test_the_empty_path_clears_the_disclosure_and_the_notice(self):
        src = (REPO / "docs" / "app.js").read_text()
        empty_branch = src.split("if (!s || s.status !== 'ready') {")[1].split("empty.hidden = false;")[0]
        assert "p1-note" in empty_branch, "the disclosure is left on screen for an unplayed season"
        assert "dropped-note" in empty_branch, "a stale filter notice is left on screen"


class TestThePrintedPageIsTheBracket:
    def test_print_hides_the_collapsed_filter_stack(self):
        """Tier 3 wrapped the panels in .tune; the print rules still named only
        .panel, so "Adjust this bracket" and its box printed with the bracket."""
        css = (REPO / "docs" / "app.css").read_text()
        block = css.split("@media print")[1]
        for selector in (".tune", ".board-tools", "#dropped-note"):
            assert selector in block, f"{selector} would print alongside the bracket"


class TestCacheBustCannotGoStale:
    """A ``?v=`` a human types can disagree with the file it names.

    index.html already carried a comment recording that stale JS shipped once.
    It happened again during this very session: app.js was edited after its bump
    to v46, so v46 named two different files and a browser holding the first had
    no way to learn about the second. It cost a verification pass -- a fix that
    was correct on disk read as broken in the browser.

    The tokens are content hashes now, so editing an asset changes its token by
    construction. This test is what makes a forgotten stamp fail here rather
    than ship.
    """

    def test_every_asset_token_matches_its_file(self):
        from scripts.stamp_asset_versions import ASSETS, asset_token
        import re as _re

        html = (REPO / "docs" / "index.html").read_text()
        for name in ASSETS:
            found = _re.search(rf"{_re.escape(name)}\?v=([A-Za-z0-9]+)", html)
            assert found, f"{name} has no ?v= token"
            assert found.group(1) == asset_token(name), (
                f"{name}?v={found.group(1)} does not match the file on disk. "
                f"Run: python scripts/stamp_asset_versions.py"
            )

    def test_the_checker_agrees(self):
        from scripts.stamp_asset_versions import stamp

        assert stamp(check=True) == 0

    def test_editing_an_asset_makes_the_stamp_stale(self, tmp_path, monkeypatch):
        """Mutation check: a guard that cannot fail is worse than none."""
        import scripts.stamp_asset_versions as st

        docs = tmp_path / "docs"
        docs.mkdir()
        for name in st.ASSETS:
            (docs / name).write_text((REPO / "docs" / name).read_text())
        (docs / "index.html").write_text((REPO / "docs" / "index.html").read_text())
        monkeypatch.setattr(st, "REPO", tmp_path)
        monkeypatch.setattr(st, "INDEX", docs / "index.html")
        assert st.stamp(check=True) == 0, "copy should start clean"

        (docs / "app.js").write_text((docs / "app.js").read_text() + "\n// edited\n")
        assert st.stamp(check=True) == 1, "an edited asset must fail the check"
