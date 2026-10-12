"""Archived public-picks timestamps must be re-derivable from committed captures."""

import json
from pathlib import Path

import pytest

from scripts.experiments.verify_picks_capture_evidence import PICKS_DIR, verify_archive

EVIDENCED = [f"espn_picks_{y}.json" for y in (2012, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025)]


@pytest.mark.parametrize("name", EVIDENCED)
def test_recorded_capture_evidence_verifies(name):
    assert verify_archive(PICKS_DIR / name) == []


def _tampered(tmp_path: Path, name: str, edit) -> Path:
    data = json.loads((PICKS_DIR / name).read_text())
    edit(data)
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_a_single_changed_share_is_caught(tmp_path):
    def edit(data):
        data["teams"]["alabama"]["S16"] = round(data["teams"]["alabama"]["S16"] + 0.001, 3)

    problems = verify_archive(_tampered(tmp_path, "espn_picks_2023.json", edit))
    assert any("differ from the capture" in p for p in problems)


def test_a_timestamp_other_than_the_archive_time_is_caught(tmp_path):
    def edit(data):
        data["captured_at"] = "2023-03-15T00:00:00+00:00"

    problems = verify_archive(_tampered(tmp_path, "espn_picks_2023.json", edit))
    assert any("differs from archive timestamp" in p for p in problems)


def test_captured_at_without_evidence_is_rejected(tmp_path):
    def edit(data):
        del data["capture_evidence"]

    assert verify_archive(_tampered(tmp_path, "espn_picks_2012.json", edit))


def test_unevidenced_archives_carry_no_capture_time():
    """Only archives with re-checkable evidence may claim a capture time."""
    claimed = [p.name for p in PICKS_DIR.glob("espn_picks_*.json") if "captured_at" in json.loads(p.read_text())]
    assert sorted(claimed) == EVIDENCED


def test_a_changed_api_share_is_caught(tmp_path):
    def edit(data):
        team = next(iter(data["teams"]))
        data["teams"][team]["R32"] = round(data["teams"][team]["R32"] + 0.0001, 4)

    problems = verify_archive(_tampered(tmp_path, "espn_picks_2024.json", edit))
    assert any("differ from the capture" in p for p in problems)
