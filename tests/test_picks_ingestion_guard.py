"""The ESPN picks ingestion must not overwrite archives that carry verified capture evidence."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts import scrape_historical_espn_picks as scraper
from scripts.experiments import verify_picks_capture_evidence as verifier

REPO = Path(__file__).resolve().parent.parent
NEW = {"year": 2023, "source": "kaggle_espn", "teams": {"alabama": {"R64": 0.5}}}


def _copy(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    shutil.copy(verifier.PICKS_DIR / name, path)
    return path


def test_an_evidenced_archive_is_not_overwritten(tmp_path):
    path = _copy(tmp_path, "espn_picks_2023.json")
    before = path.read_bytes()
    assert scraper.is_evidenced(path)
    assert scraper.write_archive(path, NEW) is False
    assert path.read_bytes() == before


def test_an_unevidenced_archive_is_still_written(tmp_path):
    path = tmp_path / "espn_picks_2011.json"
    path.write_text(json.dumps({"year": 2011, "teams": {}}))
    assert not scraper.is_evidenced(path)
    assert scraper.write_archive(path, NEW) is True
    assert json.loads(path.read_text()) == NEW
    assert scraper.write_archive(tmp_path / "espn_picks_2099.json", NEW) is True


def test_replacing_evidence_requires_the_explicit_flag(tmp_path):
    path = _copy(tmp_path, "espn_picks_2023.json")
    assert scraper.write_archive(path, NEW, replace_evidenced=True) is True
    assert json.loads(path.read_text()) == NEW


def test_the_csv_import_entry_point_refuses_too(tmp_path, monkeypatch):
    path = _copy(tmp_path, "espn_picks_2023.json")
    before = path.read_bytes()
    monkeypatch.setattr(scraper, "import_from_csv", lambda csv, year: dict(NEW))
    monkeypatch.setattr(scraper, "validate_existing_files", lambda d: 0)
    monkeypatch.setattr(sys, "argv", ["x", "--from-csv", "ignored.csv", "--year", "2023", "--output-dir", str(tmp_path)])
    assert scraper.main() == 1
    assert path.read_bytes() == before


def test_the_scrape_entry_point_keeps_the_archive(tmp_path, monkeypatch):
    path = _copy(tmp_path, "espn_picks_2023.json")
    before = path.read_bytes()
    monkeypatch.setattr(scraper, "scrape_year", lambda year: dict(NEW))
    monkeypatch.setattr(scraper.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["x", "--years", "2023", "--output-dir", str(tmp_path)])
    assert scraper.main() == 0
    assert path.read_bytes() == before


def test_the_workflow_check_catches_a_swapped_archive(tmp_path):
    """A replacement that drops the evidence is invisible to verify_archive; the base-ref check sees it."""
    repo = tmp_path / "repo"
    picks = repo / "data" / "raw" / "historical_public_picks"
    picks.mkdir(parents=True)
    shutil.copy(verifier.PICKS_DIR / "espn_picks_2023.json", picks)

    def git(*a):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)

    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "base")
    orig = verifier.REPO
    verifier.REPO = repo
    try:
        assert verifier.protected_against("HEAD", picks) == []
        (picks / "espn_picks_2023.json").write_text(json.dumps(NEW))
        assert any("differs" in p for p in verifier.protected_against("HEAD", picks))
        (picks / "espn_picks_2023.json").unlink()
        assert any("deleted" in p for p in verifier.protected_against("HEAD", picks))
    finally:
        verifier.REPO = orig
