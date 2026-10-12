"""Every untracked input a candidate artifact records must be recoverable byte-for-byte."""

import json

from scripts import restore_build_inputs as rbi


def test_archive_covers_every_untracked_recorded_input():
    archived = json.loads(rbi.MANIFEST.read_text())["files"]
    tracked = rbi._tracked()
    for file, digest in rbi.recorded_inputs().items():
        if file in tracked:
            continue
        assert file in archived, f"{file} is recorded by an artifact but not archived"
        assert archived[file]["sha256"] == digest, f"{file}: archive holds different bytes"


def test_archive_decompresses_to_the_recorded_bytes():
    assert rbi.check() == []
