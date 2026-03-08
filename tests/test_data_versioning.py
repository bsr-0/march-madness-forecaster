"""Tests for data versioning (snapshot/restore)."""

import json

import pytest

from src.data.versioning import (
    list_snapshots,
    restore_snapshot,
    snapshot_data_dir,
)


@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary data directory with some files."""
    d = tmp_path / "data" / "raw"
    d.mkdir(parents=True)
    (d / "teams.json").write_text('{"teams": []}')
    (d / "torvik.json").write_text('{"data": [1,2,3]}')
    sub = d / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("hello")
    return str(d)


@pytest.fixture
def snapshot_dir(tmp_path):
    return str(tmp_path / "snapshots")


class TestSnapshot:
    def test_create_snapshot(self, data_dir, snapshot_dir):
        sid = snapshot_data_dir(data_dir, label="test", snapshot_base=snapshot_dir)
        assert "test" in sid

        # Check manifest was created
        import os
        snap_path = os.path.join(snapshot_dir, sid)
        assert os.path.exists(os.path.join(snap_path, "manifest.json"))
        assert os.path.exists(os.path.join(snap_path, "teams.json"))
        assert os.path.exists(os.path.join(snap_path, "sub", "nested.txt"))

    def test_manifest_content(self, data_dir, snapshot_dir):
        sid = snapshot_data_dir(data_dir, label="v1", snapshot_base=snapshot_dir)

        import os
        with open(os.path.join(snapshot_dir, sid, "manifest.json")) as f:
            manifest = json.load(f)

        assert manifest["file_count"] == 3
        assert manifest["label"] == "v1"
        assert "teams.json" in manifest["file_hashes"]
        assert len(manifest["file_hashes"]["teams.json"]) == 64  # SHA-256 hex

    def test_nonexistent_dir_raises(self, snapshot_dir):
        with pytest.raises(FileNotFoundError):
            snapshot_data_dir("/nonexistent/dir", snapshot_base=snapshot_dir)


class TestListSnapshots:
    def test_no_snapshots(self, snapshot_dir):
        assert list_snapshots(snapshot_dir) == []

    def test_list_after_create(self, data_dir, snapshot_dir):
        snapshot_data_dir(data_dir, label="first", snapshot_base=snapshot_dir)
        snapshot_data_dir(data_dir, label="second", snapshot_base=snapshot_dir)

        snapshots = list_snapshots(snapshot_dir)
        assert len(snapshots) == 2
        labels = {s.label for s in snapshots}
        assert "first" in labels
        assert "second" in labels

    def test_all_snapshots_returned(self, data_dir, snapshot_dir):
        snapshot_data_dir(data_dir, label="first", snapshot_base=snapshot_dir)
        snapshot_data_dir(data_dir, label="second", snapshot_base=snapshot_dir)

        snapshots = list_snapshots(snapshot_dir)
        labels = {s.label for s in snapshots}
        assert "first" in labels
        assert "second" in labels


class TestRestoreSnapshot:
    def test_restore(self, data_dir, snapshot_dir, tmp_path):
        sid = snapshot_data_dir(data_dir, label="restore_test", snapshot_base=snapshot_dir)

        # Restore to a new directory
        target = str(tmp_path / "restored")
        n = restore_snapshot(sid, target, snapshot_base=snapshot_dir)
        assert n == 3

        import os
        assert os.path.exists(os.path.join(target, "teams.json"))
        assert os.path.exists(os.path.join(target, "sub", "nested.txt"))

    def test_restore_verifies_hashes(self, data_dir, snapshot_dir, tmp_path):
        sid = snapshot_data_dir(data_dir, label="hash_test", snapshot_base=snapshot_dir)
        target = str(tmp_path / "restored")
        # Should not raise
        n = restore_snapshot(sid, target, snapshot_base=snapshot_dir, verify=True)
        assert n > 0

    def test_restore_nonexistent_raises(self, snapshot_dir, tmp_path):
        with pytest.raises(FileNotFoundError):
            restore_snapshot("nonexistent", str(tmp_path / "r"), snapshot_base=snapshot_dir)

    def test_roundtrip_integrity(self, data_dir, snapshot_dir, tmp_path):
        """Snapshot → restore → compare content."""
        sid = snapshot_data_dir(data_dir, label="roundtrip", snapshot_base=snapshot_dir)
        target = str(tmp_path / "roundtrip_target")
        restore_snapshot(sid, target, snapshot_base=snapshot_dir)

        import os
        original = open(os.path.join(data_dir, "teams.json")).read()
        restored = open(os.path.join(target, "teams.json")).read()
        assert original == restored
