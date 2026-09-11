"""The shared pipeline path must hand rosters to feature engineering.

`PipelineRunner._engineer_features` reads `roster=getattr(p, "_rosters", {})`.
`_load_data` built the rosters and returned them, but never assigned
`p._rosters`, so on the shared path -- which production and the calibration
harness both use -- every team was built with `roster=None`. All
roster-derived features (total_warp, top5_rapm, total_rapm,
roster_continuity, bench_depth) were served as exactly 0.0 for all 64 teams
while training rows carried real values: two of the nine production features
dead at inference, on top of the Elo clip. `train_for_predictions` passed the
dict directly and never had the bug.
"""

import re
from pathlib import Path

from src.pipeline import pipeline_runner as pr

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src" / "pipeline" / "pipeline_runner.py").read_text()


def _body(func_name: str) -> str:
    start = SRC.index(f"    def {func_name}(")
    nxt = re.search(r"\n    def ", SRC[start + 10 :])
    return SRC[start : start + 10 + (nxt.start() if nxt else len(SRC))]


def test_load_data_attaches_rosters_to_the_pipeline():
    assert "p._rosters = rosters" in _body("_load_data")


def test_engineer_features_reads_the_attribute_load_data_sets():
    body = _body("_engineer_features")
    assert 'getattr(p, "_rosters"' in body or "p._rosters" in body


def test_every_private_attribute_read_on_the_shared_path_has_a_writer():
    """The bug's shape: a getattr(p, "_x", default) read whose writer was never
    written. Every such read in the runner must have a `p._x =` somewhere in
    src/pipeline."""
    reads = set(re.findall(r'getattr\(p, "(_[a-z_]+)"', SRC))
    pipeline_src = "\n".join(q.read_text() for q in (ROOT / "src" / "pipeline").rglob("*.py"))
    missing = sorted(a for a in reads if not re.search(rf"\b(?:p|self|pipeline)\.{re.escape(a)}\s*=", pipeline_src))
    assert not missing, f"read via getattr but never assigned anywhere in src/pipeline: {missing}"


def test_feature_loop_receives_rosters(monkeypatch):
    """Functional: with p._rosters set, extract_team_features gets the roster."""
    from types import SimpleNamespace

    seen = {}

    class _FE:
        def extract_team_features(self, **kw):
            seen[kw["team_id"]] = kw["roster"]
            return SimpleNamespace(external_rating_composite=None, external_rating_spread=None)

    class _Phase:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    p = SimpleNamespace(
        _team_id=lambda name: name.lower(),
        team_struct={}, team_id_to_name={}, team_name_to_id={},
        feature_engineer=_FE(), _external_composites={}, _massey_multi={},
        _torvik_map={}, _proprietary_map={},
        _rosters={"alpha": "ROSTER-A"},
        _resource_tracker=SimpleNamespace(phase=lambda name: _Phase()),
        config=SimpleNamespace(year=2030),
    )
    runner = pr._PipelineRunner.__new__(pr._PipelineRunner)
    runner._p = p
    teams = [SimpleNamespace(name="Alpha", seed=1, region="East"), SimpleNamespace(name="Beta", seed=2, region="West")]
    try:
        runner._engineer_features(teams, game_flows={})
    except Exception as exc:  # the loop continues past our stub after the roster hand-off
        if "alpha" not in seen:
            raise
    assert seen.get("alpha") == "ROSTER-A"
    assert seen.get("beta") is None
