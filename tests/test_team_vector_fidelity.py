"""TeamFeatures.to_vector() must carry every named field, unaltered.

The inference vector is built from TeamFeatures.to_vector(); training vectors
come from metrics_to_team_vector(). They must agree slot-for-slot or the model
is fit on one thing and served another. A ±1000 magnitude clip at the end of
to_vector() turned every Elo rating (1278-2128 in 2024) into exactly 1000.0,
zeroing diff_elo_rating -- the production model's dominant feature -- at
inference while training saw it intact. Predictions collapsed to ~0.5.

The module only asserted the vector's LENGTH matched the names list. These
tests assert the CONTENT: perturbing the field behind each name moves exactly
that slot, and realistic values round-trip untouched.
"""

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from src.data.features.feature_engineering import TEAM_FEATURE_DIM, TeamFeatures

NAMES = TeamFeatures.get_feature_names()
ROOT = Path(__file__).resolve().parents[1]


def _base():
    return TeamFeatures(team_id="x", team_name="X", seed=1, region="East")


def _moved_slots(field: str, value: float):
    a, b = _base(), _base()
    setattr(b, field, value)
    va, vb = a.to_vector(include_embeddings=False), b.to_vector(include_embeddings=False)
    return [i for i in range(TEAM_FEATURE_DIM) if not np.isclose(va[i], vb[i], equal_nan=True)]


NUMERIC_FIELDS = [
    f.name for f in dataclasses.fields(TeamFeatures)
    if f.name not in ("team_id", "team_name", "seed", "region", "gnn_embedding", "transformer_embedding")
    and isinstance(getattr(_base(), f.name, None), (int, float))
    and not isinstance(getattr(_base(), f.name, None), bool)
]


def test_elo_is_served_at_its_slot_with_real_magnitudes():
    for elo in (1278.0, 1500.0, 1777.0, 2128.5):
        t = _base()
        t.elo_rating = elo
        v = t.to_vector(include_embeddings=False)
        assert v[NAMES.index("elo_rating")] == elo, f"elo {elo} was altered to {v[NAMES.index('elo_rating')]}"


def test_two_teams_with_different_elo_have_a_nonzero_diff():
    a, b = _base(), _base()
    a.elo_rating, b.elo_rating = 2100.0, 1300.0
    diff = a.to_vector(include_embeddings=False) - b.to_vector(include_embeddings=False)
    assert diff[NAMES.index("elo_rating")] == pytest.approx(800.0)


@pytest.mark.parametrize("field", NUMERIC_FIELDS)
def test_each_numeric_field_moves_at_most_one_slot(field):
    """A field that feeds the vector must land in exactly one slot; a field that
    does not feed it (kept for downstream code) must move nothing."""
    moved = _moved_slots(field, 12345.0)
    assert len(moved) <= 1, f"{field} moves several slots: {[NAMES[i] for i in moved]}"


def test_large_values_are_not_clipped():
    t = _base()
    t.elo_rating = 5000.0
    assert t.to_vector(include_embeddings=False)[NAMES.index("elo_rating")] == 5000.0


def test_no_magnitude_clip_in_to_vector():
    src = (ROOT / "src" / "data" / "features" / "feature_engineering.py").read_text()
    start = src.index("    def to_vector(self, include_embeddings")
    end = src.index("        return result", start)
    body = src[start:end]
    assert not re.search(r"np\.clip\(\s*result", body), "a magnitude clip on the raw team vector is back"
