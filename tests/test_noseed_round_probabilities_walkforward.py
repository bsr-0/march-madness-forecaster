"""build_noseed_round_probabilities must be walk-forward (2026-09 audit, Step 3, F3-4).

It scales seed advancement rates by a compounding no-seed advantage. Until the
audit its seed rates came from the hardcoded 1985-2025 "full" table with no
season cutoff, so every backtested season's own results were in its base
rates -- while its blend partner seed_rp used the "recent" window with
as_of=year. The two halves of the blend must see the same, causal, table.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import seed_pick_model as SPM  # noqa: E402
from src.prediction import noseed_model as NM  # noqa: E402

pytestmark = pytest.mark.leakage


class _FlatModel:
    """A model with no opinion: predict_win_prob == the seed rate, so mean_adv == 0
    and the output must equal the seed advancement rates exactly."""

    def __init__(self, seeds, as_of):
        self.seeds, self.as_of = seeds, as_of

    def predict_win_prob(self, a, b):
        return SPM._win_rate(a["seed"], b["seed"], "recent", self.as_of)


def test_requires_as_of_and_uses_the_recent_window(monkeypatch):
    seeds = {f"t{s}": s for s in range(1, 17)}
    stats = {t: {"seed": s} for t, s in seeds.items()}
    monkeypatch.setattr(NM, "validate_stats_payload", lambda *a, **k: None)
    with pytest.raises(TypeError):
        NM.build_noseed_round_probabilities(_FlatModel(seeds, 2019), seeds, stats)  # type: ignore[call-arg]

    calls = []
    real = SPM._win_rate

    def spy(a, b, window="full", as_of=None):
        calls.append((window, as_of))
        return real(a, b, window, as_of)

    monkeypatch.setattr(NM, "_win_rate", spy)
    rp = NM.build_noseed_round_probabilities(_FlatModel(seeds, 2019), seeds, stats, as_of=2019)
    assert calls and all(w == "recent" and y == 2019 for w, y in calls), set(calls)
    expected = SPM._compute_advancement_rates("recent", 2019)
    for t, s in seeds.items():
        for R in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
            assert rp[t][R] == pytest.approx(max(0.001, min(0.99, expected[s][R])), abs=1e-12)


def test_target_season_does_not_move_its_own_rates(monkeypatch):
    seeds = {f"t{s}": s for s in range(1, 17)}
    stats = {t: {"seed": s} for t, s in seeds.items()}
    monkeypatch.setattr(NM, "validate_stats_payload", lambda *a, **k: None)
    a = NM.build_noseed_round_probabilities(_FlatModel(seeds, 2019), seeds, stats, as_of=2019)
    b = NM.build_noseed_round_probabilities(_FlatModel(seeds, 2019), seeds, stats, as_of=2020)
    # 2019's results enter the as_of=2020 table and not the as_of=2019 one
    assert any(a[t]["R64"] != b[t]["R64"] for t in seeds)
