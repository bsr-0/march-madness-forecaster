"""The artifact is the contract: the browser must not reconstruct model math.

Python owns the canonical semantics. The browser renders. Before schema 3 the
bracket board computed each game's displayed win probability from team ratings
with a client-side log5 — a second, unversioned implementation of tournament
math that could drift from the simulation bank the brackets were drawn from.

Schema 3 ships the same ``PairwiseProbabilities`` table that drove the
simulation, and this module holds the line:

  * the artifact actually carries the table, and it is coherent;
  * the browser reads it rather than deriving it;
  * Python and JavaScript render identical values for every game.

This is an implementation-contract change, not a methodology change. The
candidates in the regenerated artifact are bit-identical to the schema 2 ones;
only the transport of an already-computed number moved.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.product.selection import candidate_games, pairwise_prob, select_with_alternative

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "docs" / "data" / "candidates_2026.json"
SELECTION_JS = REPO / "docs" / "selection.js"
BUILD_JS = REPO / "docs" / "build.js"

pytestmark = pytest.mark.skipif(not ARTIFACT.exists(), reason="candidate artifact not built")


@pytest.fixture(scope="module")
def artifact():
    return json.loads(ARTIFACT.read_text())


# ---------------------------------------------------------------------------
# The artifact carries what the browser needs
# ---------------------------------------------------------------------------


def test_artifact_ships_the_pairwise_table(artifact):
    assert artifact["schema"] >= 3, "schema 3 introduced the pairwise table"
    n = len(artifact["teams"])
    assert len(artifact["pairwise"]) == n * n


def test_pairwise_table_is_coherent(artifact):
    """Probabilities, and P(a beats b) + P(b beats a) == 1 within rounding."""
    n = len(artifact["teams"])
    table = artifact["pairwise"]
    assert all(0.0 <= v <= 1.0 for v in table)
    for i in range(n):
        assert table[i * n + i] == 0.5, "self-matchup should be the neutral 0.5"
        for j in range(i + 1, n):
            assert abs(table[i * n + j] + table[j * n + i] - 1.0) <= 2e-4, (
                f"pairwise table is not symmetric for ({i}, {j})"
            )


def test_every_game_a_user_can_see_has_a_probability(artifact):
    """Rendering must never fall back to a default because the table was thin."""
    for index in select_with_alternative(artifact, "ev") + select_with_alternative(artifact, "p1"):
        games = candidate_games(artifact, index)
        assert len(games) == 63
        assert all(0.0 < g["win_prob"] < 1.0 for g in games)


# ---------------------------------------------------------------------------
# The browser does not reconstruct model probabilities
# ---------------------------------------------------------------------------


def test_browser_does_not_implement_log5(artifact):
    """The client must not turn ratings into probabilities.

    Scans for the arithmetic, not just the name: renaming the function would
    otherwise defeat this. The log5 form is distinctive — it combines two ratings
    with products of a rating and its complement.
    """
    for path in (SELECTION_JS, BUILD_JS):
        src = path.read_text()
        assert "log5" not in src, f"{path.name} still references log5"
        assert not re.search(r"\bbarthag\b", src), (
            f"{path.name} reads a team rating; the browser must consume shipped "
            "probabilities, not derive them from ratings"
        )
        # The characteristic log5 denominator: a*(1-b) + b*(1-a).
        assert not re.search(r"\(\s*1\s*-\s*\w+\s*\)\s*\*\s*\w+\s*\+", src), (
            f"{path.name} contains log5-shaped arithmetic"
        )


def test_browser_reads_the_shipped_table(artifact):
    """Positive counterpart: the lookup exists and the renderer uses it."""
    src = SELECTION_JS.read_text()
    assert "function pairwiseProb(" in src
    assert re.search(r"const wp = pairwiseProb\(", src), (
        "candidateToRounds no longer sources win_prob from the shipped table"
    )


def test_candidate_to_rounds_takes_no_probability_function(artifact):
    """The old signature accepted a log5Fn; reintroducing it reopens the hole."""
    src = SELECTION_JS.read_text()
    sig = re.search(r"function candidateToRounds\(([^)]*)\)", src)
    assert sig, "candidateToRounds not found"
    params = [p.strip() for p in sig.group(1).split(",")]
    assert not any("log5" in p.lower() or "prob" in p.lower() for p in params), (
        f"candidateToRounds accepts a probability function again: {params}"
    )


# ---------------------------------------------------------------------------
# THE GATE: Python and JavaScript render the same numbers
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_python_and_javascript_render_identical_games(artifact):
    """Game-for-game agreement, not merely 'both produce 63 games'."""
    runner = """
    const sel = require(process.argv[2]);
    const fs = require('fs');
    const art = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
    const mkTeam = (id, name, seed) => ({ id, name, seed, barthag: null });
    const out = {};
    for (const obj of ['ev', 'p1']) {
      for (const idx of sel.selectWithAlternative(art, obj)) {
        if (out[idx]) continue;
        out[idx] = sel.candidateToRounds(art, idx, mkTeam).flatMap(r =>
          r.games.map(g => [g.team1.id, g.team2.id, g.win_prob, g.precomputed_winner_id]));
      }
    }
    console.log(JSON.stringify(out));
    """
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(runner)
        script = f.name

    proc = subprocess.run(
        ["node", script, str(SELECTION_JS), str(ARTIFACT)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"node failed: {proc.stderr[:400]}"
    js_out = json.loads(proc.stdout)
    assert js_out, "the JS runner produced no brackets to compare"

    for idx_str, js_games in js_out.items():
        py_games = candidate_games(artifact, int(idx_str))
        assert len(js_games) == len(py_games) == 63
        for n, (js, py) in enumerate(zip(js_games, py_games)):
            t1, t2, wp, winner = js
            assert (t1, t2) == (py["team1"], py["team2"]), (
                f"candidate {idx_str} game {n}: JS pairs {t1} vs {t2}, Python pairs "
                f"{py['team1']} vs {py['team2']} -- the two disagree about the bracket"
            )
            assert wp == pytest.approx(py["win_prob"], abs=1e-9), (
                f"candidate {idx_str} game {n} ({t1} vs {t2}): JS shows {wp}, Python shows {py['win_prob']}"
            )
            assert winner == py["winner"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_javascript_refuses_a_schema_2_artifact():
    """A stale artifact must fail loudly, not render invented probabilities."""
    runner = """
    const sel = require(process.argv[2]);
    try {
      sel.pairwiseProb({ teams: [{}, {}] }, 0, 1);
      console.log('NO_THROW');
    } catch (e) { console.log('THREW'); }
    """
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(runner)
        script = f.name
    proc = subprocess.run(["node", script, str(SELECTION_JS)], capture_output=True, text=True, timeout=60)
    assert proc.stdout.strip() == "THREW", (
        "an artifact without a pairwise table did not raise; the browser would "
        "silently render undefined or fall back to its own arithmetic"
    )


def test_python_refuses_a_schema_2_artifact():
    with pytest.raises(KeyError, match="pairwise"):
        pairwise_prob({"teams": [{}, {}]}, 0, 1)
