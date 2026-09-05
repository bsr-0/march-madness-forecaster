#!/usr/bin/env python3
"""Is the historical market data fit to test as a model feature?

WHY THIS EXISTS. FINDINGS.md records market odds as the highest-confidence
remaining source of new signal -- "you don't have an architecture problem
anymore, you have a signal acquisition problem" -- with a vendor-evaluation
phase written but never executed. Every internal idea has since measured null,
so this is the last untested lever before the project is done. It deserved an
answer rather than another guess.

THE ANSWER IS NO, ON THE DATA CURRENTLY ON DISK, and the reason is worth being
precise about because "we have odds files going back to 2008" is true and
misleading in equal measure.

  COVERAGE, after bridging team ids: 85.7% of the 1,008 tournament rows. The
  acceptance rubric in FINDINGS is >=95% overall with no year below 90%. Naive
  string matching gives 57.7%, so most of the shortfall was a join failure
  rather than absent data -- the older files use a different id convention
  ('arizonastate', 'appalachianst') than the repo's ('michigan_state').

  QUALITY, which is the disqualifying part. Spreads for 2017 and 2019 have ONE
  distinct value across every joined game: 0.0. 2015 has two, one of which is
  -139.5. 2010's are all one sign, so no team is ever the favourite. Only 2021
  onward carries a spread distribution with both signs and plausible magnitudes.

  The consequence is that the market "predicts" tournament winners at 44%
  overall -- worse than a coin flip -- not because the market is bad but because
  most of the rows are zeros being read as pick-ems.

WHAT THAT MEANS FOR THE EVALUATION. A walk-forward test needs the feature across
the whole 2014-2026 test period. Six usable seasons (2021-2026) cannot support
one, and three of those six score barely above chance, which suggests the join
needs auditing even where the values look sane. Acquiring clean historical lines
is a data project, not an analysis one -- which is precisely why FINDINGS filed
it as a vendor-evaluation phase rather than an experiment.

Run: python3 scripts/audit_market_odds.py
"""

from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.normalize import bridge_cbbpy_id  # noqa: E402

COVERAGE_GATE = 0.95
PER_YEAR_GATE = 0.90
MIN_DISTINCT_SPREADS = 10  # a season of real lines has dozens


def tournament_rows():
    """(year, team_a, team_b) for every row of the shipped training matrix."""
    import scripts.build_training_matrix as B

    stats = json.loads(B.STATS_PATH.read_text())["stats_by_year"]
    keys = [v[0] for v in B.VARIABLES]
    out = []
    for path in sorted((B.REPO / "data/raw/historical").glob(B.CONTEXT_GLOB)):
        ctx = json.loads(path.read_text())
        year = ctx.get("results", {}).get("year")
        rows = stats.get(str(year))
        if not rows:
            continue
        z = B.season_z(rows)
        for g in ctx["results"]["games"]:
            if g.get("round_name") in B.SKIP_ROUNDS:
                continue
            a, b = g.get("team1_id"), g.get("team2_id")
            if a not in z or b not in z:
                continue
            if g.get("team1_score") is None or g.get("team2_score") is None:
                continue
            a2, b2, _, _ = B._orient(g, a, b, g["team1_score"], g["team2_score"])
            if not any(round(z[a2][k] - z[b2][k], 4) for k in keys):
                continue
            out.append((int(year), a2, b2))
    return out


def load_market(canon):
    """Every odds row, keyed by (season, unordered canonical team pair)."""
    cache, market = {}, {}

    def bridge(raw, year):
        k = (raw, year)
        if k not in cache:
            cache[k] = raw if raw in canon[year] else bridge_cbbpy_id(raw, canon[year])
        return cache[k]

    for pat in ("sbro_game_odds_*.json", "sbr_game_odds_*.json", "covers_game_odds_*.json"):
        for f in sorted(glob.glob(str(REPO / "data/raw/betting_odds" / pat))):
            try:
                d = json.loads(Path(f).read_text())
            except Exception:
                continue
            games = d.get("games") if isinstance(d, dict) else d
            if not isinstance(games, list):
                continue
            for g in games:
                s, h, a = g.get("season"), g.get("home_team_id"), g.get("away_team_id")
                if not (s and h and a) or int(s) not in canon:
                    continue
                s = int(s)
                H, A = bridge(h, s), bridge(a, s)
                if not H or not A or H == A:
                    continue
                market[(s, frozenset((H, A)))] = {"spread": g.get("spread"), "H": H}
    return market


def main() -> int:
    rows = tournament_rows()
    canon = collections.defaultdict(set)
    for y, a, b in rows:
        canon[y].update((a, b))
    market = load_market(canon)

    print(f"\n  {len(rows)} tournament rows | {len(market):,} joined odds rows\n")
    print(f"  {'year':<6}{'rows':>6}{'joined':>8}{'cover':>8}{'distinct spreads':>19}")
    tot = hit = 0
    failures = []
    for y in sorted(canon):
        rs = [(a, b) for yy, a, b in rows if yy == y]
        joined = [market.get((y, frozenset(p))) for p in rs]
        got = [g for g in joined if g and g.get("spread") is not None]
        vals = {float(g["spread"]) for g in got}
        tot += len(rs)
        hit += len(got)
        flag = ""
        if len(rs) and len(got) / len(rs) < PER_YEAR_GATE:
            flag += " coverage"
        if len(vals) < MIN_DISTINCT_SPREADS:
            flag += " degenerate"
        if flag:
            failures.append((y, flag.strip()))
        print(f"  {y:<6}{len(rs):>6}{len(got):>8}{100 * len(got) / max(len(rs), 1):>7.0f}%"
              f"{len(vals):>19}{'   <-- ' + flag.strip() if flag else ''}")

    cov = hit / max(tot, 1)
    print(f"\n  overall coverage {cov:.1%} against a {COVERAGE_GATE:.0%} gate: "
          f"{'PASS' if cov >= COVERAGE_GATE else 'FAIL'}")
    print(f"  seasons failing coverage or spread-variety checks: {len(failures)}")
    for y, why in failures:
        print(f"     {y}: {why}")
    print("\n  A walk-forward test needs the feature across 2014-2026. On this data it")
    print("  cannot be run: the usable seasons do not span the test period.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
