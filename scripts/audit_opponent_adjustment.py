#!/usr/bin/env python3
"""Which model variables still carry conference strength as a confound?

THE FAILURE THIS LOOKS FOR. A raw per-game rate is measured against whoever a
team happened to play. Two teams of identical quality post different raw eFG%
if one plays in the Big Ten and the other in the Patriot League, because one
faced better defences. The variable then encodes "which conference is this"
alongside "how good is this team". That is survivable in a 1,008-row
tournament matrix where every team is good; it gets baked in permanently once
the row set expands to all of D1, where the conference-strength range is far
wider. So this runs BEFORE the expansion is adopted.

THE TEST. Partial correlation between the variable and its conference's mean
strength, CONTROLLING for team quality. Good teams cluster in good
conferences, so a raw correlation means nothing. What should not survive is a
relationship between two teams of EQUAL quality in unequal conferences.

WHY FOUR CONTROL COLUMNS AND NOT ONE. "Team quality" has no ground truth.
Control with a single rating and the test partly measures "does this variable
disagree with THAT rating" -- not the question, and the source of two concrete
errors here before it was understood. Controlling with barthag alone read
adj_offensive_efficiency at -0.026 (clean) because adj_OE is a COMPONENT of
barthag, so the control removed the variable from itself. Controlling with SRS
alone read t_rank at +0.203 (confounded); across instruments it resolves to
clean. Verdicts are read from STABILITY across disagreeing instruments, never
from one column.

THE INSTRUMENTS AND THEIR OWN BIASES, measured on inter-conference games as
corr(residual, conference-strength differential):

    RTH        +0.038   cleanest available; not consumed by the model
    SAG        +0.061   second cleanest; not consumed by the model
    SRS        -0.007 full-season, but +0.144 solved point-in-time
    barthag    +0.213   most biased; under-credits strong conferences

None is clean enough to stand alone, which is the point. A biased instrument
is still informative when the bias and its direction are known. RTH and SAG
are both kept rather than picking a single best one, for the same reason the
panel exists. Colley (+0.184) and GLM (+0.167) were measured and did NOT earn
columns.

SAMPLE WINDOWS ARE FORCED TO MATCH. RTH and SAG cover roughly half the seasons
SRS does. Left alone, a variable reading clean on RTH and confounded on SRS
could be instrument disagreement or an era difference, and the table could not
tell them apart. Every column is therefore restricted to the seasons ALL
instruments cover, so differing windows cannot masquerade as disagreement.

sos_avg_opp_barthag IS EXPECTED TO FAIL AND MUST NOT BE "FIXED". It measures
the quality of a team's opposition, so tracking conference strength is the
measurement working. A schedule-strength column uncorrelated with conference
strength would be broken.

Run: python3 scripts/audit_opponent_adjustment.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.features.custom_ratings import ratings_to_canonical  # noqa: E402
from src.data.features.point_in_time_kaggle import (  # noqa: E402
    ADJUSTABLE_RATES,
    adjusted_rates,
    box_profile,
    load_detailed_games,
    per_game_rates,
    season_is_complete,
)
from src.data.features.point_in_time_ratings import (  # noqa: E402
    SELECTION_SUNDAY_DAY,
    games_before,
    load_season_games,
    srs,
)

KAGGLE = REPO / "data" / "kaggle"
HIST = REPO / "data" / "raw" / "historical"

BASELINE_KEYS = [
    "barthag",
    "t_rank",
    "massey_avg_rank",
    "sos_avg_opp_barthag",
    "adj_offensive_efficiency",
    "adj_defensive_efficiency",
    "adj_tempo",
    "effective_fg_pct",
    "three_pt_pct",
    "three_pt_rate",
    "offensive_reb_rate",
    "turnover_rate",
]
BY_DESIGN = {"sos_avg_opp_barthag"}
CONTROL_SYSTEMS = ("RTH", "SAG")
CONFOUND_THRESHOLD = 0.10


def zwithin(values):
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [0.0] * len(values)
    m = statistics.fmean(present)
    sd = statistics.pstdev(present)
    if sd == 0:
        return [0.0] * len(values)
    return [0.0 if v is None else (v - m) / sd for v in values]


def partial_r(x, y, *controls) -> float:
    """corr(x, y) with every control regressed out of both."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    C = np.column_stack([np.ones(len(x))] + [np.asarray(c, float) for c in controls])

    def resid(v):
        beta, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ beta

    rx, ry = resid(x), resid(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    conf_of = defaultdict(dict)
    with open(KAGGLE / "MTeamConferences.csv") as f:
        for r in csv.DictReader(f):
            conf_of[int(r["Season"])][int(r["TeamID"])] = r["ConfAbbrev"]

    control_ranks = defaultdict(lambda: defaultdict(dict))
    massey_avg = defaultdict(lambda: defaultdict(list))
    with open(KAGGLE / "MMasseyOrdinals.csv") as f:
        for r in csv.DictReader(f):
            if int(r["RankingDayNum"]) < 128:
                continue
            s, tid, rank = int(r["Season"]), int(r["TeamID"]), int(r["OrdinalRank"])
            massey_avg[s][tid].append(rank)
            if r["SystemName"] in CONTROL_SYSTEMS:
                control_ranks[r["SystemName"]][s][tid] = rank

    per_season = {}
    for season in sorted(conf_of):
        tv_path = HIST / f"torvik_{season}.json"
        if not tv_path.exists():
            continue
        compact = load_season_games(season)
        if not compact or not season_is_complete(compact):
            continue
        detailed = load_detailed_games(season)
        if not detailed:
            continue
        if any(len(control_ranks[s].get(season, {})) < 200 for s in CONTROL_SYSTEMS):
            continue  # an instrument is missing this season; window must match

        tv = {t["team_id"]: t for t in json.loads(tv_path.read_text()).get("teams", [])}
        box = box_profile(detailed)
        adj = adjusted_rates(detailed)
        srs_r = srs(games_before(compact, SELECTION_SUNDAY_DAY))

        raw_acc = defaultdict(lambda: defaultdict(list))
        for team, _opp, rates in per_game_rates(detailed):
            for k, v in rates.items():
                if v is not None:
                    raw_acc[team][k].append(v)

        canon_of = {int(kid): c for c, kid in ratings_to_canonical({k: float(k) for k in srs_r}).items()}
        ids = []
        for kid, canon in canon_of.items():
            t = tv.get(canon)
            if not t or kid not in conf_of[season] or kid not in adj:
                continue
            if not isinstance(t.get("barthag"), (int, float)):
                continue
            ids.append(kid)
        if len(ids) < 200:
            continue

        bh = {k: tv[canon_of[k]]["barthag"] for k in ids}
        opp = defaultdict(list)
        for g in compact:
            for a, b in ((g.winner, g.loser), (g.loser, g.winner)):
                if b in bh:
                    opp[a].append(bh[b])

        zs = dict(zip(ids, zwithin([srs_r[k] for k in ids])))
        by_conf = defaultdict(list)
        for k in ids:
            by_conf[conf_of[season][k]].append(zs[k])
        cs = {c: statistics.fmean(v) for c, v in by_conf.items()}

        def col(fn):
            return zwithin([fn(k) for k in ids])

        def tvget(k, f):
            return tv[canon_of[k]].get(f)

        rec = {
            "conf": col(lambda k: cs[conf_of[season][k]]),
            "q_srs": col(lambda k: srs_r[k]),
            "q_barthag": col(lambda k: bh[k]),
            "adjusted": {},
            "raw_kaggle": {},
        }
        for sysname in CONTROL_SYSTEMS:
            tab = control_ranks[sysname][season]
            # negate so higher is better, matching every other instrument
            rec[f"q_{sysname}"] = col(lambda k, t=tab: -t[k] if k in t else None)

        rec["vars"] = {
            "barthag": col(lambda k: bh[k]),
            "t_rank": col(lambda k: tvget(k, "t_rank")),
            "massey_avg_rank": col(
                lambda k: -statistics.fmean(massey_avg[season][k]) if massey_avg[season].get(k) else None
            ),
            "sos_avg_opp_barthag": col(lambda k: statistics.fmean(opp[k]) if opp.get(k) else None),
            "adj_offensive_efficiency": col(lambda k: tvget(k, "adj_offensive_efficiency")),
            "adj_defensive_efficiency": col(lambda k: tvget(k, "adj_defensive_efficiency")),
            "adj_tempo": col(lambda k: tvget(k, "adj_tempo")),
            "effective_fg_pct": col(lambda k: tvget(k, "effective_fg_pct")),
            "three_pt_pct": col(lambda k: (box.get(k) or {}).get("three_pt_pct")),
            "three_pt_rate": col(lambda k: (box.get(k) or {}).get("three_pt_rate")),
            "offensive_reb_rate": col(lambda k: tvget(k, "offensive_reb_rate")),
            "turnover_rate": col(lambda k: tvget(k, "turnover_rate")),
        }
        for k in ADJUSTABLE_RATES:
            rec["adjusted"][k] = col(lambda t, kk=k: adj[t].get(kk))
            rec["raw_kaggle"][k] = col(lambda t, kk=k: statistics.fmean(raw_acc[t][kk]) if raw_acc[t].get(kk) else None)
        per_season[season] = rec

    common = sorted(per_season)
    if not common:
        print("no seasons carry every instrument")
        return 1

    def stack(path):
        out = []
        for s in common:
            node = per_season[s]
            for p in path:
                node = node[p]
            out.extend(node)
        return np.array(out)

    controls = ["q_RTH", "q_SAG", "q_srs", "q_barthag"]
    labels = {"q_RTH": "RTH", "q_SAG": "SAG", "q_srs": "SRS", "q_barthag": "barthag"}
    n_rows = sum(len(per_season[s]["conf"]) for s in common)

    print(f"\n{n_rows:,} team-seasons across {len(common)} seasons ({min(common)}-{max(common)})")
    print("every column on the identical window, so a disagreement is the instrument and not the era\n")
    print("  " + f"{'variable':<26}" + "".join(f"{labels[c]:>9}" for c in controls) + f"{'all':>9}   verdict")

    y = stack(["conf"])
    ctrl = {c: stack([c]) for c in controls}
    confounded = []

    for k in BASELINE_KEYS:
        x = stack(["vars", k])
        cells = [float("nan") if labels[c].lower() == k else partial_r(x, y, ctrl[c]) for c in controls]
        usable = [ctrl[c] for c in controls if labels[c].lower() != k]
        joint = partial_r(x, y, *usable)

        if k in BY_DESIGN:
            verdict = "by design (measures opposition)"
        elif abs(joint) >= CONFOUND_THRESHOLD:
            verdict = "CONFOUNDED"
            confounded.append((abs(joint), k, joint))
        elif any(abs(v) >= CONFOUND_THRESHOLD for v in cells if not np.isnan(v)):
            verdict = "unstable across instruments"
        else:
            verdict = "opponent-adjusted"
        cellstr = "".join("      n/a" if np.isnan(v) else f"{v:>+9.3f}" for v in cells)
        print(f"  {k:<26}{cellstr}{joint:>+9.3f}   {verdict}")

    print()
    if confounded:
        confounded.sort(reverse=True)
        print(
            f"  {len(confounded)} of {len(BASELINE_KEYS)} carry conference strength "
            f"at |partial r| >= {CONFOUND_THRESHOLD}:"
        )
        for _, k, v in confounded:
            way = "depressed" if v < 0 else "inflated"
            print(f"    {k:<26} {v:+.4f}  ({way} by a strong conference at fixed quality)")

    # NOT AN INDEPENDENT CONTROL. These columns are the OUTPUT of the opponent
    # adjustment this audit motivated, so this section asks only "did the fix
    # attenuate the confound it targeted". It is the fix grading its own
    # homework: legitimate for measuring attenuation, and NOT evidence that the
    # adjusted variable is clean in the sense the panel above establishes.
    # Read it as a before/after, never as a fifth instrument.
    print("\n  FIX VERIFICATION -- not an independent control; this is the")
    print("  adjustment's own output measured against the confound it targeted.")
    print(f"  {'rate':<26}{'raw':>9}{'adjusted':>10}   attenuation")
    allc = [ctrl[c] for c in controls]
    for k in ADJUSTABLE_RATES:
        r0 = partial_r(stack(["raw_kaggle", k]), y, *allc)
        r1 = partial_r(stack(["adjusted", k]), y, *allc)
        mark = "attenuated" if abs(r1) < CONFOUND_THRESHOLD else "still confounded"
        print(f"  {k:<26}{r0:>+9.3f}{r1:>+10.3f}   {abs(r0) - abs(r1):+.3f} {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
