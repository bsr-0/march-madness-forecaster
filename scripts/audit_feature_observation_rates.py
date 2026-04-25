#!/usr/bin/env python3
"""Audit tournament-team feature observation rates by season.

This script measures how often each team-level feature is *actually observed*
for seeded/bracket teams using the same incremental vector construction path
that historical training uses:

  historical_games -> IncrementalMetricsEngine -> metrics_to_team_vector
  + roster overlay
  + external ratings / Massey overlays

The output is a single CSV table with one row per feature and one column per
audited year. Each feature chooses one observation rule:

* ``non_zero``: feature counts as observed when ``abs(value) > 1e-8``
* ``non_default_0_25``: used for ``preseason_ap_rank`` because the incremental
  path hardcodes an unranked default of ``0.25``
* ``non_nan``: used for external-rating / Massey features, where missingness is
  represented as ``NaN`` rather than zero
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.features.feature_engineering import TeamFeatures
from src.data.features.proprietary_metrics import (
    IncrementalMetricsEngine,
    _team_id,
    team_games_to_game_records,
)
from src.data.features.torvik_ff_lookup import TorVikFFLookup
from src.pipeline.config import TOURNAMENT_START_DATES
from src.pipeline.stages.data_loader import load_roster_overlay

logger = logging.getLogger(__name__)


EXTERNAL_NAN_FEATURES = {
    "external_rating_composite",
    "external_rating_spread",
    "massey_pom",
    "massey_sag",
    "massey_mor",
    "massey_dol",
    "massey_col",
    "massey_wol",
    "massey_rth",
    "massey_ap",
    "massey_usa",
    "massey_rpi",
    "massey_rank_mean",
    "massey_rank_std",
}

DEFAULT_025_FEATURES = {"preseason_ap_rank"}


@dataclass
class YearAudit:
    year: int
    team_count: int
    rates: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--cut-start-year", type=int, default=2012)
    parser.add_argument("--cut-end-year", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--historical-dir", default="data/raw/historical")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--kaggle-dir", default="data/kaggle")
    parser.add_argument(
        "--output-csv",
        default="artifacts/feature_observation_rates_2008_2026.csv",
    )
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _load_conference_map(metrics_path: Path) -> Optional[Dict[str, str]]:
    try:
        payload = _load_json(metrics_path)
    except Exception as exc:
        logger.debug("Conference map extraction failed from %s: %s", metrics_path, exc)
        return None

    conference_map: Dict[str, str] = {}
    if isinstance(payload, dict):
        if isinstance(payload.get("teams"), list):
            for row in payload["teams"]:
                if not isinstance(row, dict):
                    continue
                raw_name = row.get("team_id") or row.get("team_name") or row.get("name") or ""
                tid = _team_id(str(raw_name))
                conf = row.get("conference") or row.get("conf") or row.get("conference_name")
                if tid and conf:
                    conference_map[tid] = str(conf)
        else:
            for tid, info in payload.items():
                if isinstance(info, dict) and "conference" in info:
                    conference_map[_team_id(str(tid))] = str(info["conference"])

    return conference_map or None


def _load_seed_map(year: int, historical_dir: Path, raw_dir: Path) -> Dict[str, int]:
    seed_map: Dict[str, int] = {}

    seeds_path = historical_dir / f"tournament_seeds_{year}.json"
    if seeds_path.exists():
        payload = _load_json(seeds_path)
        teams = payload.get("teams", payload) if isinstance(payload, dict) else payload
        if isinstance(teams, list):
            for entry in teams:
                if not isinstance(entry, dict):
                    continue
                seed = int(entry.get("seed", 0) or 0)
                if not seed:
                    continue
                for key in ("team_id", "school_slug", "team_name", "name"):
                    raw = entry.get(key)
                    if raw:
                        seed_map[_team_id(str(raw))] = seed
                        break
        if seed_map:
            return seed_map

    bracket_path = raw_dir / f"bracket_{year}.json"
    if bracket_path.exists():
        payload = _load_json(bracket_path)
        for entry in payload.get("teams", []):
            if not isinstance(entry, dict):
                continue
            seed = int(entry.get("seed", 0) or 0)
            if not seed:
                continue
            for key in ("team_id", "school_slug", "team_name", "name"):
                raw = entry.get(key)
                if raw:
                    seed_map[_team_id(str(raw))] = seed
                    break

    return seed_map


def _build_prefix_aliases(seed_map: Dict[str, int], available_ids: Iterable[str]) -> Dict[str, int]:
    aliases: Dict[str, int] = {}
    available_ids = set(available_ids)
    for aid in available_ids:
        if aid in seed_map:
            continue
        matches = [(sid, seed) for sid, seed in seed_map.items() if aid.startswith(sid + "_")]
        if len(matches) == 1:
            aliases[aid] = matches[0][1]
    return aliases


def _load_external_maps(
    year: int, historical_dir: Path, kaggle_dir: Optional[str]
) -> Tuple[Dict[str, float], Dict[str, float], Dict]:
    composite: Dict[str, float] = {}
    spread: Dict[str, float] = {}
    multi: Dict = {}

    cache_path = historical_dir / f"external_massey_composite_{year}.json"
    if cache_path.exists():
        try:
            payload = _load_json(cache_path)
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                tid = entry.get("team_id")
                if not tid:
                    continue
                composite[_team_id(str(tid))] = float(entry.get("normalized", float("nan")))
                spread[_team_id(str(tid))] = float(entry.get("spread", float("nan")))
        except Exception as exc:
            logger.warning("Failed to load cached Massey composite for %d: %s", year, exc)

    if kaggle_dir:
        try:
            from src.data.scrapers.external_ratings import ExternalRatingsLoader

            loader = ExternalRatingsLoader(cache_dir=str(historical_dir))
            if not composite:
                n_cached = loader.populate_from_massey_ordinals(kaggle_dir, year)
                if n_cached > 0:
                    ratings = loader.load_all(year)
                    if ratings:
                        comps = loader.compute_composite(ratings)
                        for tid, comp in comps.items():
                            composite[tid] = float(comp.composite_rating)
                            spread[tid] = float(comp.rating_spread)
            multi = loader.load_massey_multi_system(kaggle_dir, year) or {}
        except Exception as exc:
            logger.info("External ratings unavailable for %d: %s", year, exc)

    return composite, spread, multi


def _resolve_team_id(team_id: str, available_ids: Iterable[str]) -> Optional[str]:
    available = set(available_ids)
    if team_id in available:
        return team_id

    matches = [aid for aid in available if aid.startswith(team_id + "_")]
    if len(matches) == 1:
        return matches[0]

    reverse_matches = [aid for aid in available if team_id.startswith(aid + "_")]
    if len(reverse_matches) == 1:
        return reverse_matches[0]

    return None


def _observation_rule(name: str) -> str:
    if name in EXTERNAL_NAN_FEATURES:
        return "non_nan"
    if name in DEFAULT_025_FEATURES:
        return "non_default_0_25"
    return "non_zero"


def _is_observed(name: str, value: float) -> bool:
    if name in EXTERNAL_NAN_FEATURES:
        return not np.isnan(value)
    if name in DEFAULT_025_FEATURES:
        return (not np.isnan(value)) and (not np.isclose(value, 0.25, atol=1e-8))
    return (not np.isnan(value)) and abs(float(value)) > 1e-8


def _selection_cutoff(year: int) -> str:
    return TOURNAMENT_START_DATES.get(year, date(year, 3, 14)).isoformat()


def audit_year(
    year: int,
    historical_dir: Path,
    raw_dir: Path,
    kaggle_dir: Optional[str],
    feature_names: List[str],
) -> Optional[YearAudit]:
    games_path = historical_dir / f"historical_games_{year}.json"
    metrics_path = historical_dir / f"team_metrics_{year}.json"
    if not games_path.exists() or not metrics_path.exists():
        logger.info("Skipping %d: missing games/metrics files", year)
        return None

    seed_map = _load_seed_map(year, historical_dir, raw_dir)
    if not seed_map:
        logger.info("Skipping %d: no tournament seed/bracket data", year)
        return None

    payload = _load_json(games_path)
    team_games_raw = payload.get("team_games", []) if isinstance(payload, dict) else payload
    if not team_games_raw:
        logger.info("Skipping %d: no team_games data", year)
        return None

    game_records = team_games_to_game_records(team_games_raw, year)
    if len(game_records) < 100:
        logger.info("Skipping %d: too few game records (%d)", year, len(game_records))
        return None

    available_ids = {g.team_id for g in game_records} | {g.opponent_id for g in game_records}
    seed_map.update(_build_prefix_aliases(seed_map, available_ids))

    conference_map = _load_conference_map(metrics_path)
    inc_engine = IncrementalMetricsEngine(game_records, conference_map=conference_map, prior_elo=None)

    cutoff = _selection_cutoff(year)
    metrics = inc_engine.compute_as_of(cutoff)

    ff_lookup = TorVikFFLookup(year)
    if ff_lookup.has_snapshots:
        ff_lookup.overlay_metrics(metrics, cutoff)
    else:
        logger.warning("Year %d: no Torvik FF snapshots; proceeding without overlay", year)

    roster_path = historical_dir / f"cbbpy_rosters_{year}.json"
    roster_overlay = load_roster_overlay(str(roster_path), year=year) if roster_path.exists() else {}
    composite_map, spread_map, massey_multi = _load_external_maps(year, historical_dir, kaggle_dir)

    observed_counts = defaultdict(int)
    team_count = 0

    for raw_tid, seed in sorted(seed_map.items()):
        tid = _resolve_team_id(raw_tid, metrics.keys())
        if tid is None:
            logger.debug("Year %d: could not resolve seeded team id %s", year, raw_tid)
            continue
        metric = metrics.get(tid)
        if metric is None:
            continue

        vector = IncrementalMetricsEngine.metrics_to_team_vector(
            metric,
            seed=seed,
            external_rating_composite=composite_map.get(tid, float("nan")),
            external_rating_spread=spread_map.get(tid, float("nan")),
            massey_features=massey_multi.get(tid),
        )
        for idx, val in roster_overlay.get(tid, {}).items():
            vector[idx] = val

        team_count += 1
        for idx, name in enumerate(feature_names):
            if idx >= len(vector):
                break
            if _is_observed(name, float(vector[idx])):
                observed_counts[name] += 1

    if team_count == 0:
        logger.info("Skipping %d: no seeded teams resolved into vectors", year)
        return None

    rates = {name: round(observed_counts[name] / team_count, 4) for name in feature_names}
    return YearAudit(year=year, team_count=team_count, rates=rates)


def write_table(
    audits: List[YearAudit],
    feature_names: List[str],
    cut_years: List[int],
    threshold: float,
    output_csv: Path,
) -> List[str]:
    years = [audit.year for audit in audits]
    year_to_audit = {audit.year: audit for audit in audits}

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cut_features: List[str] = []
    header = (
        ["feature_name", "rule"]
        + [f"rate_{year}" for year in years]
        + ["target_window_rate", "target_window_years", "threshold", "cut"]
    )

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for name in feature_names:
            rule = _observation_rule(name)
            per_year = [year_to_audit[year].rates[name] for year in years]
            target_values = [year_to_audit[year].rates[name] for year in cut_years if year in year_to_audit]
            target_rate = round(float(sum(target_values) / len(target_values)), 4) if target_values else 0.0
            should_cut = target_rate < threshold
            if should_cut:
                cut_features.append(name)
            writer.writerow(
                [name, rule] + per_year + [target_rate, len(target_values), threshold, "CUT" if should_cut else "KEEP"]
            )

    return cut_features


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    feature_names = TeamFeatures.get_feature_names(include_embeddings=False)
    years = list(range(args.start_year, args.end_year + 1))
    cut_years = [year for year in range(args.cut_start_year, args.cut_end_year + 1) if year != 2020]

    audits: List[YearAudit] = []
    for year in years:
        audit = audit_year(
            year=year,
            historical_dir=Path(args.historical_dir),
            raw_dir=Path(args.raw_dir),
            kaggle_dir=args.kaggle_dir,
            feature_names=feature_names,
        )
        if audit is not None:
            audits.append(audit)

    if not audits:
        logger.error("No audit rows produced")
        return 1

    output_csv = Path(args.output_csv)
    cut_features = write_table(
        audits=audits,
        feature_names=feature_names,
        cut_years=cut_years,
        threshold=args.threshold,
        output_csv=output_csv,
    )

    print(f"Wrote {output_csv}")
    print(f"Audited years: {[audit.year for audit in audits]}")
    print(f"Cut threshold years: {cut_years}")
    print(f"Features below threshold ({args.threshold:.2f}): {len(cut_features)}")
    for name in cut_features:
        print(f" - {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
