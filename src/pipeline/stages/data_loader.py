"""Data loading pipeline stage — real implementations.

Extracts data-loading logic from ``SOTAPipeline`` as module-level functions
with explicit parameters.  Each function returns its outputs explicitly;
``SOTAPipeline`` delegates call these and unpacks the results.

Implements Agent Directive V7 S2 Phase 1 (data loading decomposition).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...data.features.feature_engineering import compute_rapm
from ...data.features.proprietary_metrics import (
    IncrementalMetricsEngine,
    ProprietaryMetricsEngine,
    _load_cbbpy_team_map,
    team_games_to_game_records,
    torvik_to_game_records,
)
from ...data.loader import DataLoader
from ...data.models.game_flow import GameFlow
from ...data.models.player import InjuryStatus, Player, Position, Roster
from ...data.normalize import normalize_team_id as _team_id
from ...data.scrapers.bracket_ingestion import BIGDANCE_AVAILABLE
from ...data.scrapers.injury_report import (
    InjuryReportScraper,
    apply_injury_reports_to_roster,
)
from ...conference_tournament.data_enrichment import enrich_torvik_teams
from ...data.scrapers.torvik import BartTorvikScraper
from ...data.scrapers.tournament_context import TournamentContextScraper
from ...exceptions import LeakageError
from ...models.team import Team
from ..config import DataRequirementError, SOTAPipelineConfig, TOURNAMENT_START_DATES
from . import LoadedData, PipelineStage
from .context import PipelineContext
from .game_utils import (
    coerce_game_date,
    is_target_season_game,
    normalize_key,
    validate_source_coverage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses for multi-output functions
# ---------------------------------------------------------------------------


@dataclass
class StatSourcesResult:
    """Outputs produced by :func:`load_team_stat_sources`."""

    torvik_map: Dict[str, Dict] = field(default_factory=dict)
    proprietary_map: Dict[str, Dict] = field(default_factory=dict)
    prior_year_elo: Optional[Dict[str, float]] = None
    current_year_game_records: list = field(default_factory=list)
    current_year_conference_map: Optional[Dict[str, str]] = None
    proprietary_metrics: Dict = field(default_factory=dict)
    torvik_name_to_id: Dict[str, str] = field(default_factory=dict)
    cbbpy_map: Dict[str, str] = field(default_factory=dict)
    mascot_cache: Dict[str, str] = field(default_factory=dict)
    resolve_to_canonical: Optional[Callable] = None


@dataclass
class RosterResult:
    """Outputs produced by :func:`build_rosters`."""

    rosters: Dict[str, Roster] = field(default_factory=dict)
    roster_rapm_quality: Dict[str, float] = field(default_factory=dict)


@dataclass
class GameFlowResult:
    """Outputs produced by :func:`build_or_load_game_flows`."""

    team_to_games: Dict[str, list] = field(default_factory=dict)
    all_game_flows: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure utility functions (no pipeline state)
# ---------------------------------------------------------------------------


def player_from_dict(team_id: str, raw: Dict) -> Player:
    """Convert a raw player dict to a :class:`Player` model object."""
    pos_raw = str(raw.get("position", "PG"))
    if pos_raw not in {p.value for p in Position}:
        pos_raw = "PG"

    injury_raw = str(raw.get("injury_status", "healthy"))
    if injury_raw not in {i.value for i in InjuryStatus}:
        injury_raw = "healthy"

    return Player(
        player_id=str(raw.get("player_id") or f"{team_id}_{raw.get('name', 'player')}"),
        name=str(raw.get("name", "Unknown")),
        team_id=team_id,
        position=Position(pos_raw),
        minutes_per_game=float(raw.get("minutes_per_game", 0.0)),
        games_played=int(raw.get("games_played", 0)),
        games_started=int(raw.get("games_started", 0)),
        rapm_offensive=float(raw.get("rapm_offensive", 0.0)),
        rapm_defensive=float(raw.get("rapm_defensive", 0.0)),
        warp=float(raw.get("warp", 0.0)),
        box_plus_minus=float(raw.get("box_plus_minus", 0.0)),
        usage_rate=float(raw.get("usage_rate", 0.0)),
        injury_status=InjuryStatus(injury_raw),
        is_transfer=bool(raw.get("is_transfer", False)),
        transfer_from=raw.get("transfer_from"),
        eligibility_year=int(raw.get("eligibility_year", 1)),
    )


def apply_transfer_portal_updates(
    rosters: Dict[str, Roster],
    transfer_json_path: str,
) -> None:
    """Apply transfer portal updates to rosters (in-place)."""
    with open(transfer_json_path, "r") as f:
        payload = json.load(f)
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return

    for entry in entries:
        destination = entry.get("destination_team_id") or entry.get("destination_team_name")
        if not destination:
            continue
        team_id = _team_id(str(destination))
        roster = rosters.get(team_id)
        if not roster:
            continue

        player_id = str(entry.get("player_id", "")).strip()
        player_name = str(entry.get("player_name", "")).strip().lower()
        source_team = entry.get("source_team_id") or entry.get("source_team_name")

        for player in roster.players:
            id_match = bool(player_id) and player.player_id == player_id
            name_match = bool(player_name) and player.name.strip().lower() == player_name
            if id_match or name_match:
                player.is_transfer = True
                if source_team:
                    player.transfer_from = str(source_team)
                break


def enrich_roster_rapm(
    players: List[Player],
    team_block: Dict,
    min_rapm_players: int,
) -> None:
    """Enrich player RAPM from stint data or BPM/WARP priors (in-place)."""
    if not players:
        return

    non_zero = sum(1 for p in players if abs(p.rapm_total) > 1e-8)
    if non_zero >= min_rapm_players:
        return

    stints = team_block.get("stints", [])
    if isinstance(stints, list) and stints:
        rapm_map = compute_rapm(players, stints, regularization=0.05)
        for player in players:
            rapm_pair = rapm_map.get(player.player_id)
            if rapm_pair is None:
                continue
            if abs(player.rapm_total) <= 1e-8:
                player.rapm_offensive = float(rapm_pair[0])
                player.rapm_defensive = float(rapm_pair[1])

    # Backfill remaining missing RAPM from BPM/WARP/usage priors.
    for player in players:
        if abs(player.rapm_total) > 1e-8:
            continue
        bpm = float(player.box_plus_minus or 0.0)
        warp_signal = 4.0 * float(player.warp or 0.0)
        usage_signal = (float(player.usage_rate or 0.0) - 20.0) / 25.0
        proxy = 0.6 * bpm + 0.3 * warp_signal + 0.1 * usage_signal
        off_share = 0.6 if float(player.usage_rate or 0.0) >= 20.0 else 0.45
        player.rapm_offensive = proxy * off_share
        player.rapm_defensive = proxy * (1.0 - off_share)


def assess_roster_rapm_quality(
    rosters: Dict[str, Roster],
    min_rapm_players: int,
) -> Dict[str, float]:
    """Compute RAPM quality statistics across all rosters."""
    if not rosters:
        return {"teams": 0.0, "team_coverage_ratio": 0.0, "avg_nonzero_rapm_share": 0.0}

    qualified = 0
    shares: List[float] = []
    for roster in rosters.values():
        player_count = max(len(roster.players), 1)
        non_zero = sum(1 for p in roster.players if abs(p.rapm_total) > 1e-8)
        share = non_zero / player_count
        shares.append(share)
        threshold = min(min_rapm_players, player_count)
        if non_zero >= threshold:
            qualified += 1

    teams = len(rosters)
    return {
        "teams": float(teams),
        "team_coverage_ratio": float(qualified / teams),
        "avg_nonzero_rapm_share": float(np.mean(shares)),
    }


def validate_feed_freshness(
    config: SOTAPipelineConfig,
    source_name: str,
    payload: Dict,
) -> None:
    """Raise if payload timestamp exceeds max feed age."""
    if not config.enforce_feed_freshness:
        return
    if not isinstance(payload, dict):
        return

    from .game_utils import parse_timestamp
    from datetime import datetime

    ts = (
        payload.get("timestamp")
        or payload.get("generated_at")
        or payload.get("updated_at")
        or payload.get("last_updated")
    )
    if not ts:
        raise DataRequirementError(
            f"{source_name} payload missing required timestamp for freshness checks."
        )

    ts_dt = parse_timestamp(ts)
    if ts_dt is None:
        raise DataRequirementError(f"{source_name} timestamp is invalid: {ts}")

    now = datetime.now(ts_dt.tzinfo)
    age_hours = max(0.0, (now - ts_dt).total_seconds() / 3600.0)
    if age_hours > float(config.max_feed_age_hours):
        raise DataRequirementError(
            f"{source_name} feed is stale ({age_hours:.1f}h old, max {config.max_feed_age_hours}h)."
        )


def bracket_data_to_teams(bracket: Any) -> List[Team]:
    """Convert ``TournamentBracketData`` to ``List[Team]``."""
    teams = []
    for bt in bracket.teams:
        team = Team(
            name=bt.display_name,
            seed=bt.seed,
            region=bt.region,
        )
        if bt.rating:
            team.stats["bracket_rating"] = bt.rating
        teams.append(team)
    return teams


# ---------------------------------------------------------------------------
# Data loading functions (previously SOTAPipeline methods)
# ---------------------------------------------------------------------------


def load_teams(
    config: SOTAPipelineConfig,
    bracket_pipeline: Any,
) -> List[Team]:
    """Load tournament teams from configured data source.

    Priority: teams_json > bracket_json > auto bracket fetch.
    """
    if config.teams_json:
        teams = DataLoader.load_teams_from_json(config.teams_json)
        if teams:
            return teams

    if config.bracket_json:
        bracket = bracket_pipeline.fetch(source=config.bracket_json)
        return bracket_data_to_teams(bracket)

    if config.bracket_source != "auto" or BIGDANCE_AVAILABLE:
        try:
            bracket = bracket_pipeline.fetch(source=config.bracket_source)
            if bracket.resolution_warnings:
                for w in bracket.resolution_warnings:
                    logger.warning("Bracket name resolution: %s", w)
            saved_path = bracket_pipeline.save(bracket)
            logger.info("Bracket saved to %s", saved_path)
            return bracket_data_to_teams(bracket)
        except Exception as _bracket_exc:
            logger.warning("Bracket auto-fetch failed: %s", _bracket_exc)

    raise DataRequirementError(
        "Missing teams dataset. Provide --input teams JSON, --bracket-json, "
        "or install bigdance for live bracket ingestion."
    )


def compute_prior_year_elo(config: SOTAPipelineConfig) -> Optional[Dict[str, float]]:
    """Compute end-of-season Elo for year before ``config.year``.

    Returns ``None`` if prior-year data is unavailable — the pipeline
    degrades gracefully to the flat-1500 baseline.
    """
    prior_year = config.year - 1
    if prior_year == 2020:
        prior_year = 2019  # Skip COVID-cancelled season

    candidates = []
    if config.multi_year_games_dir and config.multi_year_games_dir != "auto":
        candidates.append(
            os.path.join(config.multi_year_games_dir, f"historical_games_{prior_year}.json")
        )
    auto_dir = os.path.join(os.getcwd(), "data", "raw", "historical")
    candidates.append(os.path.join(auto_dir, f"historical_games_{prior_year}.json"))

    games_path = None
    for c in candidates:
        if os.path.isfile(c):
            games_path = c
            break

    if games_path is None:
        logger.info(
            "Prior-year Elo: no historical data for %d — using flat 1500 baseline.",
            prior_year,
        )
        return None

    try:
        with open(games_path, "r") as f:
            payload = json.load(f)

        team_games_raw = payload.get("team_games", []) if isinstance(payload, dict) else []
        if not team_games_raw:
            logger.info("Prior-year Elo: year %d has no box-score data.", prior_year)
            return None

        game_records = team_games_to_game_records(team_games_raw, prior_year)
        if len(game_records) < 100:
            logger.info(
                "Prior-year Elo: year %d has too few games (%d).", prior_year, len(game_records)
            )
            return None

        engine = IncrementalMetricsEngine(game_records, conference_map={}, prior_elo=None)
        end_elo = engine.get_end_of_season_elo()

        if not end_elo:
            logger.info("Prior-year Elo: year %d produced no Elo data.", prior_year)
            return None

        return end_elo

    except Exception as e:
        logger.warning("Prior-year Elo: failed to compute for %d: %s", prior_year, e)
        return None


def enrich_tournament_context(
    config: SOTAPipelineConfig,
    torvik_map: Dict[str, Dict],
    proprietary_map: Dict[str, Dict],
    teams: List[Team],
) -> None:
    """Load AP rankings, coach experience, conference champions and inject
    into ``torvik_map`` / ``proprietary_map`` in-place."""
    # --- 1. Preseason AP rankings ---
    ap_rankings: Dict[str, int] = {}
    if config.preseason_ap_json:
        ap_rankings = TournamentContextScraper.load_preseason_ap_from_json(
            config.preseason_ap_json
        )

    # --- 2. Coach tournament experience ---
    coach_data: Dict[str, Dict] = {}
    if config.coach_tournament_json:
        coach_data = TournamentContextScraper.load_coach_data_from_json(
            config.coach_tournament_json
        )

    if coach_data and config.year < 2026:
        if config.coach_data_cutoff_year is None:
            msg = (
                "Coach tournament data contains career totals that may "
                "include future years.  Zeroing coach features for "
                f"year={config.year} to prevent leakage.  Set "
                "coach_data_cutoff_year to suppress this guard."
            )
            if config.strict_leakage_mode:
                raise LeakageError(msg)
            logger.warning(msg)
            coach_data = {}
        else:
            logger.info(
                "Coach data loaded with cutoff_year=%d for year=%d",
                config.coach_data_cutoff_year,
                config.year,
            )

    # --- 3. Conference tournament champions ---
    conf_champions: Dict[str, str] = {}
    if config.conf_champions_json:
        conf_champions = TournamentContextScraper.load_conf_champions_from_json(
            config.conf_champions_json
        )

    if not ap_rankings and not coach_data and not conf_champions:
        return

    # Build team→coach map from roster JSON if available
    team_to_coach_map: Dict[str, str] = {}
    if config.roster_json:
        try:
            with open(config.roster_json, "r") as f:
                roster_payload = json.load(f)
            for team_block in roster_payload.get("teams", []):
                tid = _team_id(
                    str(team_block.get("team_id") or team_block.get("team_name") or "")
                )
                coach = team_block.get("coach") or team_block.get("head_coach") or ""
                if tid and coach:
                    team_to_coach_map[tid] = str(coach)
        except Exception as _coach_exc:
            logger.debug("Coach map extraction failed: %s", _coach_exc)

    # Map teams to coach appearances + win rate
    coach_appearances_by_team: Dict[str, int] = {}
    coach_win_rate_by_team: Dict[str, float] = {}
    if coach_data and team_to_coach_map:
        ctx = TournamentContextScraper()
        coach_appearances_by_team = ctx.build_team_to_coach_appearances(
            coach_data, team_to_coach_map
        )
        coach_win_rate_by_team = ctx.build_team_to_coach_win_rate(
            coach_data, team_to_coach_map
        )

    # Inject values into maps for each team
    for team in teams:
        team_id = _team_id(team.name)
        norm_name = normalize_key(team_id)

        # AP rank
        ap_rank = 0
        if ap_rankings:
            ap_rank = ap_rankings.get(norm_name, 0)
            if not ap_rank:
                for ap_key, rank_val in ap_rankings.items():
                    if norm_name in ap_key or ap_key in norm_name:
                        ap_rank = rank_val
                        break

        # Coach tournament appearances + win rate + stage experience
        coach_apps = coach_appearances_by_team.get(team_id, 0)
        coach_win_rate = coach_win_rate_by_team.get(team_id, 0.0)

        coach_deep_run_rate = 0.0
        coach_stage_consistency = 0.0
        coach_f4_apps = 0
        coach_e8_apps = 0
        coach_s16_apps = 0
        if coach_data and team_to_coach_map:
            coach_name = team_to_coach_map.get(team_id, "")
            raw_coach = coach_data.get(coach_name, {})
            if not raw_coach:
                last = coach_name.split()[-1].lower() if coach_name else ""
                for ck, cv in coach_data.items():
                    if last and last in ck.lower():
                        raw_coach = cv
                        break
            if raw_coach:
                from ...data.features.tournament_features import compute_coach_tournament_power

                ctp = compute_coach_tournament_power(raw_coach)
                coach_deep_run_rate = ctp.deep_run_rate
                coach_stage_consistency = ctp.stage_consistency
                coach_f4_apps = ctp.final_fours
                coach_e8_apps = ctp.elite_8_appearances
                coach_s16_apps = ctp.sweet_16_appearances

        # Conference tournament champion
        is_conf_champ = 0.0
        if conf_champions:
            if norm_name in conf_champions:
                is_conf_champ = 1.0
            else:
                for champ_key in conf_champions:
                    if norm_name in champ_key or champ_key in norm_name:
                        is_conf_champ = 1.0
                        break

        # Write into torvik_map
        if team_id in torvik_map:
            torvik_map[team_id]["preseason_ap_rank"] = ap_rank
            torvik_map[team_id]["coach_tournament_appearances"] = coach_apps
            torvik_map[team_id]["coach_tournament_win_rate"] = coach_win_rate
            torvik_map[team_id]["coach_deep_run_rate"] = coach_deep_run_rate
            torvik_map[team_id]["coach_stage_consistency"] = coach_stage_consistency
            torvik_map[team_id]["coach_f4_appearances"] = coach_f4_apps
            torvik_map[team_id]["coach_e8_appearances"] = coach_e8_apps
            torvik_map[team_id]["coach_s16_appearances"] = coach_s16_apps
            torvik_map[team_id]["conf_tourney_champion"] = is_conf_champ

        # Write into proprietary_map
        if team_id in proprietary_map:
            proprietary_map[team_id]["preseason_ap_rank"] = ap_rank
            proprietary_map[team_id]["coach_tournament_appearances"] = coach_apps
            proprietary_map[team_id]["coach_tournament_win_rate"] = coach_win_rate
            proprietary_map[team_id]["coach_deep_run_rate"] = coach_deep_run_rate
            proprietary_map[team_id]["coach_stage_consistency"] = coach_stage_consistency
            proprietary_map[team_id]["coach_f4_appearances"] = coach_f4_apps
            proprietary_map[team_id]["coach_e8_appearances"] = coach_e8_apps
            proprietary_map[team_id]["coach_s16_appearances"] = coach_s16_apps
            proprietary_map[team_id]["conf_tourney_champion"] = is_conf_champ


def load_team_stat_sources(
    config: SOTAPipelineConfig,
    teams: List[Team],
    proprietary_engine: ProprietaryMetricsEngine,
) -> StatSourcesResult:
    """Load Torvik stats and compute proprietary metrics for all teams.

    Returns a :class:`StatSourcesResult` containing torvik_map,
    proprietary_map, and all side-effect state needed by the pipeline.
    """
    # --- Load Torvik data ---
    if config.torvik_json:
        with open(config.torvik_json, "r") as f:
            torvik_payload = json.load(f)
        validate_feed_freshness(config, "Torvik", torvik_payload)
        torvik_teams = BartTorvikScraper().load_from_json(config.torvik_json)
    elif config.scrape_live:
        torvik_teams = BartTorvikScraper(cache_dir=config.data_cache_dir).fetch_current_rankings(
            config.year
        )
    else:
        raise DataRequirementError(
            "Missing Torvik data. Provide --torvik JSON or run with --scrape-live."
        )

    if not torvik_teams:
        raise DataRequirementError("Torvik data source is empty.")

    # --- Enrich Torvik data with Four Factors if missing ---
    # The main torvik CSV doesn't include Four Factors; they come from
    # separate files (torvik_four_factors_YYYY.json, torvik_shooting_YYYY.json).
    # If these fields are zero, attempt enrichment from those files.
    _sample_efg = [
        getattr(t, "effective_fg_pct", 0.0) or 0.0 for t in torvik_teams[:20]
    ]
    if all(abs(v) < 1e-6 for v in _sample_efg):
        _data_dir = os.path.dirname(config.torvik_json) if config.torvik_json else "data/raw"
        logger.warning(
            "Four Factors are all zero in Torvik data — running enrichment "
            "from %s/torvik_four_factors_%d.json", _data_dir, config.year,
        )
        with open(config.torvik_json, "r") as f:
            _torvik_payload = json.load(f)
        _torvik_payload = enrich_torvik_teams(_torvik_payload, data_dir=_data_dir, year=config.year)
        # Re-parse enriched data
        torvik_teams = [
            BartTorvikScraper()._dict_to_team(t)
            for t in _torvik_payload.get("teams", [])
        ]
        # Check if enrichment succeeded
        _sample_efg_after = [
            getattr(t, "effective_fg_pct", 0.0) or 0.0 for t in torvik_teams[:20]
        ]
        if all(abs(v) < 1e-6 for v in _sample_efg_after):
            raise DataRequirementError(
                "Four Factors (eFG%, TO%, ORB%, FTR) are ALL ZERO for every team "
                "even after enrichment. The torvik_four_factors_{}.json file is "
                "missing or empty. Re-run data collection to fetch Four Factors "
                "from barttorvik.com/getadvstats.php.".format(config.year)
            )
        else:
            logger.info(
                "Four Factors enrichment succeeded: %d/%d teams now have nonzero eFG%%.",
                sum(1 for v in _sample_efg_after if abs(v) > 1e-6),
                len(_sample_efg_after),
            )

    # --- Load historical games for proprietary metrics ---
    historical_games: List[Dict] = []
    if config.historical_games_json:
        with open(config.historical_games_json, "r") as f:
            hist_payload = json.load(f)
        validate_feed_freshness(config, "Historical games", hist_payload)
        historical_games = hist_payload.get("games", [])
        _fallback_date = f"{config.year - 1}-11-01"
        _fb_count = sum(1 for _g in historical_games if _g.get("date") == _fallback_date)
        if _fb_count > len(historical_games) * 0.5 and len(historical_games) > 50:
            raise DataRequirementError(
                f"Historical data for season {config.year} has "
                f"{_fb_count}/{len(historical_games)} games with fallback "
                f"date {_fallback_date}. Game dates are corrupted. "
                f"Run `python -m src.main repair-dates` to fix."
            )
    elif config.scrape_live:
        historical_games = []
    if not historical_games:
        raise DataRequirementError(
            "Missing historical game data. Provide --historical-games JSON with box-score rows."
        )

    # --- Build conference map from Torvik data ---
    torvik_teams_dicts = []
    conference_map: Dict[str, str] = {}
    for tv in torvik_teams:
        d = tv.to_dict() if hasattr(tv, "to_dict") else tv
        torvik_teams_dicts.append(d)
        tid = normalize_key(d.get("team_id", ""))
        conf = d.get("conference", "")
        if tid and conf:
            conference_map[tid] = conf

    # --- Compute prior-year Elo ---
    prior_year_elo = compute_prior_year_elo(config)
    if prior_year_elo:
        proprietary_engine._elo_prior = prior_year_elo
        logger.info(
            "Cross-season Elo: loaded %d team priors from year %d.",
            len(prior_year_elo),
            config.year - 1,
        )

    # --- Compute proprietary metrics ---
    from datetime import date as _date
    _t_start = TOURNAMENT_START_DATES.get(config.year, _date(config.year, 3, 14))
    pre_tournament_cutoff = _t_start.isoformat()
    game_records = torvik_to_game_records(
        torvik_teams_dicts,
        historical_games,
        season_year=config.year,
    )
    current_year_game_records = game_records
    current_year_conference_map = conference_map if conference_map else None
    proprietary_results = proprietary_engine.compute(
        game_records,
        conference_map=conference_map if conference_map else None,
        cutoff_date=pre_tournament_cutoff,
    )

    # --- Build index maps ---
    def _normalize_entry(entry, id_keys, name_keys):
        value = ""
        if isinstance(entry, dict):
            for key in id_keys:
                if key in entry and entry[key]:
                    value = entry[key]
                    break
        else:
            for key in id_keys:
                value = getattr(entry, key, None) or value
                if value:
                    break
        if not value:
            for key in name_keys:
                if isinstance(entry, dict):
                    value = entry.get(key, value)
                    if value:
                        break
                else:
                    value = getattr(entry, key, value)
                    if value:
                        break
        return normalize_key(value)

    torvik_index = {_normalize_entry(t, ["team_id"], ["name"]): t for t in torvik_teams}

    torvik_map: Dict[str, Dict] = {}
    proprietary_map: Dict[str, Dict] = {}

    # Build Torvik name→canonical_id lookup
    _torvik_name_to_id: Dict[str, str] = {}
    for t in torvik_teams:
        if isinstance(t, dict):
            tid = t.get("team_id", "")
            tname = t.get("name", "")
        else:
            tid = getattr(t, "team_id", "")
            tname = getattr(t, "name", "")
        if tid and tname:
            canon = normalize_key(tid)
            _torvik_name_to_id[normalize_key(_team_id(tname))] = canon
            _torvik_name_to_id[canon] = canon
            cleaned = tname.replace("&amp;", "&")
            if cleaned != tname:
                _torvik_name_to_id[normalize_key(_team_id(cleaned))] = canon
            stripped = re.sub(r"\s*\([^)]*\)\s*", "", tname).strip()
            if stripped != tname:
                _torvik_name_to_id[normalize_key(_team_id(stripped))] = canon
                stripped_clean = re.sub(r"\s*\([^)]*\)\s*", "", cleaned).strip()
                if stripped_clean != stripped:
                    _torvik_name_to_id[normalize_key(_team_id(stripped_clean))] = canon

    # CBBpy→Torvik alias overrides
    _cbbpy_torvik_aliases = {
        "mcneese": "mcneese_state",
        "american_university": "american",
    }
    for alias, target in _cbbpy_torvik_aliases.items():
        if target in _torvik_name_to_id:
            _torvik_name_to_id[alias] = _torvik_name_to_id[target]

    _torvik_id_set = set(_torvik_name_to_id.values())

    _cbbpy_map = _load_cbbpy_team_map()
    _mascot_cache: Dict[str, str] = {}

    def _resolve_to_canonical(raw_id: str, display_name: str = "") -> str:
        if raw_id in _mascot_cache:
            return _mascot_cache[raw_id]
        if display_name:
            location = _cbbpy_map.get(display_name)
            if location:
                norm_loc = normalize_key(_team_id(location))
                canon = _torvik_name_to_id.get(norm_loc)
                if canon:
                    _mascot_cache[raw_id] = canon
                    return canon
        if raw_id in _torvik_id_set:
            _mascot_cache[raw_id] = raw_id
            return raw_id
        _mascot_cache[raw_id] = raw_id
        return raw_id

    for team in teams:
        tid = _team_id(team.name)
        key = normalize_key(tid)

        tv = torvik_index.get(key)
        if tv:
            if isinstance(tv, dict):
                data = tv
            else:
                data = tv.to_dict()
            torvik_map[tid] = {
                "effective_fg_pct": data.get("effective_fg_pct", 0.5),
                "turnover_rate": data.get("turnover_rate", 0.18),
                "offensive_reb_rate": data.get("offensive_reb_rate", 0.30),
                "free_throw_rate": data.get("free_throw_rate", 0.30),
                "opp_effective_fg_pct": data.get("opp_effective_fg_pct", 0.5),
                "opp_turnover_rate": data.get("opp_turnover_rate", 0.18),
                "defensive_reb_rate": data.get("defensive_reb_rate", 0.70),
                "opp_free_throw_rate": data.get("opp_free_throw_rate", 0.30),
                "adj_offensive_efficiency": data.get("adj_offensive_efficiency", 100.0),
                "adj_defensive_efficiency": data.get("adj_defensive_efficiency", 100.0),
                "adj_tempo": data.get("adj_tempo", 68.0),
                "barthag": data.get("barthag", 0.5),
                "t_rank": data.get("t_rank", 999),
                "two_pt_pct": data.get("two_pt_pct", 0.0),
                "three_pt_pct": data.get("three_pt_pct", 0.0),
                "three_pt_rate": data.get("three_pt_rate", 0.0),
                "ft_pct": data.get("ft_pct", 0.0),
                "opp_two_pt_pct": data.get("opp_two_pt_pct", 0.0),
                "opp_three_pt_pct": data.get("opp_three_pt_pct", 0.0),
                "opp_three_pt_rate": data.get("opp_three_pt_rate", 0.0),
                "wab": data.get("wab", 0.0),
                "wins": data.get("wins", 0),
                "losses": data.get("losses", 0),
                "conference": data.get("conference", ""),
                "conf_wins": data.get("conf_wins", 0),
                "conf_losses": data.get("conf_losses", 0),
            }

        pm = proprietary_results.get(key)
        if pm is not None:
            proprietary_map[tid] = pm.to_dict()
        else:
            pm = proprietary_results.get(tid)
            if pm is not None:
                proprietary_map[tid] = pm.to_dict()

    # Backfill from Sports Reference if available
    if config.sports_reference_json:
        with open(config.sports_reference_json, "r") as f:
            sr_payload = json.load(f)
        sr_rows = sr_payload.get("teams", [])

        _sr_off = [float(r.get("off_rtg", 0)) for r in sr_rows if isinstance(r, dict)]
        if _sr_off and all(abs(v) < 1e-6 for v in _sr_off):
            logger.warning(
                "Sports Reference JSON has all-zero off_rtg — skipping "
                "entire SR backfill (corrupted scrape)."
            )
            sr_rows = []

        sr_index = {}
        for row in sr_rows:
            team_name = row.get("team_name") or row.get("name")
            if team_name:
                sr_index[normalize_key(_team_id(str(team_name)))] = row

        for team in teams:
            tid = _team_id(team.name)
            key = normalize_key(tid)
            sr = sr_index.get(key)
            if not sr:
                continue

            if tid not in proprietary_map:
                off = float(sr.get("off_rtg", 0))
                deff = float(sr.get("def_rtg", 0))
                pace = float(sr.get("pace", 0))
                if off < 1e-6 or deff < 1e-6:
                    continue
                proprietary_map[tid] = {
                    "adj_offensive_efficiency": off,
                    "adj_defensive_efficiency": deff,
                    "adj_tempo": pace if pace > 1e-6 else 68.0,
                    "adj_efficiency_margin": off - deff,
                    "sos_adj_em": 0.0,
                    "sos_opp_o": 100.0,
                    "sos_opp_d": 100.0,
                    "ncsos_adj_em": 0.0,
                    "luck": 0.0,
                }

    # Enrich with tournament context data
    enrich_tournament_context(config, torvik_map, proprietary_map, teams)

    validate_source_coverage("Torvik", torvik_map, teams, min_ratio=0.8)
    validate_source_coverage("Proprietary metrics", proprietary_map, teams, min_ratio=0.8)

    return StatSourcesResult(
        torvik_map=torvik_map,
        proprietary_map=proprietary_map,
        prior_year_elo=prior_year_elo,
        current_year_game_records=current_year_game_records,
        current_year_conference_map=current_year_conference_map,
        proprietary_metrics=proprietary_results,
        torvik_name_to_id=_torvik_name_to_id,
        cbbpy_map=_cbbpy_map,
        mascot_cache=_mascot_cache,
        resolve_to_canonical=_resolve_to_canonical,
    )


def apply_injury_reports(
    config: SOTAPipelineConfig,
    rosters: Dict[str, Roster],
    positional_depth_chart: Any,
    injury_severity_model: Any,
) -> Tuple[Dict, Dict[str, Dict[str, float]]]:
    """Load injury reports and apply severity modeling + positional depth.

    Returns:
        Tuple of (stats dict, positional_impacts dict).
    """
    stats: Dict = {
        "injury_report_loaded": False,
        "players_updated": 0,
        "teams_with_injuries": 0,
        "severity_model_enabled": config.enable_injury_severity_model,
        "positional_depth_enabled": config.enable_positional_depth,
    }
    positional_impacts: Dict[str, Dict[str, float]] = {}

    if config.injury_report_json:
        scraper = InjuryReportScraper(cache_dir=config.data_cache_dir)
        team_reports = scraper.load_from_json(config.injury_report_json)

        total_updated = 0
        teams_injured = 0
        for team_id, roster in rosters.items():
            norm_id = normalize_key(team_id)
            report = team_reports.get(team_id) or team_reports.get(norm_id)
            if report is None:
                for rk, rv in team_reports.items():
                    if normalize_key(rk) == norm_id:
                        report = rv
                        break

            if report is not None:
                updated = apply_injury_reports_to_roster(roster, report)
                total_updated += updated
                if report.has_injuries:
                    teams_injured += 1

        stats["injury_report_loaded"] = True
        stats["players_updated"] = total_updated
        stats["teams_with_injuries"] = teams_injured

    # Positional depth analysis
    if config.enable_positional_depth:
        for team_id, roster in rosters.items():
            impacts = positional_depth_chart.compute_injury_impact(
                roster,
                severity_model=(
                    injury_severity_model if config.enable_injury_severity_model else None
                ),
            )
            positional_impacts[team_id] = impacts

        if positional_impacts:
            avg_vulnerability = float(
                np.mean(
                    [v.get("positional_vulnerability", 0.0) for v in positional_impacts.values()]
                )
            )
            stats["avg_positional_vulnerability"] = round(avg_vulnerability, 4)

    return stats, positional_impacts


def build_rosters(
    config: SOTAPipelineConfig,
    teams: List[Team],
) -> RosterResult:
    """Load rosters from JSON and enrich with RAPM.

    Returns a :class:`RosterResult` containing rosters and quality stats.
    """
    if not config.roster_json:
        raise DataRequirementError(
            "Missing roster data. Provide --rosters JSON with player-level metrics."
        )

    with open(config.roster_json, "r") as f:
        payload = json.load(f)
    validate_feed_freshness(config, "Rosters", payload)

    teams_payload = payload.get("teams", [])
    if not isinstance(teams_payload, list):
        raise DataRequirementError("Invalid roster JSON: expected top-level 'teams' list.")

    rosters: Dict[str, Roster] = {}
    for team_block in teams_payload:
        source_team = (
            team_block.get("team_id") or team_block.get("team_name") or team_block.get("name")
        )
        if not source_team:
            continue
        tid = _team_id(str(source_team))
        players_raw = team_block.get("players", [])
        players: List[Player] = []
        for player_data in players_raw:
            players.append(player_from_dict(tid, player_data))
        enrich_roster_rapm(players, team_block, config.min_rapm_players_per_team)
        if players:
            rosters[tid] = Roster(team_id=tid, players=players)

    if config.transfer_portal_json:
        apply_transfer_portal_updates(rosters, config.transfer_portal_json)

    quality = assess_roster_rapm_quality(rosters, config.min_rapm_players_per_team)
    if quality.get("team_coverage_ratio", 0.0) < 0.8:
        raise DataRequirementError(
            "Roster RAPM quality is too low. Provide richer player RAPM/stint inputs "
            f"(coverage={quality.get('team_coverage_ratio', 0.0):.1%})."
        )
    validate_source_coverage("Roster", rosters, teams, min_ratio=0.8)
    return RosterResult(rosters=rosters, roster_rapm_quality=quality)


def historical_game_to_flow(
    game: Dict,
    config_year: int,
    resolve_to_canonical: Optional[Callable] = None,
) -> Optional[GameFlow]:
    """Convert a raw game dict to a :class:`GameFlow`."""
    game_id = str(game.get("game_id") or game.get("id") or "")
    t1 = (
        game.get("team_id")
        or game.get("team1_id")
        or game.get("team1")
        or game.get("home_team")
    )
    t2 = (
        game.get("opponent_id")
        or game.get("team2_id")
        or game.get("team2")
        or game.get("away_team")
    )
    if not game_id or not t1 or not t2:
        return None

    raw1 = _team_id(str(t1))
    raw2 = _team_id(str(t2))
    if resolve_to_canonical is not None:
        disp1 = str(game.get("team_name") or game.get("team1_name") or "")
        disp2 = str(game.get("opponent_name") or game.get("team2_name") or "")
        team1_id = resolve_to_canonical(raw1, display_name=disp1)
        team2_id = resolve_to_canonical(raw2, display_name=disp2)
    else:
        team1_id = raw1
        team2_id = raw2

    flow = GameFlow(game_id=game_id, team1_id=team1_id, team2_id=team2_id)

    lead_history = game.get("lead_history")
    if isinstance(lead_history, list) and lead_history:
        flow.lead_history = [int(x) for x in lead_history]
    else:
        s1 = int(game.get("team1_score", game.get("home_score", 0)))
        s2 = int(game.get("team2_score", game.get("away_score", 0)))
        flow.lead_history = [0, s1 - s2]

    raw_date = game.get("game_date") or game.get("date") or game.get("start_date")
    fallback_year = infer_game_year(game, config_year)
    flow.game_date = coerce_game_date(
        raw_date,
        fallback_year=fallback_year,
        game_id=game_id,
        source="historical_game",
    )
    neutral = bool(game.get("neutral_site", False))
    flow.location_weight = 0.5 if neutral else 1.0
    return flow


def infer_game_year(game: Dict, default_year: int) -> int:
    """Infer the season year from game metadata."""
    for key in ("season", "season_year", "year"):
        value = game.get(key)
        if isinstance(value, int) and 1900 <= value <= 2100:
            return value
        if isinstance(value, str):
            match = re.search(r"(19|20)\d{2}", value)
            if match:
                return int(match.group(0))
    return default_year


def build_or_load_game_flows(
    config: SOTAPipelineConfig,
    teams: List[Team],
    resolve_to_canonical: Optional[Callable] = None,
    mascot_cache: Optional[Dict[str, str]] = None,
) -> GameFlowResult:
    """Load game flows from historical games JSON.

    Returns a :class:`GameFlowResult` with per-team game lists and all flows.
    """
    team_to_games: Dict[str, list] = {_team_id(t.name): [] for t in teams}
    all_flows_dict: Dict[str, GameFlow] = {}

    if config.historical_games_json:
        with open(config.historical_games_json, "r") as f:
            payload = json.load(f)
        games = payload.get("games", [])

        # Pre-scan games to populate canonical ID cache
        if resolve_to_canonical is not None and mascot_cache is not None:
            for game in games:
                if not isinstance(game, dict):
                    continue
                for id_keys, name_keys in [
                    (["team_id", "team1_id"], ["team_name", "team1_name"]),
                    (["opponent_id", "team2_id"], ["opponent_name", "team2_name"]),
                ]:
                    raw = ""
                    for k in id_keys:
                        if game.get(k):
                            raw = _team_id(str(game[k]))
                            break
                    disp = ""
                    for k in name_keys:
                        if game.get(k):
                            disp = str(game[k])
                            break
                    if raw and disp and raw not in mascot_cache:
                        resolve_to_canonical(raw, display_name=disp)

        for game in games:
            flow = historical_game_to_flow(game, config.year, resolve_to_canonical)
            if not flow:
                continue
            all_flows_dict[flow.game_id] = flow
    else:
        raise DataRequirementError(
            "Missing game-level data. Provide --historical-games JSON."
        )

    in_season_flows = {
        game_id: flow
        for game_id, flow in all_flows_dict.items()
        if is_target_season_game(str(getattr(flow, "game_date", "")), config.year)
    }
    if not in_season_flows:
        raise DataRequirementError(
            f"No game-level rows found for target season {config.year}. "
            "Expected games from the 2025-26 window for a 2026 run."
        )

    for flow in in_season_flows.values():
        if flow.team1_id in team_to_games:
            team_to_games[flow.team1_id].append(flow)
        if flow.team2_id in team_to_games:
            team_to_games[flow.team2_id].append(flow)

    all_game_flows = list(in_season_flows.values())

    validate_source_coverage(
        "Historical games",
        {k: v for k, v in team_to_games.items() if v},
        teams,
        min_ratio=0.6,
    )
    return GameFlowResult(team_to_games=team_to_games, all_game_flows=all_game_flows)


def load_external_ratings(
    config: SOTAPipelineConfig,
    teams: List[Team],
) -> Dict:
    """Load external rating composites (Massey Ordinals, etc.).

    Returns dict of {team_id: CompositeRating} or empty dict.
    """
    if not config.enable_external_ratings:
        return {}

    try:
        from ...data.scrapers.external_ratings import ExternalRatingsLoader
    except ImportError:
        return {}

    year = config.year
    cache_dir = config.external_ratings_dir or config.data_cache_dir

    loader = ExternalRatingsLoader(cache_dir=cache_dir)

    # Step 1: Populate from Kaggle Massey Ordinals
    if config.kaggle_dir:
        try:
            n = loader.populate_from_massey_ordinals(config.kaggle_dir, year)
            if n > 0:
                logger.info("Loaded %d Massey Ordinal systems from Kaggle", n)
            else:
                logger.warning(
                    "Massey Ordinals: kaggle_dir=%s set but 0 systems cached. "
                    "Check that MMasseyOrdinals.csv exists and has data for season %d.",
                    config.kaggle_dir,
                    year,
                )
        except Exception as e:
            logger.warning(
                "Massey Ordinals ingestion failed (kaggle_dir=%s): %s. "
                "Falling back to cached ratings or seed-based estimates.",
                config.kaggle_dir,
                e,
            )

    # Step 2: Load all cached external rating systems
    all_ratings = loader.load_all(year)

    if all_ratings:
        composites = loader.compute_composite(all_ratings)
        n_systems = len(all_ratings)
        n_teams = len(composites)
        has_massey = "massey_composite" in all_ratings
        logger.info(
            "External ratings: %d systems, %d teams composited (massey_composite=%s)",
            n_systems,
            n_teams,
            "present" if has_massey else "MISSING",
        )
        if not has_massey and config.kaggle_dir:
            logger.warning(
                "massey_composite not in loaded systems despite kaggle_dir "
                "being set. Systems found: %s. This indicates a data "
                "loading issue — predictions will lack Massey signal.",
                list(all_ratings.keys()),
            )
        return composites

    # Step 3: Fallback to seed-based estimates
    logger.warning(
        "No cached external ratings found for year %d. Using seed-based "
        "fallback. This is significantly less accurate than Massey Ordinals "
        "(-0.008 to -0.015 Brier). Set kaggle_dir or run: "
        "python -m src.data.kaggle_downloader",
        year,
    )
    seed_map = {}
    for team in teams:
        tid = _team_id(team.name)
        seed_map[tid] = team.seed
    if seed_map:
        composites = loader.generate_from_seeds(seed_map)
        logger.info("External ratings: seed-based fallback for %d teams", len(composites))
        return composites

    return {}


def verify_massey_coverage(
    teams: List[Team],
    composites: Dict,
    team_features: Dict,
) -> Dict:
    """Verify that Massey Ordinals composites flow through with sufficient coverage.

    Returns dict of coverage statistics for the pipeline report.
    """
    n_teams = len(teams)
    n_with_composite = 0
    n_with_spread = 0
    n_seed_only = 0
    n_multi_system = 0
    composite_values = []
    spread_values = []
    missing_teams = []

    for team in teams:
        tid = _team_id(team.name)
        comp = composites.get(tid)
        if comp is None:
            missing_teams.append(team.name)
            continue
        n_with_composite += 1
        composite_values.append(comp.composite_rating)

        if comp.rating_spread > 0:
            n_with_spread += 1
            spread_values.append(comp.rating_spread)

        if hasattr(comp, "n_systems"):
            if comp.n_systems <= 1:
                n_seed_only += 1
            else:
                n_multi_system += 1

    coverage_pct = n_with_composite / max(n_teams, 1)

    # Verify feature vectors contain external ratings
    n_vec_nonzero = 0
    for team in teams:
        tid = _team_id(team.name)
        tf = team_features.get(tid)
        if tf is not None and abs(tf.external_rating_composite) > 1e-8:
            n_vec_nonzero += 1
    vec_coverage_pct = n_vec_nonzero / max(n_teams, 1)

    stats = {
        "n_teams": n_teams,
        "n_with_composite": n_with_composite,
        "coverage_pct": round(coverage_pct, 3),
        "n_multi_system": n_multi_system,
        "n_seed_only_fallback": n_seed_only,
        "n_missing": len(missing_teams),
        "feature_vector_coverage_pct": round(vec_coverage_pct, 3),
    }

    if composite_values:
        stats["composite_mean"] = round(float(np.mean(composite_values)), 4)
        stats["composite_std"] = round(float(np.std(composite_values)), 4)
    if spread_values:
        stats["spread_mean"] = round(float(np.mean(spread_values)), 4)

    if coverage_pct < 0.50:
        logger.warning(
            "FIX-MASSEY CRITICAL: Only %.0f%% of tournament teams have "
            "external rating composites (%d/%d).",
            coverage_pct * 100,
            n_with_composite,
            n_teams,
        )
        if missing_teams:
            logger.warning("FIX-MASSEY: Missing teams (first 10): %s", missing_teams[:10])
    elif coverage_pct < 0.90:
        logger.warning(
            "FIX-MASSEY: %.0f%% coverage (%d/%d teams). "
            "%d teams using seed-based fallback.",
            coverage_pct * 100,
            n_with_composite,
            n_teams,
            n_seed_only,
        )
    else:
        logger.info(
            "FIX-MASSEY: External ratings coverage %.0f%% (%d/%d teams, "
            "%d multi-system, %d seed-fallback). Feature vector propagation: %.0f%%.",
            coverage_pct * 100,
            n_with_composite,
            n_teams,
            n_multi_system,
            n_seed_only,
            vec_coverage_pct * 100,
        )

    if vec_coverage_pct < coverage_pct * 0.8 and coverage_pct > 0.5:
        logger.warning(
            "FIX-MASSEY: Feature vector propagation gap — "
            "%.0f%% of teams have composites but only %.0f%% have "
            "non-zero external_rating_composite in feature vectors.",
            coverage_pct * 100,
            vec_coverage_pct * 100,
        )

    return stats


# ---------------------------------------------------------------------------
# Stage class (wrapper for StageRegistry compatibility)
# ---------------------------------------------------------------------------


class DataLoadingStage:
    """Load all raw data for the pipeline.

    Wraps the module-level functions above as a :class:`PipelineStage`.
    """

    name = "data_loading"

    def run(self, ctx: PipelineContext, pipeline: Any) -> LoadedData:
        """Execute data loading and return a typed contract."""
        timer = ctx.resource_tracker or ctx.phase_timer
        phase_ctx = timer.phase(self.name) if timer else _noop_ctx()

        with phase_ctx:
            teams = load_teams(pipeline.config, pipeline.bracket_pipeline)

            stat_result = load_team_stat_sources(
                pipeline.config, teams, pipeline.proprietary_engine
            )
            # Unpack side-effects onto pipeline
            pipeline._prior_year_elo = stat_result.prior_year_elo
            pipeline._current_year_game_records = stat_result.current_year_game_records
            pipeline._current_year_conference_map = stat_result.current_year_conference_map
            pipeline.proprietary_metrics = stat_result.proprietary_metrics
            pipeline._torvik_name_to_id = stat_result.torvik_name_to_id
            pipeline._cbbpy_map = stat_result.cbbpy_map
            pipeline._mascot_cache = stat_result.mascot_cache
            pipeline._resolve_to_canonical = stat_result.resolve_to_canonical

            roster_result = build_rosters(pipeline.config, teams)
            pipeline.roster_rapm_quality = roster_result.roster_rapm_quality

            injury_stats, positional_impacts = apply_injury_reports(
                pipeline.config,
                roster_result.rosters,
                pipeline.positional_depth_chart,
                pipeline.injury_severity_model,
            )
            pipeline.positional_impacts = positional_impacts

            gf_result = build_or_load_game_flows(
                pipeline.config,
                teams,
                resolve_to_canonical=stat_result.resolve_to_canonical,
                mascot_cache=stat_result.mascot_cache,
            )
            pipeline.all_game_flows = gf_result.all_game_flows

            external_composites = load_external_ratings(pipeline.config, teams)
            pipeline._external_composites = external_composites

        result = LoadedData(
            teams=teams,
            torvik_map=stat_result.torvik_map,
            proprietary_map=stat_result.proprietary_map,
            rosters=roster_result.rosters,
            injury_stats=injury_stats,
            game_flows=gf_result.team_to_games,
            external_composites=external_composites,
        )

        # Validate at boundary
        try:
            from ...data.schemas import validate_loaded_data

            validation = validate_loaded_data(result)
            if not validation.passed:
                logger.error("Data loading validation failed: %s", validation.errors)
            for w in validation.warnings:
                logger.warning("Data loading: %s", w)
        except ImportError:
            pass

        logger.info(result.summary())
        return result


def _noop_ctx():
    """Fallback context manager when no timer is available."""
    from contextlib import nullcontext

    return nullcontext()
