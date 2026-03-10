"""Conference tournament prediction engine.

Reuses the SOTA pipeline's trained models and feature engineering to
predict conference tournament matchups.  This module does NOT train
new models — it leverages the existing pairwise prediction capability
(SOTAPipeline.predict_probability) to forecast conference tournament
brackets.

Usage:
    from src.conference_tournament.predictor import ConferenceTournamentPredictor

    predictor = ConferenceTournamentPredictor.from_torvik_json(
        "data/raw/torvik_2026.json"
    )

    # Predict a single conference tournament
    bracket = predictor.predict_conference("ACC")
    print(bracket.summary())

    # Predict all 31 conference tournaments
    results = predictor.predict_all()
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.conference_tournament import (
    ConferenceTournamentBracket,
    ConferenceTournamentGame,
    ConferenceTeam,
    _CONFERENCE_FULL_NAMES,
)

logger = logging.getLogger(__name__)


# Default conference tournament sizes (teams invited) for all 31 D1
# conferences.  Reflects 2025-26 formats.  Sizes may vary year-to-year
# for conferences that change their qualifying rules.
# Can be overridden via the conf_sizes parameter.
_DEFAULT_CONF_TOURNAMENT_SIZES: Dict[str, int] = {
    # Power conferences
    "ACC": 15,     # 18 members, bottom 3 don't qualify
    "B10": 18,     # all teams
    "B12": 16,     # all teams
    "SEC": 16,     # all teams
    "BE": 11,      # all teams
    # Mid-major conferences
    "Amer": 10,    # 13 members, top 10 qualify
    "MWC": 12,     # all teams
    "WCC": 12,     # all teams
    "A10": 14,     # all teams
    "MVC": 11,     # all teams
    "CAA": 13,     # all teams
    # Smaller conferences
    "AE": 8,       # 9 members, top 8 qualify
    "ASun": 12,    # all teams
    "BSky": 10,    # all teams
    "BSth": 9,     # all teams
    "BW": 8,       # 11 members, top 8 qualify
    "CUSA": 10,    # 12 members, top 10 qualify
    "Horz": 11,    # all teams
    "Ivy": 4,      # 8 members, only top 4 qualify
    "MAAC": 10,    # 11 members, top 10 qualify
    "MAC": 8,      # 13 members, top 8 qualify
    "MEAC": 8,     # all eligible members
    "NEC": 8,      # 10 members, top 8 qualify
    "OVC": 8,      # 11 members, top 8 qualify
    "Pat": 10,     # all teams
    "SB": 14,      # all teams
    "SC": 10,      # all teams
    "SWAC": 12,    # all teams
    "Slnd": 8,     # 12 members, top 8 qualify
    "Sum": 8,      # 9 members, top 8 qualify
    "WAC": 7,      # all teams
}

# -----------------------------------------------------------------------
# Standalone prediction model coefficients
# -----------------------------------------------------------------------
# Multi-feature logistic model calibrated from basketball analytics
# literature (KenPom, Oliver "Basketball on Paper" 2004, Lopez & Matthews
# JQAS 2015).  Coefficients are in logit-space per unit of each feature
# difference.
#
# Primary signal: AdjO and AdjD differences capture team strength.
# Secondary: Four Factors provide granularity on *how* teams score.
# Tempo difference captures pace-mismatch variance.

_COEFF_ADJ_O = 0.075       # Offensive efficiency diff (per point/100 poss)
_COEFF_ADJ_D = -0.075      # Defensive efficiency diff (lower is better, so negative)
_COEFF_EFG = 1.5            # eFG% diff (most important Four Factor per Oliver)
_COEFF_TO_RATE = -1.0       # Turnover rate diff (lower is better)
_COEFF_ORB_RATE = 0.8       # Offensive rebound rate diff
_COEFF_FTR = 0.5            # Free throw rate diff
_COEFF_TEMPO_VAR = 0.001    # Tempo mismatch → increased variance (slight)


class ConferenceTournamentPredictor:
    """Predicts conference tournament outcomes using existing team data.

    Operates in two modes:
    1. **Standalone mode** (no SOTA pipeline): Uses Torvik metrics with
       a multi-feature logistic model.  Fast, no ML dependencies.

    2. **Pipeline mode** (with trained SOTAPipeline): Uses the full ensemble
       (LightGBM + XGBoost + Logistic + calibration) for predictions.
       More accurate, requires a trained pipeline instance.
    """

    def __init__(
        self,
        teams_by_conference: Dict[str, List[ConferenceTeam]],
        pipeline=None,
        conf_tournament_sizes: Optional[Dict[str, int]] = None,
    ):
        """Initialize predictor.

        Args:
            teams_by_conference: Conference abbrev -> list of teams, sorted
                by conference standing (conf_seed).
            pipeline: Optional trained SOTAPipeline instance.  If None, uses
                standalone Torvik-based predictions.
            conf_tournament_sizes: Optional override for how many teams each
                conference invites to its tournament.
        """
        self.teams_by_conference = teams_by_conference
        self.pipeline = pipeline
        self.conf_sizes = conf_tournament_sizes or _DEFAULT_CONF_TOURNAMENT_SIZES

    @classmethod
    def from_torvik_json(
        cls,
        torvik_path: str,
        pipeline=None,
        conf_tournament_sizes: Optional[Dict[str, int]] = None,
        data_dir: Optional[str] = None,
        year: int = 2026,
        seed_overrides_path: Optional[str] = None,
    ) -> "ConferenceTournamentPredictor":
        """Build predictor from a Torvik JSON data file.

        Automatically enriches data with Four Factors and shooting stats
        from supplementary files if available.

        Args:
            torvik_path: Path to torvik_YYYY.json file.
            pipeline: Optional trained SOTAPipeline instance.
            conf_tournament_sizes: Optional override for tournament sizes.
            data_dir: Directory containing supplementary data files.
                Defaults to the parent directory of torvik_path.
            year: Season year for data file lookups.
            seed_overrides_path: Optional path to seed overrides JSON.

        Returns:
            Configured ConferenceTournamentPredictor.
        """
        with open(torvik_path) as f:
            data = json.load(f)

        # Determine data directory
        if data_dir is None:
            data_dir = str(Path(torvik_path).parent)

        # Enrich with Four Factors and shooting data
        from .data_enrichment import enrich_torvik_teams
        data = enrich_torvik_teams(data, data_dir=data_dir, year=year)

        # Load seed overrides if provided, or auto-detect from data directory
        if seed_overrides_path is None:
            auto_path = Path(data_dir) / f"seed_overrides_{year}.json"
            if auto_path.exists():
                seed_overrides_path = str(auto_path)
                logger.info("Auto-detected seed overrides: %s", seed_overrides_path)
            else:
                # Try to auto-scrape seeds from ESPN
                try:
                    from ..data.scrapers.conference_seeds import ConferenceSeedsScraper
                    logger.info("No seed overrides found — scraping from ESPN...")
                    scraper = ConferenceSeedsScraper(
                        cache_dir=str(Path(data_dir) / ".." / "cache")
                    )
                    scraped = scraper.scrape_seeds(year)
                    if scraped:
                        scraper.save_to_file(scraped, str(auto_path))
                        seed_overrides_path = str(auto_path)
                        logger.info(
                            "Auto-scraped seeds for %d conferences, saved to %s",
                            len(scraped), auto_path,
                        )
                except Exception as e:
                    logger.debug("Auto-scrape failed (will use AdjEM fallback): %s", e)

        seed_overrides: Dict[str, Dict[str, int]] = {}
        if seed_overrides_path:
            from .seeding import load_seed_overrides
            seed_overrides = load_seed_overrides(seed_overrides_path)

        # Normalize team IDs for pipeline compatibility
        try:
            from ..data.normalize import normalize_team_id
            _normalize = normalize_team_id
        except ImportError:
            _normalize = None

        teams_by_conf: Dict[str, List[ConferenceTeam]] = {}

        for team_data in data.get("teams", []):
            conf = team_data.get("conference", "")
            if not conf:
                continue

            raw_id = team_data.get("team_id", "")
            team_id = _normalize(raw_id) if _normalize else raw_id

            adj_o = team_data.get("adj_offensive_efficiency", 100.0)
            adj_d = team_data.get("adj_defensive_efficiency", 100.0)

            # Build stats dict from all available numeric fields
            stats = {}
            for key in (
                "effective_fg_pct", "turnover_rate", "offensive_reb_rate",
                "free_throw_rate", "opp_effective_fg_pct", "opp_turnover_rate",
                "defensive_reb_rate", "opp_free_throw_rate",
                "ft_pct", "three_pt_pct",
            ):
                val = team_data.get(key, 0.0)
                if val and val != 0.0:
                    stats[key] = val

            # Include enriched stats if present
            enriched = team_data.get("enriched_stats", {})
            for k, v in enriched.items():
                if k not in stats or stats[k] == 0.0:
                    stats[k] = v

            team = ConferenceTeam(
                team_id=team_id,
                name=team_data.get("team_name", "") or team_data.get("name", ""),
                conf_seed=0,  # Will be assigned below
                conference=conf,
                t_rank=team_data.get("t_rank", 999),
                adj_em=adj_o - adj_d,
                adj_o=adj_o,
                adj_d=adj_d,
                tempo=team_data.get("adj_tempo", 0.0),
                stats=stats,
            )
            teams_by_conf.setdefault(conf, []).append(team)

        # Apply seedings
        from .seeding import apply_seed_overrides, seed_by_adj_em
        for conf, teams in teams_by_conf.items():
            if conf in seed_overrides:
                teams_by_conf[conf] = apply_seed_overrides(teams, seed_overrides[conf])
            else:
                teams_by_conf[conf] = seed_by_adj_em(teams)

        return cls(
            teams_by_conference=teams_by_conf,
            pipeline=pipeline,
            conf_tournament_sizes=conf_tournament_sizes,
        )

    def list_conferences(self) -> List[str]:
        """Return sorted list of available conference abbreviations."""
        return sorted(self.teams_by_conference.keys())

    def get_conference_teams(self, conference: str) -> List[ConferenceTeam]:
        """Get teams for a conference, sorted by seed."""
        teams = self.teams_by_conference.get(conference, [])
        return sorted(teams, key=lambda t: t.conf_seed)

    def predict_matchup(self, team1: ConferenceTeam, team2: ConferenceTeam) -> float:
        """Predict P(team1 wins) for a single matchup.

        Uses the SOTA pipeline if available, otherwise uses a multi-feature
        logistic model calibrated from basketball analytics literature.

        Args:
            team1: First team.
            team2: Second team.

        Returns:
            Win probability for team1 (0.0 to 1.0).
        """
        if self.pipeline is not None:
            try:
                return self.pipeline.predict_probability(team1.team_id, team2.team_id)
            except Exception as e:
                logger.warning(
                    "Pipeline prediction failed for %s vs %s: %s. "
                    "Falling back to standalone model.",
                    team1.team_id, team2.team_id, e,
                )

        return self._standalone_predict(team1, team2)

    def _standalone_predict(self, team1: ConferenceTeam, team2: ConferenceTeam) -> float:
        """Multi-feature standalone prediction model.

        Uses available Torvik features in a logistic model with
        literature-derived coefficients.  Falls back gracefully when
        Four Factors are missing (uses AdjO/AdjD only).
        """
        # Core signal: offensive and defensive efficiency differences
        logit = _COEFF_ADJ_O * (team1.adj_o - team2.adj_o)
        logit += _COEFF_ADJ_D * (team1.adj_d - team2.adj_d)

        # Four Factors contributions (when available)
        s1 = team1.stats
        s2 = team2.stats

        efg1 = s1.get("effective_fg_pct", 0.0)
        efg2 = s2.get("effective_fg_pct", 0.0)
        if efg1 > 0 and efg2 > 0:
            logit += _COEFF_EFG * (efg1 - efg2)

        to1 = s1.get("turnover_rate", 0.0)
        to2 = s2.get("turnover_rate", 0.0)
        if to1 > 0 and to2 > 0:
            logit += _COEFF_TO_RATE * (to1 - to2)

        orb1 = s1.get("offensive_reb_rate", 0.0)
        orb2 = s2.get("offensive_reb_rate", 0.0)
        if orb1 > 0 and orb2 > 0:
            logit += _COEFF_ORB_RATE * (orb1 - orb2)

        ftr1 = s1.get("free_throw_rate", 0.0)
        ftr2 = s2.get("free_throw_rate", 0.0)
        if ftr1 > 0 and ftr2 > 0:
            logit += _COEFF_FTR * (ftr1 - ftr2)

        prob = 1.0 / (1.0 + math.exp(-logit))
        return float(max(0.02, min(0.98, prob)))

    def predict_conference(
        self,
        conference: str,
        num_teams: Optional[int] = None,
    ) -> ConferenceTournamentBracket:
        """Predict an entire conference tournament bracket.

        Args:
            conference: Conference abbreviation (e.g. "ACC", "B12").
            num_teams: Number of teams in the tournament.  If None, uses
                the default for that conference (or all teams).

        Returns:
            ConferenceTournamentBracket with predictions filled in.

        Raises:
            ValueError: If conference not found or invalid team count.
        """
        all_teams = self.get_conference_teams(conference)
        if not all_teams:
            raise ValueError(
                f"Conference '{conference}' not found. "
                f"Available: {', '.join(self.list_conferences())}"
            )

        # Determine tournament size
        if num_teams is None:
            num_teams = self.conf_sizes.get(conference, len(all_teams))

        num_teams = min(num_teams, len(all_teams))
        tournament_teams = all_teams[:num_teams]

        bracket = ConferenceTournamentBracket(
            conference=conference,
            teams=tournament_teams,
        )

        # Simulate the tournament round by round
        self._simulate_bracket(bracket)

        return bracket

    def predict_all(
        self,
        conferences: Optional[List[str]] = None,
    ) -> Dict[str, ConferenceTournamentBracket]:
        """Predict all (or selected) conference tournaments.

        Args:
            conferences: List of conference abbreviations to predict.
                If None, predicts all available conferences.

        Returns:
            Dict mapping conference abbreviation to predicted bracket.
        """
        if conferences is None:
            conferences = self.list_conferences()

        results = {}
        for conf in conferences:
            try:
                results[conf] = self.predict_conference(conf)
                logger.info(
                    "Predicted %s tournament: champion = %s",
                    conf,
                    results[conf].champion,
                )
            except Exception as e:
                logger.warning("Failed to predict %s tournament: %s", conf, e)

        return results

    def _simulate_bracket(self, bracket: ConferenceTournamentBracket):
        """Simulate all rounds of a bracket, advancing winners."""
        bye_teams = bracket.teams[:bracket.num_byes]

        # Round 1: predict first-round games
        for game in bracket.games[0]:
            if game.team1 and game.team2:
                prob = self.predict_matchup(game.team1, game.team2)
                winner = game.team1 if prob >= 0.5 else game.team2
                game.set_prediction(winner, prob if winner == game.team1 else 1 - prob)

        # Round 2+: fill in teams from previous round winners + byes
        for round_idx in range(1, len(bracket.games)):
            round_games = bracket.games[round_idx]
            prev_round_games = bracket.games[round_idx - 1]
            prev_winners = [g.winner for g in prev_round_games]

            if round_idx == 1 and bye_teams:
                # Merge bye teams with first-round winners
                entrants = list(bye_teams) + list(prev_winners)
                # Sort by seed then reorder into proper bracket positions
                # so that #1 and #2 are on opposite sides of the bracket
                entrants.sort(key=lambda t: t.conf_seed if t else 999)
                from ..models.conference_tournament import bracket_seed_order
                positions = bracket_seed_order(len(entrants))
                entrants = [entrants[p] for p in positions]
            else:
                entrants = list(prev_winners)

            for i, game in enumerate(round_games):
                if 2 * i < len(entrants) and 2 * i + 1 < len(entrants):
                    game.team1 = entrants[2 * i]
                    game.team2 = entrants[2 * i + 1]
                elif 2 * i < len(entrants):
                    # Odd number of teams — auto-advance
                    game.team1 = entrants[2 * i]
                    game.winner = entrants[2 * i]
                    game.win_probability = 1.0
                    game.is_bye = True
                    continue

                if game.team1 and game.team2:
                    prob = self.predict_matchup(game.team1, game.team2)
                    winner = game.team1 if prob >= 0.5 else game.team2
                    game.set_prediction(
                        winner, prob if winner == game.team1 else 1 - prob,
                    )

        # Set champion from final game
        final_round = bracket.games[-1]
        if final_round and final_round[0].winner:
            bracket.champion = final_round[0].winner

    def generate_report(
        self,
        conferences: Optional[List[str]] = None,
        simulation_results: Optional[Dict] = None,
    ) -> str:
        """Generate a human-readable prediction report.

        Args:
            conferences: Conferences to include.  None = all.
            simulation_results: Optional Monte Carlo results to include.

        Returns:
            Formatted string report.
        """
        results = self.predict_all(conferences)

        lines = [
            "=" * 70,
            "  CONFERENCE TOURNAMENT PREDICTIONS",
            "  Pre-NCAA Tournament Validation",
            "=" * 70,
        ]

        # Summary table
        lines.append(f"\n{'Conference':<25} {'Champion':<25} {'Seed':>4}  {'T-Rank':>6}")
        lines.append("-" * 70)

        for conf in sorted(results.keys()):
            bracket = results[conf]
            if bracket.champion:
                conf_name = _CONFERENCE_FULL_NAMES.get(conf, conf)
                lines.append(
                    f"{conf_name:<25} {bracket.champion.name:<25} "
                    f"{bracket.champion.conf_seed:>4}  {bracket.champion.t_rank:>6}"
                )

        # Monte Carlo championship odds (if available)
        if simulation_results:
            lines.append(f"\n{'='*70}")
            lines.append("  CHAMPIONSHIP ODDS (Monte Carlo)")
            lines.append(f"{'='*70}")
            for conf in sorted(simulation_results.keys()):
                sim = simulation_results[conf]
                conf_name = _CONFERENCE_FULL_NAMES.get(conf, conf)
                lines.append(f"\n  {conf_name}:")
                lines.append(sim.summary())

        # Upset alerts
        lines.append(f"\n{'='*70}")
        lines.append("  UPSET ALERTS (lower seed predicted to win)")
        lines.append(f"{'='*70}")

        upset_count = 0
        for conf in sorted(results.keys()):
            bracket = results[conf]
            for game in bracket.get_all_games():
                if game.is_upset and not game.is_bye:
                    conf_name = _CONFERENCE_FULL_NAMES.get(conf, conf)
                    lines.append(
                        f"  {conf_name}: {game.winner} over {game.team1 if game.winner != game.team1 else game.team2} "
                        f"({game.win_probability:.1%})"
                    )
                    upset_count += 1

        if upset_count == 0:
            lines.append("  No upsets predicted.")
        else:
            lines.append(f"\n  Total upsets predicted: {upset_count}")

        # Detailed brackets for power conferences
        power_confs = ["ACC", "B10", "B12", "SEC", "BE"]
        lines.append(f"\n{'='*70}")
        lines.append("  DETAILED BRACKETS (Power Conferences)")
        lines.append(f"{'='*70}")

        for conf in power_confs:
            if conf in results:
                lines.append(results[conf].summary())

        return "\n".join(lines)

    def to_json(
        self,
        conferences: Optional[List[str]] = None,
    ) -> str:
        """Export predictions as JSON.

        Args:
            conferences: Conferences to include.  None = all.

        Returns:
            JSON string of all predictions.
        """
        results = self.predict_all(conferences)
        return json.dumps(
            {conf: bracket.to_dict() for conf, bracket in results.items()},
            indent=2,
        )
