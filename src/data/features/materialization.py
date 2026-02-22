"""Leakage-safe historical feature materialization for NCAA modeling."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MaterializationConfig:
    """Configuration for leakage-safe historical feature generation."""

    start_season: int = 2022
    end_season: int = 2025
    historical_dir: str = "data/raw/historical"
    raw_dir: str = "data/raw"
    output_dir: str = "data/processed"
    historical_manifest_path: Optional[str] = None
    strict_validation: bool = True
    require_all_seasons: bool = True
    min_tournament_matchups: int = 1


class HistoricalFeatureMaterializer:
    """Build team-game and matchup feature tables with strict temporal leakage controls."""

    BASE_PRIOR_METRIC_COLUMNS = ["off_rtg", "def_rtg", "pace", "srs", "sos", "wins", "losses"]

    def __init__(self, config: Optional[MaterializationConfig] = None):
        self.config = config or MaterializationConfig()
        self.historical_dir = Path(self.config.historical_dir)
        self.raw_dir = Path(self.config.raw_dir)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        artifacts = self._discover_artifacts()
        team_games = self._load_team_games(artifacts)
        season_coverage = self._season_coverage_report(team_games)
        missing_seasons = season_coverage.get("missing_seasons", [])
        if self.config.strict_validation and self.config.require_all_seasons and missing_seasons:
            raise ValueError(
                "Missing requested seasons in historical games: "
                + ", ".join(str(season) for season in missing_seasons)
            )
        canonical_teams = self._canonical_team_index(team_games)
        team_metrics = self._load_team_metrics(artifacts)
        team_metrics = self._align_source_team_ids(team_metrics, canonical_teams, source_name_col="team_name")
        optional_priors = self._load_optional_prior_sources()
        optional_priors = self._align_source_team_ids(optional_priors, canonical_teams, source_name_col="team_name")
        tournament_seeds = self._load_tournament_seeds(artifacts)
        tournament_seeds = self._align_source_team_ids(tournament_seeds, canonical_teams, source_name_col="team_name")

        team_game_features = self._build_team_game_features(team_games, team_metrics, optional_priors)
        matchup_features = self._build_matchup_features(team_game_features)
        tournament_matchup_features = self._build_tournament_matchup_features(matchup_features, tournament_seeds)
        if self.config.strict_validation and len(tournament_matchup_features) < max(self.config.min_tournament_matchups, 0):
            raise ValueError(
                "Tournament matchup feature table is undersized: "
                f"{len(tournament_matchup_features)} rows < required minimum {self.config.min_tournament_matchups}. "
                "Ensure tournament seed artifacts and tournament-window games are available."
            )

        leakage = self._leakage_checks(team_game_features)
        quality = self._quality_report(team_game_features, matchup_features, tournament_matchup_features, season_coverage)
        coverage = self._variable_coverage_report(team_game_features, matchup_features, tournament_matchup_features)
        study_alignment = self._study_alignment_report(coverage)
        if self.config.strict_validation and not leakage["passed"]:
            raise ValueError(f"Leakage checks failed: {leakage['issues']}")

        team_table_path, team_table_format = self._write_table(
            team_game_features,
            f"team_game_features_{self.config.start_season}_{self.config.end_season}",
        )
        matchup_table_path, matchup_table_format = self._write_table(
            matchup_features,
            f"matchup_features_{self.config.start_season}_{self.config.end_season}",
        )
        tournament_table_path, tournament_table_format = self._write_table(
            tournament_matchup_features,
            f"tournament_matchup_features_{self.config.start_season}_{self.config.end_season}",
        )

        feature_dictionary = self._build_feature_dictionary(
            team_game_features,
            matchup_features,
            tournament_matchup_features,
        )
        feature_dict_path = self.output_dir / f"feature_dictionary_{self.config.start_season}_{self.config.end_season}.json"
        with open(feature_dict_path, "w") as f:
            json.dump(feature_dictionary, f, indent=2)

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "start_season": self.config.start_season,
            "end_season": self.config.end_season,
            "input_sources": artifacts,
            "artifacts": {
                "team_game_features_path": str(team_table_path),
                "team_game_features_format": team_table_format,
                "matchup_features_path": str(matchup_table_path),
                "matchup_features_format": matchup_table_format,
                "tournament_matchup_features_path": str(tournament_table_path),
                "tournament_matchup_features_format": tournament_table_format,
                "feature_dictionary_path": str(feature_dict_path),
            },
            "leakage_checks": leakage,
            "quality_report": quality,
            "coverage_report": coverage,
            "study_alignment_report": study_alignment,
        }

        manifest_path = self.output_dir / f"materialization_manifest_{self.config.start_season}_{self.config.end_season}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def _discover_artifacts(self) -> Dict[str, Dict[str, str]]:
        if self.config.historical_manifest_path:
            manifest_path = Path(self.config.historical_manifest_path)
            with open(manifest_path, "r") as f:
                payload = json.load(f)
            artifacts = payload.get("artifacts", {})
            out = {}
            for season in range(self.config.start_season, self.config.end_season + 1):
                season_key = str(season)
                if season_key in artifacts:
                    out[season_key] = artifacts[season_key]
            if out:
                return out

        out = {}
        for season in range(self.config.start_season, self.config.end_season + 1):
            out[str(season)] = {
                "historical_games_json": str(self.historical_dir / f"historical_games_{season}.json"),
                "team_metrics_json": str(self.historical_dir / f"team_metrics_{season}.json"),
            }
        return out

    def _load_team_games(self, artifacts: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        rows: List[Dict] = []
        for season_key, paths in artifacts.items():
            season = int(season_key)
            path = paths.get("historical_games_json")
            if not path:
                continue
            p = Path(path)
            if not p.exists():
                continue
            with open(p, "r") as f:
                payload = json.load(f)
            team_games = payload.get("team_games", [])
            if isinstance(team_games, list) and team_games:
                for row in team_games:
                    if isinstance(row, dict):
                        row["season"] = int(row.get("season", season))
                        rows.append(row)
                continue
            games = payload.get("games", [])
            for game in games:
                if not isinstance(game, dict):
                    continue
                rows.extend(self._fallback_team_game_rows(game, season))

        if not rows:
            raise ValueError("No team-game rows found. Run ingest-historical first.")

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
        for col in ("location", "site", "home_away", "is_home", "is_away", "is_neutral_site"):
            if col not in df:
                df[col] = np.nan
        for col in ("team_score", "opponent_score", "possessions", "fgm", "fga", "fg3m", "fg3a", "fta", "turnovers", "orb", "drb"):
            if col not in df:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("is_home", "is_away", "is_neutral_site"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        required = ["game_id", "team_id", "opponent_id", "team_score", "opponent_score"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Team-game dataset missing required columns: {missing}")
        return df

    def _load_team_metrics(self, artifacts: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        rows: List[Dict] = []
        for season_key, paths in artifacts.items():
            season = int(season_key)
            path = paths.get("team_metrics_json")
            if not path:
                continue
            p = Path(path)
            if not p.exists():
                continue
            with open(p, "r") as f:
                payload = json.load(f)
            teams = payload.get("teams", [])
            for row in teams:
                if not isinstance(row, dict):
                    continue
                team_id = row.get("team_id") or self._normalize_team_id(row.get("team_name") or row.get("name"))
                if not team_id:
                    continue
                rows.append(
                    {
                        "season": season,
                        "team_id": team_id,
                        "source_team_id": team_id,
                        "team_name": row.get("team_name") or row.get("name") or team_id,
                        "conference": row.get("conference") or row.get("conf") or row.get("conference_name") or "",
                        "off_rtg": self._coalesce_numeric(
                            row,
                            "off_rtg",
                            "adj_offensive_efficiency",
                            "adj_off",
                        ),
                        "def_rtg": self._coalesce_numeric(
                            row,
                            "def_rtg",
                            "adj_defensive_efficiency",
                            "adj_def",
                        ),
                        "pace": self._coalesce_numeric(row, "pace", "adj_tempo", "tempo"),
                        "srs": self._to_float(row.get("srs")),
                        "sos": self._to_float(row.get("sos")),
                        "wins": self._to_float(row.get("wins")),
                        "losses": self._to_float(row.get("losses")),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["season", "team_id", "conference"] + self.BASE_PRIOR_METRIC_COLUMNS)
        return pd.DataFrame(rows).drop_duplicates(subset=["season", "team_id"])

    def _load_optional_prior_sources(self) -> pd.DataFrame:
        season_rows: List[pd.DataFrame] = []
        for season in range(self.config.start_season, self.config.end_season + 1):
            blocks: List[pd.DataFrame] = []
            for loader in (
                self._load_advanced_metrics_season,
                self._load_torvik_season,
                self._load_roster_season,
                self._load_transfer_season,
                self._load_market_season,
                self._load_polls_season,
                self._load_torvik_splits_season,
                self._load_ncaa_team_stats_season,
                self._load_weather_context_season,
                self._load_travel_context_season,
            ):
                block = loader(season)
                if block is not None and not block.empty:
                    blocks.append(block)
            if blocks:
                merged = blocks[0]
                for block in blocks[1:]:
                    merged = merged.merge(block, on=["season", "team_id"], how="outer")
                season_rows.append(merged)
        if not season_rows:
            return pd.DataFrame(columns=["season", "team_id"])
        out = pd.concat(season_rows, ignore_index=True)
        out = out.drop_duplicates(subset=["season", "team_id"])
        return out

    def _load_tournament_seeds(self, artifacts: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        rows: List[Dict] = []
        for season_key, paths in artifacts.items():
            season = int(season_key)
            seed_path = paths.get("tournament_seeds_json")
            if not seed_path:
                default_seed_path = self.historical_dir / f"tournament_seeds_{season}.json"
                if default_seed_path.exists():
                    seed_path = str(default_seed_path)
            if not seed_path:
                continue
            p = Path(seed_path)
            if not p.exists():
                continue
            with open(p, "r") as f:
                payload = json.load(f)
            for row in payload.get("teams", []):
                if not isinstance(row, dict):
                    continue
                team_id = row.get("team_id") or self._normalize_team_id(row.get("team_name"))
                rows.append(
                    {
                        "season": season,
                        "team_id": team_id,
                        "source_team_id": team_id,
                        "team_name": row.get("team_name") or team_id,
                        "seed": self._to_float(row.get("seed")),
                        "region": row.get("region", ""),
                        "school_slug": row.get("school_slug", ""),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["season", "team_id", "team_name", "seed", "region", "school_slug"])
        return pd.DataFrame(rows).drop_duplicates(subset=["season", "team_id"])

    def _canonical_team_index(self, team_games: pd.DataFrame) -> pd.DataFrame:
        return (
            team_games[["season", "team_id", "team_name"]]
            .dropna(subset=["season", "team_id", "team_name"])
            .drop_duplicates(subset=["season", "team_id"])
            .copy()
        )

    def _align_source_team_ids(
        self,
        source_df: pd.DataFrame,
        canonical_teams: pd.DataFrame,
        source_name_col: str = "team_name",
    ) -> pd.DataFrame:
        if source_df.empty:
            return source_df

        out = source_df.copy()
        effective_name_col = source_name_col
        if effective_name_col not in out.columns:
            name_candidates = [c for c in out.columns if c.startswith(source_name_col)]
            if name_candidates:
                effective_name_col = name_candidates[0]
            else:
                out[effective_name_col] = out["team_id"].astype(str)
        if "source_team_id" not in out.columns:
            out["source_team_id"] = out["team_id"]

        canonical = canonical_teams.copy()
        canonical["name_norm"] = canonical["team_name"].map(self._normalize_name)
        canonical["name_key"] = canonical["team_name"].map(self._team_key)

        resolved_ids: List[str] = []
        match_quality: List[float] = []
        for row in out.itertuples(index=False):
            season = int(getattr(row, "season"))
            src_id = str(getattr(row, "team_id"))
            src_name = str(getattr(row, effective_name_col, src_id))
            season_base = canonical[canonical["season"] == season]
            matched_id, score = self._resolve_team_id(src_id, src_name, season_base)
            resolved_ids.append(matched_id)
            match_quality.append(score)

        out["resolved_team_id"] = resolved_ids
        out["team_match_score"] = match_quality
        out["team_id"] = out["resolved_team_id"]
        return out

    def _build_team_game_features(
        self,
        team_games: pd.DataFrame,
        team_metrics: pd.DataFrame,
        optional_priors: pd.DataFrame,
    ) -> pd.DataFrame:
        df = team_games.copy()
        df = df.sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)

        # Normalize venue context if the upstream source provides location-style fields.
        location_raw = (
            df["location"]
            .fillna(df["site"])
            .fillna(df["home_away"])
            .astype(str)
            .str.lower()
        )
        has_location = location_raw.str.strip().ne("") & location_raw.ne("nan")
        inferred_neutral = location_raw.str.contains("neutral")
        inferred_home = location_raw.str.contains("home")
        inferred_away = location_raw.str.contains("away|road|visitor")
        if {"home_team_id", "away_team_id"}.issubset(df.columns):
            inferred_home = inferred_home | (df["home_team_id"].astype(str) == df["team_id"].astype(str))
            inferred_away = inferred_away | (df["away_team_id"].astype(str) == df["team_id"].astype(str))
        df["is_neutral_site"] = df["is_neutral_site"].where(df["is_neutral_site"].notna(), inferred_neutral.astype(float))
        df["is_home"] = df["is_home"].where(df["is_home"].notna(), inferred_home.astype(float))
        df["is_away"] = df["is_away"].where(df["is_away"].notna(), inferred_away.astype(float))
        df.loc[~has_location, ["is_neutral_site", "is_home", "is_away"]] = np.nan

        df["margin_game"] = df["team_score"] - df["opponent_score"]
        df["win"] = (df["margin_game"] > 0).astype(float)
        df["pace_game"] = df["possessions"]
        df["off_eff_game"] = 100.0 * self._safe_div(df["team_score"], df["possessions"])
        df["def_eff_game"] = 100.0 * self._safe_div(df["opponent_score"], df["possessions"])
        df["net_eff_game"] = df["off_eff_game"] - df["def_eff_game"]
        df["efg_game"] = self._safe_div(df["fgm"] + 0.5 * df["fg3m"], df["fga"])
        df["to_rate_game"] = self._safe_div(df["turnovers"], df["possessions"])
        df["ft_rate_game"] = self._safe_div(df["fta"], df["fga"])
        df["three_rate_game"] = self._safe_div(df["fg3a"], df["fga"])

        # Complete Four Factors with rebound rates using opponent rebound context in the same game.
        rebound_opp = df[["game_id", "team_id", "orb", "drb"]].copy()
        rebound_opp = rebound_opp.rename(
            columns={
                "team_id": "opponent_id",
                "orb": "opp_orb_raw",
                "drb": "opp_drb_raw",
            }
        )
        df = df.merge(rebound_opp, on=["game_id", "opponent_id"], how="left")
        df["orb_rate_game"] = self._safe_div(df["orb"], df["orb"] + df["opp_drb_raw"])
        df["drb_rate_game"] = self._safe_div(df["drb"], df["drb"] + df["opp_orb_raw"])
        df = df.drop(columns=["opp_orb_raw", "opp_drb_raw"])

        df["games_played_prior"] = df.groupby("team_id").cumcount().astype(float)

        prev_date = df.groupby("team_id")["date"].shift(1)
        df["rest_days"] = (df["date"] - prev_date).dt.days.astype(float)
        df["back_to_back"] = (df["rest_days"] <= 1).astype(float).where(df["rest_days"].notna(), np.nan)
        df["games_in_last_7_days"] = (
            df.groupby("team_id")["date"].transform(lambda s: pd.Series(self._rolling_day_counts(s, 7), index=s.index))
        )

        season_start = pd.to_datetime((df["season"] - 1).astype(str) + "-11-01", errors="coerce")
        df["season_day"] = (df["date"] - season_start).dt.days.clip(lower=0)
        df["season_progress"] = self._safe_div(df["season_day"], 180.0)

        base_metrics = {
            "win": "win_pct_prior",
            "off_eff_game": "off_eff_prior",
            "def_eff_game": "def_eff_prior",
            "net_eff_game": "net_eff_prior",
            "pace_game": "pace_prior",
            "efg_game": "efg_prior",
            "to_rate_game": "to_rate_prior",
            "ft_rate_game": "ft_rate_prior",
            "three_rate_game": "three_rate_prior",
            "orb_rate_game": "orb_rate_prior",
            "drb_rate_game": "drb_rate_prior",
            "margin_game": "margin_prior",
        }
        for src_col, out_col in base_metrics.items():
            df[out_col] = df.groupby("team_id")[src_col].transform(self._shifted_expanding_mean)

        for window in (3, 5, 10):
            df[f"win_pct_l{window}"] = df.groupby("team_id")["win"].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f"net_eff_l{window}"] = df.groupby("team_id")["net_eff_game"].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f"pace_l{window}"] = df.groupby("team_id")["pace_game"].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f"to_rate_l{window}"] = df.groupby("team_id")["to_rate_game"].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f"ft_rate_l{window}"] = df.groupby("team_id")["ft_rate_game"].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f"orb_rate_l{window}"] = df.groupby("team_id")["orb_rate_game"].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f"drb_rate_l{window}"] = df.groupby("team_id")["drb_rate_game"].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
            )

        df["margin_std_l10"] = df.groupby("team_id")["margin_game"].transform(
            lambda s: s.shift(1).rolling(window=10, min_periods=2).std()
        )
        df["net_eff_ewm5"] = df.groupby("team_id")["net_eff_game"].transform(
            lambda s: s.shift(1).ewm(span=5, adjust=False, min_periods=1).mean()
        )

        close_game_wins = df["win"].where(df["margin_game"].abs() <= 5)
        df["close_game_win_pct_prior"] = close_game_wins.groupby(df["team_id"]).transform(
            self._shifted_expanding_mean
        )
        df["lead_entropy_proxy"] = df["to_rate_l5"] * (1.0 - df["ft_rate_l5"])

        prior_metrics = self._shift_prior_season_table(team_metrics, self.BASE_PRIOR_METRIC_COLUMNS, prefix="prior_season_")
        df = df.merge(prior_metrics, on=["season", "team_id"], how="left")
        if "conference" in team_metrics.columns:
            conference_lookup = team_metrics[["season", "team_id", "conference"]].copy()
            conference_lookup["conference"] = conference_lookup["conference"].astype(str)
            conference_lookup = conference_lookup.drop_duplicates(subset=["season", "team_id"])
            df = df.merge(conference_lookup, on=["season", "team_id"], how="left")

            conf_prior = team_metrics[["season", "conference", "srs", "sos", "off_rtg", "def_rtg"]].copy()
            conf_prior["conference"] = conf_prior["conference"].astype(str)
            conf_prior = (
                conf_prior.groupby(["season", "conference"], as_index=False)
                .agg(
                    conference_srs_mean=("srs", "mean"),
                    conference_sos_mean=("sos", "mean"),
                    conference_off_rtg_mean=("off_rtg", "mean"),
                    conference_def_rtg_mean=("def_rtg", "mean"),
                )
            )
            conf_prior["season"] = pd.to_numeric(conf_prior["season"], errors="coerce") + 1
            conf_prior = conf_prior.rename(
                columns={
                    "conference_srs_mean": "prior_conference_srs_mean",
                    "conference_sos_mean": "prior_conference_sos_mean",
                    "conference_off_rtg_mean": "prior_conference_off_rtg_mean",
                    "conference_def_rtg_mean": "prior_conference_def_rtg_mean",
                }
            )
            df = df.merge(conf_prior, on=["season", "conference"], how="left")
        if not optional_priors.empty:
            optional_columns = [
                c
                for c in optional_priors.columns
                if c not in {"season", "team_id", "team_name", "source_team_id", "resolved_team_id"}
                and pd.api.types.is_numeric_dtype(optional_priors[c])
            ]
            shifted_optional = self._shift_prior_season_table(optional_priors, optional_columns, prefix="prior_")
            df = df.merge(shifted_optional, on=["season", "team_id"], how="left")

        if "prior_season_wins" in df and "prior_season_losses" in df:
            df["prior_season_win_pct"] = self._safe_div(
                df["prior_season_wins"],
                df["prior_season_wins"] + df["prior_season_losses"],
            )

        # opponent pregame features for style-matchup variables
        opp_cols = [
            "off_eff_prior",
            "def_eff_prior",
            "net_eff_prior",
            "win_pct_prior",
            "pace_prior",
            "efg_prior",
            "to_rate_prior",
            "ft_rate_prior",
            "three_rate_prior",
            "orb_rate_prior",
            "drb_rate_prior",
            "prior_season_off_rtg",
            "prior_season_def_rtg",
            "prior_season_pace",
            "prior_season_sos",
            "prior_season_srs",
            "prior_conference_srs_mean",
            "prior_conference_sos_mean",
            "prior_conference_off_rtg_mean",
            "prior_conference_def_rtg_mean",
        ]
        opp_cols = [c for c in opp_cols if c in df.columns]
        opp_frame = df[["game_id", "team_id"] + opp_cols].copy()
        rename_map = {"team_id": "opponent_id"}
        rename_map.update({col: f"opp_{col}" for col in opp_cols})
        opp_frame = opp_frame.rename(columns=rename_map)
        df = df.merge(opp_frame, on=["game_id", "opponent_id"], how="left")

        if "opp_def_eff_prior" in df and "off_eff_prior" in df:
            df["offense_vs_opp_def"] = df["off_eff_prior"] - df["opp_def_eff_prior"]
        if "opp_off_eff_prior" in df and "def_eff_prior" in df:
            df["defense_vs_opp_off"] = df["opp_off_eff_prior"] - df["def_eff_prior"]
        if "opp_pace_prior" in df and "pace_prior" in df:
            df["tempo_mismatch"] = (df["pace_prior"] - df["opp_pace_prior"]).abs()

        # strict ordering for reproducibility
        df = df.sort_values(["season", "date", "game_id", "team_id"]).reset_index(drop=True)
        return df

    def _build_matchup_features(self, team_features: pd.DataFrame) -> pd.DataFrame:
        groups = team_features.groupby("game_id", sort=False)
        rows: List[Dict] = []

        candidate_cols = [
            "win_pct_prior",
            "off_eff_prior",
            "def_eff_prior",
            "net_eff_prior",
            "pace_prior",
            "efg_prior",
            "to_rate_prior",
            "ft_rate_prior",
            "three_rate_prior",
            "orb_rate_prior",
            "drb_rate_prior",
            "win_pct_l3",
            "win_pct_l5",
            "win_pct_l10",
            "net_eff_l3",
            "net_eff_l5",
            "net_eff_l10",
            "orb_rate_l5",
            "drb_rate_l5",
            "margin_std_l10",
            "close_game_win_pct_prior",
            "lead_entropy_proxy",
            "rest_days",
            "back_to_back",
            "games_in_last_7_days",
            "is_home",
            "is_away",
            "is_neutral_site",
            "prior_season_off_rtg",
            "prior_season_def_rtg",
            "prior_season_pace",
            "prior_season_sos",
            "prior_season_srs",
            "prior_season_win_pct",
            "prior_conference_srs_mean",
            "prior_conference_sos_mean",
            "prior_conference_off_rtg_mean",
            "prior_conference_def_rtg_mean",
            "tempo_mismatch",
            "offense_vs_opp_def",
            "defense_vs_opp_off",
        ]
        candidate_cols.extend([c for c in team_features.columns if c.startswith("prior_prop_")])
        candidate_cols.extend([c for c in team_features.columns if c.startswith("prior_torvik_")])
        candidate_cols.extend([c for c in team_features.columns if c.startswith("prior_roster_")])
        candidate_cols.extend([c for c in team_features.columns if c.startswith("prior_transfer_")])
        feature_cols = [c for c in dict.fromkeys(candidate_cols) if c in team_features.columns]

        for game_id, group in groups:
            if len(group) != 2:
                continue
            g = group.sort_values("team_id")
            t1 = g.iloc[0]
            t2 = g.iloc[1]

            row = {
                "game_id": game_id,
                "season": int(t1["season"]),
                "date": t1["date"].isoformat() if pd.notna(t1["date"]) else "",
                "team1_id": t1["team_id"],
                "team2_id": t2["team_id"],
                "team1_score": self._to_float(t1["team_score"]),
                "team2_score": self._to_float(t2["team_score"]),
                "team1_win": int(self._to_float(t1["team_score"]) > self._to_float(t2["team_score"])),
                "margin": self._to_float(t1["team_score"]) - self._to_float(t2["team_score"]),
            }
            for col in feature_cols:
                v1 = self._to_float(t1.get(col))
                v2 = self._to_float(t2.get(col))
                row[f"diff_{col}"] = v1 - v2
                row[f"avg_{col}"] = (v1 + v2) / 2.0
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_tournament_matchup_features(
        self,
        matchup_features: pd.DataFrame,
        tournament_seeds: pd.DataFrame,
    ) -> pd.DataFrame:
        if matchup_features.empty or tournament_seeds.empty:
            return pd.DataFrame(
                columns=[
                    "game_id",
                    "season",
                    "date",
                    "team1_id",
                    "team2_id",
                    "team1_seed",
                    "team2_seed",
                    "seed_diff",
                    "is_seed_upset",
                    "team1_win",
                    "margin",
                ]
            )

        seeds = tournament_seeds[["season", "team_id", "seed", "region"]].drop_duplicates(
            subset=["season", "team_id"]
        )
        t1 = seeds.rename(
            columns={
                "team_id": "team1_id",
                "seed": "team1_seed",
                "region": "team1_region",
            }
        )
        t2 = seeds.rename(
            columns={
                "team_id": "team2_id",
                "seed": "team2_seed",
                "region": "team2_region",
            }
        )
        df = matchup_features.merge(t1, on=["season", "team1_id"], how="inner")
        df = df.merge(t2, on=["season", "team2_id"], how="inner")
        if df.empty:
            return df

        df["seed_diff"] = df["team1_seed"] - df["team2_seed"]
        df["is_seed_upset"] = (
            ((df["team1_seed"] > df["team2_seed"]) & (df["team1_win"] == 1))
            | ((df["team2_seed"] > df["team1_seed"]) & (df["team1_win"] == 0))
        ).astype(int)
        df["same_region"] = (df["team1_region"] == df["team2_region"]).astype(int)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # NCAA tournament window (Selection Sunday onward)
        season_start = pd.to_datetime(df["season"].astype(str) + "-03-13", errors="coerce")
        season_end = pd.to_datetime(df["season"].astype(str) + "-04-15", errors="coerce")
        df = df[(df["date"] >= season_start) & (df["date"] <= season_end)]
        if df.empty:
            return df
        return df.sort_values(["season", "date", "game_id"]).reset_index(drop=True)

    def _leakage_checks(self, team_features: pd.DataFrame) -> Dict:
        issues: List[str] = []
        checked_rows = 0

        df = team_features.sort_values(["team_id", "date", "game_id"]).copy()
        expected = df.groupby("team_id")["off_eff_game"].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        mask = df["games_played_prior"] > 0
        checked_rows = int(mask.sum())
        diff = (df.loc[mask, "off_eff_prior"] - expected.loc[mask]).abs()
        max_diff = float(diff.max()) if len(diff) else 0.0
        if len(diff) and max_diff > 1e-9:
            issues.append(f"off_eff_prior leakage mismatch max diff={max_diff:.6f}")

        first_game_mask = df["games_played_prior"] == 0
        if first_game_mask.any():
            leaked_first = df.loc[first_game_mask, "off_eff_prior"].notna().sum()
            if int(leaked_first) > 0:
                issues.append(f"{int(leaked_first)} first-game rows have non-null off_eff_prior")

        per_game_counts = df.groupby("game_id")["team_id"].nunique()
        invalid_games = int((per_game_counts != 2).sum())
        if invalid_games > 0:
            issues.append(f"{invalid_games} games do not have exactly 2 team rows")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "rows_checked": checked_rows,
        }

    def _quality_report(
        self,
        team_features: pd.DataFrame,
        matchup_features: pd.DataFrame,
        tournament_matchup_features: pd.DataFrame,
        season_coverage: Dict,
    ) -> Dict:
        numeric = team_features.select_dtypes(include=[np.number])
        missing = numeric.isna().mean().sort_values(ascending=False).head(20)
        season_counts = (
            team_features.groupby("season")
            .agg(
                team_games=("game_id", "count"),
                unique_games=("game_id", "nunique"),
                unique_teams=("team_id", "nunique"),
            )
            .reset_index()
            .to_dict("records")
        )
        return {
            "team_game_rows": int(len(team_features)),
            "matchup_rows": int(len(matchup_features)),
            "tournament_matchup_rows": int(len(tournament_matchup_features)),
            "season_counts": season_counts,
            "season_coverage": season_coverage,
            "top_missing_numeric_columns": {k: float(v) for k, v in missing.items()},
        }

    def _season_coverage_report(self, team_games: pd.DataFrame) -> Dict:
        expected = list(range(self.config.start_season, self.config.end_season + 1))
        present = sorted({int(s) for s in team_games["season"].dropna().astype(int).tolist()})
        missing = [s for s in expected if s not in set(present)]
        return {
            "expected_seasons": expected,
            "present_seasons": present,
            "missing_seasons": missing,
        }

    def _variable_coverage_report(
        self,
        team_features: pd.DataFrame,
        matchup_features: pd.DataFrame,
        tournament_matchup_features: pd.DataFrame,
    ) -> Dict:
        categories = [
            {
                "name": "efficiency_four_factors",
                "critical": True,
                "columns": [
                    "off_eff_prior",
                    "def_eff_prior",
                    "net_eff_prior",
                    "efg_prior",
                    "to_rate_prior",
                    "ft_rate_prior",
                    "orb_rate_prior",
                    "drb_rate_prior",
                ],
                "table": "team",
            },
            {
                "name": "schedule_strength",
                "critical": True,
                "columns": ["prior_season_sos", "prior_season_srs"],
                "table": "team",
            },
            {
                "name": "conference_strength",
                "critical": False,
                "columns": ["prior_conference_srs_mean", "prior_conference_sos_mean"],
                "table": "team",
            },
            {
                "name": "recent_form_and_variance",
                "critical": True,
                "columns": ["win_pct_l5", "net_eff_l5", "margin_std_l10", "close_game_win_pct_prior"],
                "table": "team",
            },
            {
                "name": "rest_fatigue",
                "critical": True,
                "columns": ["rest_days", "back_to_back", "games_in_last_7_days"],
                "table": "team",
            },
            {
                "name": "tournament_seed_context",
                "critical": True,
                "columns": ["team1_seed", "team2_seed", "seed_diff"],
                "table": "tournament",
            },
            {
                "name": "talent_roster_transfer",
                "critical": False,
                "columns": [
                    "prior_roster_total_rapm",
                    "prior_roster_transfer_share",
                    "prior_roster_minutes_returning_share",
                    "prior_roster_continuity_learning_rate",
                    "prior_transfer_net",
                ],
                "table": "team",
            },
            {
                "name": "proprietary_xp",
                "critical": False,
                "columns": ["prior_prop_off_xp", "prior_prop_def_xp", "prior_prop_shot_dist"],
                "table": "team",
            },
            {
                "name": "market_priors",
                "critical": False,
                "columns": ["prior_market_implied_win", "prior_market_title_odds"],
                "table": "team",
            },
        ]

        by_category: List[Dict] = []
        critical_gaps: List[str] = []
        for cat in categories:
            table = team_features if cat["table"] == "team" else tournament_matchup_features
            available = [col for col in cat["columns"] if col in table.columns]
            missing = [col for col in cat["columns"] if col not in table.columns]
            coverage_ratio = float(len(available) / max(len(cat["columns"]), 1))
            status = "covered" if coverage_ratio >= 0.75 else "partial" if coverage_ratio > 0 else "missing"
            if cat["critical"] and status != "covered":
                critical_gaps.append(cat["name"])
            by_category.append(
                {
                    "name": cat["name"],
                    "critical": cat["critical"],
                    "status": status,
                    "coverage_ratio": coverage_ratio,
                    "available_columns": available,
                    "missing_columns": missing,
                }
            )

        return {
            "categories": by_category,
            "critical_gaps": critical_gaps,
            "summary": {
                "team_columns": len(team_features.columns),
                "matchup_columns": len(matchup_features.columns),
                "tournament_matchup_columns": len(tournament_matchup_features.columns),
            },
        }

    def _study_alignment_report(self, coverage_report: Dict) -> Dict:
        category_status = {c["name"]: c["status"] for c in coverage_report.get("categories", [])}
        studies = {
            "proprietary_efficiency_model": {
                "reference": "proprietary: iterative SOS-adjusted efficiency (KenPom method)",
                "expects": ["efficiency_four_factors", "schedule_strength", "recent_form_and_variance"],
            },
            "fivethirtyeight_march_methodology": {
                "reference": "https://fivethirtyeight.com/methodology/how-our-march-madness-predictions-work-2/",
                "expects": ["efficiency_four_factors", "schedule_strength", "tournament_seed_context", "market_priors", "proprietary_xp"],
            },
            "arxiv_1701_07316": {
                "reference": "https://arxiv.org/abs/1701.07316",
                "expects": ["efficiency_four_factors", "schedule_strength", "tournament_seed_context"],
            },
            "arxiv_2503_21790": {
                "reference": "https://arxiv.org/abs/2503.21790",
                "expects": ["efficiency_four_factors", "recent_form_and_variance", "tournament_seed_context"],
            },
        }

        out = {}
        for key, cfg in studies.items():
            statuses = {name: category_status.get(name, "missing") for name in cfg["expects"]}
            covered = sum(1 for s in statuses.values() if s == "covered")
            partial = sum(1 for s in statuses.values() if s == "partial")
            completeness = (covered + 0.5 * partial) / max(len(statuses), 1)
            out[key] = {
                "reference": cfg["reference"],
                "expected_categories": cfg["expects"],
                "category_statuses": statuses,
                "completeness_score": round(float(completeness), 3),
            }
        return out

    def _write_table(self, df: pd.DataFrame, stem: str) -> Tuple[Path, str]:
        parquet_path = self.output_dir / f"{stem}.parquet"
        try:
            df.to_parquet(parquet_path, index=False)
            return parquet_path, "parquet"
        except Exception:
            csv_path = self.output_dir / f"{stem}.csv"
            df.to_csv(csv_path, index=False)
            return csv_path, "csv"

    def _build_feature_dictionary(
        self,
        team_features: pd.DataFrame,
        matchup_features: pd.DataFrame,
        tournament_matchup_features: pd.DataFrame,
    ) -> Dict:
        return {
            "team_game_columns": team_features.columns.tolist(),
            "matchup_columns": matchup_features.columns.tolist(),
            "tournament_matchup_columns": tournament_matchup_features.columns.tolist(),
            "key_columns": {
                "team_game": ["season", "date", "game_id", "team_id"],
                "matchup": ["season", "date", "game_id", "team1_id", "team2_id"],
                "tournament_matchup": ["season", "date", "game_id", "team1_id", "team2_id"],
            },
            "targets": {
                "team_game": ["win", "margin_game"],
                "matchup": ["team1_win", "margin"],
                "tournament_matchup": ["team1_win", "margin", "is_seed_upset"],
            },
        }

    def _load_advanced_metrics_season(self, season: int) -> Optional[pd.DataFrame]:
        # Try proprietary metrics first, then fall back to legacy advanced_metrics artifact
        for filename in (f"advanced_metrics_{season}.json", f"kenpom_{season}.json"):
            path = self.raw_dir / filename
            if not path.exists():
                continue
            with open(path, "r") as f:
                payload = json.load(f)
            rows = []
            for row in payload.get("teams", []):
                if not isinstance(row, dict):
                    continue
                team_id = row.get("team_id") or self._normalize_team_id(row.get("name"))
                if not team_id:
                    continue
                rows.append(
                    {
                        "season": season,
                        "team_id": team_id,
                        "source_team_id": team_id,
                        "team_name": row.get("name") or row.get("team_name") or team_id,
                        "prop_adj_off": self._to_float(row.get("adj_offensive_efficiency")),
                        "prop_adj_def": self._to_float(row.get("adj_defensive_efficiency")),
                        "prop_adj_em": self._to_float(row.get("adj_efficiency_margin")),
                        "prop_adj_tempo": self._to_float(row.get("adj_tempo")),
                        "prop_luck": self._to_float(row.get("luck")),
                        "prop_sos_adj_em": self._to_float(row.get("sos_adj_em")),
                        "prop_off_xp": self._to_float(row.get("offensive_xp_per_possession")),
                        "prop_def_xp": self._to_float(row.get("defensive_xp_per_possession")),
                        "prop_shot_dist": self._to_float(row.get("shot_distribution_score")),
                    }
                )
            if rows:
                return pd.DataFrame(rows)
        return None

    def _load_torvik_season(self, season: int) -> Optional[pd.DataFrame]:
        path = self.raw_dir / f"torvik_{season}.json"
        if not path.exists():
            alt = self.historical_dir / f"torvik_{season}.json"
            if alt.exists():
                path = alt
            else:
                return None
        with open(path, "r") as f:
            payload = json.load(f)
        rows = []
        for row in payload.get("teams", []):
            if not isinstance(row, dict):
                continue
            team_id = row.get("team_id") or self._normalize_team_id(row.get("name"))
            if not team_id:
                continue
            rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "source_team_id": team_id,
                    "team_name": row.get("name") or row.get("team_name") or team_id,
                    "torvik_barthag": self._to_float(row.get("barthag")),
                    "torvik_efg": self._to_float(row.get("effective_fg_pct")),
                    "torvik_to_rate": self._to_float(row.get("turnover_rate")),
                    "torvik_orb_rate": self._to_float(row.get("offensive_reb_rate")),
                    "torvik_ft_rate": self._to_float(row.get("free_throw_rate")),
                }
            )
        return pd.DataFrame(rows) if rows else None

    def _load_roster_season(self, season: int) -> Optional[pd.DataFrame]:
        path = self.raw_dir / f"rosters_{season}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        rows = []
        for team_row in payload.get("teams", []):
            if not isinstance(team_row, dict):
                continue
            team_id = team_row.get("team_id") or self._normalize_team_id(team_row.get("team_name") or team_row.get("name"))
            players = [p for p in team_row.get("players", []) if isinstance(p, dict)]
            if not team_id or not players:
                continue
            total_rapm = 0.0
            total_warp = 0.0
            transfer_count = 0
            healthy_count = 0
            for player in players:
                rapm_total = self._to_float(player.get("rapm_total"))
                if np.isnan(rapm_total):
                    rapm_total = self._to_float(player.get("rapm_offensive")) + self._to_float(player.get("rapm_defensive"))
                total_rapm += 0.0 if np.isnan(rapm_total) else rapm_total
                warp_val = self._to_float(player.get("warp"))
                total_warp += 0.0 if np.isnan(warp_val) else warp_val
                transfer_count += int(bool(player.get("is_transfer")))
                healthy_count += int(str(player.get("injury_status", "healthy")).lower() == "healthy")
            rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "source_team_id": team_id,
                    "team_name": team_row.get("team_name") or team_row.get("name") or team_id,
                    "roster_total_rapm": total_rapm,
                    "roster_total_warp": total_warp,
                    "roster_transfer_share": transfer_count / max(len(players), 1),
                    "roster_health_share": healthy_count / max(len(players), 1),
                    "roster_player_count": float(len(players)),
                    "roster_minutes_returning_share": np.nan,
                    "roster_continuity_learning_rate": np.nan,
                    "roster_upperclass_share": np.nan,
                    "roster_injury_rapm_share": np.nan,
                }
            )
            minutes_total = 0.0
            minutes_returning = 0.0
            upperclass_count = 0
            injured_rapm = 0.0
            for player in players:
                minutes = self._to_float(
                    player.get("minutes_per_game")
                    or player.get("minutes")
                    or player.get("mpg")
                    or player.get("minutes_share")
                )
                if np.isnan(minutes):
                    minutes = 0.0
                is_transfer = bool(player.get("is_transfer", False))
                is_returning = bool(player.get("is_returning", not is_transfer))
                class_year = str(
                    player.get("class_year")
                    or player.get("year")
                    or player.get("academic_year")
                    or ""
                ).strip().lower()
                injury_status = str(player.get("injury_status", "healthy")).strip().lower()
                rapm_val = self._to_float(player.get("rapm_total"))
                if player.get("rapm_total") is None:
                    rapm_val = self._to_float(player.get("rapm_offensive")) + self._to_float(player.get("rapm_defensive"))
                if np.isnan(rapm_val):
                    rapm_val = 0.0

                minutes_total += max(minutes, 0.0)
                if is_returning:
                    minutes_returning += max(minutes, 0.0)
                if class_year in {"jr", "junior", "sr", "senior", "gr", "grad", "graduate"}:
                    upperclass_count += 1
                if injury_status not in {"healthy", "available", ""}:
                    injured_rapm += rapm_val

            continuity = minutes_returning / max(minutes_total, 1.0) if minutes_total > 0 else 1.0 - (transfer_count / max(len(players), 1))
            rows[-1]["roster_minutes_returning_share"] = continuity
            rows[-1]["roster_continuity_learning_rate"] = 1.0 + 0.15 * (1.0 - continuity)
            rows[-1]["roster_upperclass_share"] = upperclass_count / max(len(players), 1)
            rows[-1]["roster_injury_rapm_share"] = injured_rapm / max(abs(total_rapm), 1.0)
        return pd.DataFrame(rows) if rows else None

    def _load_transfer_season(self, season: int) -> Optional[pd.DataFrame]:
        path = self.raw_dir / f"transfer_portal_{season}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        in_counts: Dict[str, int] = {}
        out_counts: Dict[str, int] = {}
        for row in payload.get("entries", []):
            if not isinstance(row, dict):
                continue
            source_id = self._normalize_team_id(row.get("source_team_name") or row.get("from_team"))
            dest_id = self._normalize_team_id(row.get("destination_team_name") or row.get("to_team"))
            if source_id:
                out_counts[source_id] = out_counts.get(source_id, 0) + 1
            if dest_id:
                in_counts[dest_id] = in_counts.get(dest_id, 0) + 1
        team_ids = sorted(set(in_counts) | set(out_counts))
        if not team_ids:
            return None
        rows = []
        for team_id in team_ids:
            incoming = float(in_counts.get(team_id, 0))
            outgoing = float(out_counts.get(team_id, 0))
            rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "source_team_id": team_id,
                    "team_name": team_id,
                    "transfer_in": incoming,
                    "transfer_out": outgoing,
                    "transfer_net": incoming - outgoing,
                }
            )
        return pd.DataFrame(rows)

    def _load_market_season(self, season: int) -> Optional[pd.DataFrame]:
        path = self.raw_dir / f"odds_{season}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        teams = payload.get("teams", payload if isinstance(payload, list) else [])
        if not isinstance(teams, list):
            return None
        rows = []
        for row in teams:
            if not isinstance(row, dict):
                continue
            team_name = row.get("team_name") or row.get("name") or ""
            team_id = row.get("team_id") or self._normalize_team_id(team_name)
            if not team_id:
                continue
            rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "source_team_id": team_id,
                    "team_name": team_name or team_id,
                    "market_implied_win": self._to_float(row.get("implied_win_probability")),
                    "market_title_odds": self._to_float(row.get("title_odds")),
                }
            )
        return pd.DataFrame(rows) if rows else None

    def _load_polls_season(self, season: int) -> Optional[pd.DataFrame]:
        path = self.raw_dir / f"polls_{season}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        rows = payload.get("records", payload.get("polls", payload if isinstance(payload, list) else []))
        if not isinstance(rows, list):
            return None
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            team_name = row.get("team_name") or row.get("name") or ""
            team_id = row.get("team_id") or self._normalize_team_id(team_name)
            if not team_id:
                continue
            out.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "source_team_id": team_id,
                    "team_name": team_name or team_id,
                    "polls_ap_preseason": self._to_float(row.get("ap_preseason_rank")),
                    "polls_coaches_preseason": self._to_float(row.get("coaches_preseason_rank")),
                    "polls_ap_weekly_mean": self._to_float(row.get("ap_weekly_mean")),
                    "polls_coaches_weekly_mean": self._to_float(row.get("coaches_weekly_mean")),
                    "polls_ap_weekly_std": self._to_float(row.get("ap_weekly_std")),
                    "polls_coaches_weekly_std": self._to_float(row.get("coaches_weekly_std")),
                }
            )
        return pd.DataFrame(out) if out else None

    def _load_torvik_splits_season(self, season: int) -> Optional[pd.DataFrame]:
        path = self.raw_dir / f"torvik_splits_{season}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        teams = payload.get("records", payload.get("teams", payload if isinstance(payload, list) else []))
        if not isinstance(teams, list):
            return None
        rows = []
        for row in teams:
            if not isinstance(row, dict):
                continue
            team_name = row.get("team_name") or row.get("name") or ""
            team_id = row.get("team_id") or self._normalize_team_id(team_name)
            if not team_id:
                continue
            rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "source_team_id": team_id,
                    "team_name": team_name or team_id,
                    "torvik_split_wab": self._to_float(row.get("wab")),
                    "torvik_split_sos": self._to_float(row.get("sos")),
                    "torvik_split_ncsos": self._to_float(row.get("ncsos")),
                    "torvik_split_vs_top50": self._to_float(row.get("vs_top50_win_pct")),
                }
            )
        return pd.DataFrame(rows) if rows else None

    def _load_ncaa_team_stats_season(self, season: int) -> Optional[pd.DataFrame]:
        path = self.raw_dir / f"ncaa_team_stats_{season}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        teams = payload.get("records", payload.get("teams", payload if isinstance(payload, list) else []))
        if not isinstance(teams, list):
            return None
        rows = []
        for row in teams:
            if not isinstance(row, dict):
                continue
            team_name = row.get("team_name") or row.get("name") or ""
            team_id = row.get("team_id") or self._normalize_team_id(team_name)
            if not team_id:
                continue
            rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "source_team_id": team_id,
                    "team_name": team_name or team_id,
                    "ncaa_ast_to_ratio": self._to_float(row.get("assist_turnover_ratio")),
                    "ncaa_rebound_margin": self._to_float(row.get("rebound_margin")),
                    "ncaa_foul_rate": self._to_float(row.get("foul_rate")),
                }
            )
        return pd.DataFrame(rows) if rows else None

    def _load_weather_context_season(self, season: int) -> Optional[pd.DataFrame]:
        path = self.raw_dir / f"weather_context_{season}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        records = payload.get("records", payload if isinstance(payload, list) else [])
        if not isinstance(records, list):
            return None
        by_team: Dict[str, Dict[str, float]] = {}
        for row in records:
            if not isinstance(row, dict):
                continue
            team_name = row.get("team_name") or row.get("name") or ""
            team_id = row.get("team_id") or self._normalize_team_id(team_name)
            if not team_id:
                continue
            bucket = by_team.setdefault(team_id, {"count": 0.0, "temp": 0.0, "wind": 0.0, "alerts": 0.0})
            bucket["count"] += 1.0
            bucket["temp"] += max(self._to_float(row.get("temperature_f")) or 0.0, 0.0)
            bucket["wind"] += max(self._to_float(row.get("wind_mph")) or 0.0, 0.0)
            bucket["alerts"] += max(self._to_float(row.get("severe_alert")) or 0.0, 0.0)
        rows = []
        for team_id, agg in by_team.items():
            count = max(agg["count"], 1.0)
            rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "source_team_id": team_id,
                    "team_name": team_id,
                    "weather_avg_temp_f": agg["temp"] / count,
                    "weather_avg_wind_mph": agg["wind"] / count,
                    "weather_severe_alert_rate": agg["alerts"] / count,
                }
            )
        return pd.DataFrame(rows) if rows else None

    def _load_travel_context_season(self, season: int) -> Optional[pd.DataFrame]:
        path = self.raw_dir / f"travel_context_{season}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        records = payload.get("records", payload if isinstance(payload, list) else [])
        if not isinstance(records, list):
            return None
        by_team: Dict[str, Dict[str, float]] = {}
        for row in records:
            if not isinstance(row, dict):
                continue
            team_name = row.get("team_name") or row.get("name") or ""
            team_id = row.get("team_id") or self._normalize_team_id(team_name)
            if not team_id:
                continue
            bucket = by_team.setdefault(team_id, {"count": 0.0, "miles": 0.0, "tz": 0.0, "days": 0.0})
            bucket["count"] += 1.0
            bucket["miles"] += max(self._to_float(row.get("travel_miles")) or 0.0, 0.0)
            bucket["tz"] += max(self._to_float(row.get("timezone_change_hours")) or 0.0, 0.0)
            bucket["days"] += max(self._to_float(row.get("trip_days")) or 0.0, 0.0)
        rows = []
        for team_id, agg in by_team.items():
            count = max(agg["count"], 1.0)
            rows.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "source_team_id": team_id,
                    "team_name": team_id,
                    "travel_avg_miles": agg["miles"] / count,
                    "travel_avg_timezone_change": agg["tz"] / count,
                    "travel_avg_trip_days": agg["days"] / count,
                }
            )
        return pd.DataFrame(rows) if rows else None

    def _shift_prior_season_table(self, df: pd.DataFrame, value_columns: List[str], prefix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["season", "team_id"])
        cols = [c for c in value_columns if c in df.columns]
        if not cols:
            return pd.DataFrame(columns=["season", "team_id"])
        ranked = df.copy()
        if "team_match_score" in ranked.columns:
            ranked = ranked.sort_values(["season", "team_id", "team_match_score"], ascending=[True, True, False])
            ranked = ranked.drop_duplicates(subset=["season", "team_id"])
        out = ranked[["season", "team_id"] + cols].copy()
        out["season"] = pd.to_numeric(out["season"], errors="coerce") + 1
        rename = {col: f"{prefix}{col}" for col in cols}
        out = out.rename(columns=rename)
        return out.dropna(subset=["season", "team_id"]).drop_duplicates(subset=["season", "team_id"])

    def _resolve_team_id(
        self,
        source_team_id: str,
        source_name: str,
        season_canonical: pd.DataFrame,
    ) -> Tuple[str, float]:
        if season_canonical.empty:
            return source_team_id, 0.0

        base_ids = set(season_canonical["team_id"].astype(str))
        if source_team_id in base_ids:
            return source_team_id, 1.0

        normalized_to_id = {}
        key_to_ids: Dict[str, List[str]] = {}
        for row in season_canonical.itertuples(index=False):
            norm = self._normalize_name(row.team_name)
            if norm and norm not in normalized_to_id:
                normalized_to_id[norm] = row.team_id
            key = self._team_key(row.team_name)
            if key:
                key_to_ids.setdefault(key, []).append(row.team_id)

        src_norm = self._normalize_name(source_name)
        src_key = self._team_key(source_name)
        if src_norm in normalized_to_id:
            return normalized_to_id[src_norm], 0.99
        if src_key in key_to_ids and len(key_to_ids[src_key]) == 1:
            return key_to_ids[src_key][0], 0.95

        # heuristic: key containment
        if src_key:
            for key, ids in key_to_ids.items():
                if len(ids) != 1:
                    continue
                if src_key in key or key in src_key:
                    return ids[0], 0.9

        # fallback fuzzy matching
        candidates = season_canonical[["team_id", "team_name"]].drop_duplicates()
        best_id = source_team_id
        best_score = 0.0
        for row in candidates.itertuples(index=False):
            score = difflib.SequenceMatcher(None, src_norm, self._normalize_name(row.team_name)).ratio()
            if score > best_score:
                best_score = score
                best_id = row.team_id
        if best_score >= 0.84:
            return str(best_id), float(best_score)
        return source_team_id, float(best_score)

    def _fallback_team_game_rows(self, game: Dict, season: int) -> List[Dict]:
        game_id = game.get("game_id") or game.get("id")
        t1_id = game.get("team1_id") or self._normalize_team_id(game.get("team1_name"))
        t2_id = game.get("team2_id") or self._normalize_team_id(game.get("team2_name"))
        if not game_id or not t1_id or not t2_id:
            return []
        date_value = game.get("date") or game.get("game_date") or f"{season-1}-11-01"
        team1_score = self._to_float(game.get("team1_score"))
        team2_score = self._to_float(game.get("team2_score"))
        return [
            {
                "game_id": game_id,
                "season": season,
                "date": date_value,
                "team_id": t1_id,
                "team_name": game.get("team1_name", t1_id),
                "opponent_id": t2_id,
                "opponent_name": game.get("team2_name", t2_id),
                "team_score": team1_score,
                "opponent_score": team2_score,
            },
            {
                "game_id": game_id,
                "season": season,
                "date": date_value,
                "team_id": t2_id,
                "team_name": game.get("team2_name", t2_id),
                "opponent_id": t1_id,
                "opponent_name": game.get("team1_name", t1_id),
                "team_score": team2_score,
                "opponent_score": team1_score,
            },
        ]

    @staticmethod
    def _rolling_day_counts(dates: pd.Series, window_days: int) -> List[float]:
        arr = dates.sort_values().astype("datetime64[ns]").view("int64")
        arr = (arr // 86_400_000_000_000).to_numpy()  # day index
        out = np.zeros(len(arr), dtype=float)
        left = 0
        for idx in range(len(arr)):
            while left < idx and arr[idx] - arr[left] > window_days:
                left += 1
            out[idx] = float(idx - left)
        # map back original order
        reorder = pd.Series(out, index=dates.sort_values().index)
        return reorder.reindex(dates.index).fillna(0.0).tolist()

    @staticmethod
    def _shifted_expanding_mean(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding().mean()

    @staticmethod
    def _safe_div(numerator, denominator):
        num = pd.to_numeric(numerator, errors="coerce")
        if isinstance(num, pd.Series):
            num_series = num
        else:
            num_series = pd.Series(num)

        den = pd.to_numeric(denominator, errors="coerce")
        if isinstance(den, pd.Series):
            den_series = den.reindex(num_series.index)
        else:
            den_series = pd.Series([den] * len(num_series), index=num_series.index)
        return pd.Series(
            np.where((den_series > 0) & den_series.notna(), num_series / den_series, np.nan),
            index=num_series.index,
        )

    @staticmethod
    def _normalize_team_id(name) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name or "")).strip("_")

    @staticmethod
    def _normalize_name(name: str) -> str:
        clean = re.sub(r"[^a-z0-9 ]+", " ", str(name or "").lower().replace("&", " and "))
        tokens = [t for t in clean.split() if t not in {"the", "university", "college", "at", "of"}]
        return " ".join(tokens)

    @classmethod
    def _team_key(cls, name: str) -> str:
        norm = cls._normalize_name(name)
        tokens = norm.split()
        if len(tokens) <= 2:
            return norm
        # remove likely mascot suffix (1-2 trailing tokens) for ESPN-style names.
        base1 = " ".join(tokens[:-1]).strip()
        base2 = " ".join(tokens[:-2]).strip()
        if base2 and len(base2) >= 4:
            return base2
        return base1 or norm

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return np.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _coalesce_numeric(row: Dict, *keys: str) -> float:
        for key in keys:
            if key not in row:
                continue
            try:
                return float(row.get(key))
            except (TypeError, ValueError):
                continue
        return np.nan
