"""Open-data proxy for ShotQuality-style possession xP features."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..models.game_flow import Possession, PossessionOutcome, ShotType


@dataclass
class _TeamGameRow:
    game_id: str
    team_id: str
    team_name: str
    opponent_id: str
    opponent_name: str
    team_score: float
    opponent_score: float
    possessions: float
    fga: float
    fg3a: float
    fta: float
    turnovers: float
    orb: float
    game_date: str
    location_weight: float


class OpenShotQualityProxyBuilder:
    """
    Build ShotQuality-compatible team and possession-level game payloads from open boxscore feeds.

    This is a deterministic proxy when proprietary ShotQuality endpoints are unavailable.
    """

    def build(self, team_game_rows: List[Dict]) -> Optional[Dict]:
        rows = self._normalize_rows(team_game_rows)
        if not rows:
            return None

        games = self._build_games(rows)
        if not games:
            return None

        team_metrics = self._build_team_metrics(games)
        if not team_metrics:
            return None

        return {
            "teams": team_metrics,
            "games": games,
            "metadata": {
                "source": "open_boxscore_proxy",
                "method": "synthetic_possession_xp_from_team_boxscores",
                "assumptions": [
                    "Shot type distribution derived from team FGA/3PA/FTA profile",
                    "Possession-level xP computed via shot-type priors",
                    "Actual points stochastically allocated to preserve observed final score",
                ],
            },
        }

    def _normalize_rows(self, team_game_rows: List[Dict]) -> List[_TeamGameRow]:
        out: List[_TeamGameRow] = []
        for raw in team_game_rows:
            if not isinstance(raw, dict):
                continue
            game_id = str(raw.get("game_id") or raw.get("id") or "").strip()
            team_id = self._team_id(str(raw.get("team_id") or raw.get("team_name") or ""))
            opponent_id = self._team_id(str(raw.get("opponent_id") or raw.get("opponent_name") or ""))
            if not game_id or not team_id or not opponent_id:
                continue
            row = _TeamGameRow(
                game_id=game_id,
                team_id=team_id,
                team_name=str(raw.get("team_name") or raw.get("team_id") or team_id),
                opponent_id=opponent_id,
                opponent_name=str(raw.get("opponent_name") or raw.get("opponent_id") or opponent_id),
                team_score=self._to_float(raw.get("team_score")),
                opponent_score=self._to_float(raw.get("opponent_score")),
                possessions=max(self._to_float(raw.get("possessions")), 1.0),
                fga=max(self._to_float(raw.get("fga")), 1.0),
                fg3a=max(self._to_float(raw.get("fg3a")), 0.0),
                fta=max(self._to_float(raw.get("fta")), 0.0),
                turnovers=max(self._to_float(raw.get("turnovers")), 0.0),
                orb=max(self._to_float(raw.get("orb")), 0.0),
                game_date=str(raw.get("date") or raw.get("game_date") or ""),
                location_weight=0.5 if bool(raw.get("neutral_site", False)) else 1.0,
            )
            out.append(row)
        return out

    def _build_games(self, rows: List[_TeamGameRow]) -> List[Dict]:
        by_game: Dict[str, List[_TeamGameRow]] = defaultdict(list)
        for row in rows:
            by_game[row.game_id].append(row)

        games: List[Dict] = []
        for game_id, candidates in by_game.items():
            # Keep one row per team and choose the pair with the fullest stat coverage.
            dedup = {}
            for row in candidates:
                key = row.team_id
                current = dedup.get(key)
                if current is None or row.fga > current.fga:
                    dedup[key] = row
            pair = sorted(dedup.values(), key=lambda r: (-r.fga, r.team_id))
            if len(pair) < 2:
                continue
            t1, t2 = pair[:2]

            poss1 = max(int(round(t1.possessions)), 1)
            poss2 = max(int(round(t2.possessions)), 1)

            rng1 = self._rng(game_id, t1.team_id)
            rng2 = self._rng(game_id, t2.team_id)
            team1_poss = self._synthesize_team_possessions(t1, poss1, rng1)
            team2_poss = self._synthesize_team_possessions(t2, poss2, rng2)

            merged = self._merge_possessions(team1_poss, team2_poss)
            if not merged:
                continue
            games.append(
                {
                    "game_id": game_id,
                    "team_id": t1.team_id,
                    "opponent_id": t2.team_id,
                    "game_date": t1.game_date or t2.game_date,
                    "location_weight": t1.location_weight,
                    "possessions": merged,
                }
            )
        return games

    def _build_team_metrics(self, games: List[Dict]) -> List[Dict]:
        team_totals: Dict[str, Dict] = {}
        for game in games:
            team_a = str(game.get("team_id", ""))
            team_b = str(game.get("opponent_id", ""))
            poss = [p for p in game.get("possessions", []) if isinstance(p, dict)]
            if not team_a or not team_b or not poss:
                continue

            a_poss = [p for p in poss if p.get("team_id") == team_a]
            b_poss = [p for p in poss if p.get("team_id") == team_b]
            if not a_poss or not b_poss:
                continue

            self._accumulate_team(team_totals, team_a, team_a, a_poss, b_poss)
            self._accumulate_team(team_totals, team_b, team_b, b_poss, a_poss)

        out = []
        for team_id, stats in team_totals.items():
            gp = max(stats["games"], 1)
            poss_count = max(stats["possessions"], 1.0)
            out.append(
                {
                    "team_id": team_id,
                    "team_name": stats["team_name"],
                    "offensive_xp_per_possession": stats["off_xp"] / poss_count,
                    "defensive_xp_per_possession": stats["def_xp"] / poss_count,
                    "rim_rate": stats["rim"] / poss_count,
                    "three_rate": stats["three"] / poss_count,
                    "midrange_rate": stats["mid"] / poss_count,
                    "games": gp,
                }
            )
        out.sort(key=lambda row: row["team_name"])
        return out

    def _accumulate_team(self, totals: Dict[str, Dict], team_id: str, team_name: str, team_poss: List[Dict], opp_poss: List[Dict]) -> None:
        bucket = totals.setdefault(
            team_id,
            {
                "team_name": team_name,
                "games": 0,
                "possessions": 0.0,
                "off_xp": 0.0,
                "def_xp": 0.0,
                "rim": 0.0,
                "three": 0.0,
                "mid": 0.0,
            },
        )
        bucket["games"] += 1
        for poss in team_poss:
            xp = self._to_float(poss.get("xp"))
            bucket["possessions"] += 1.0
            bucket["off_xp"] += xp
            shot_type = str(poss.get("shot_type", ""))
            if shot_type == ShotType.RIM.value:
                bucket["rim"] += 1.0
            elif shot_type in {ShotType.CORNER_THREE.value, ShotType.ABOVE_BREAK_THREE.value}:
                bucket["three"] += 1.0
            else:
                bucket["mid"] += 1.0
        for poss in opp_poss:
            bucket["def_xp"] += self._to_float(poss.get("xp"))

    def _synthesize_team_possessions(self, row: _TeamGameRow, n_possessions: int, rng: np.random.Generator) -> List[Dict]:
        three_rate = np.clip(row.fg3a / max(row.fga, 1.0), 0.15, 0.55)
        ft_rate = np.clip(row.fta / max(row.fga, 1.0), 0.10, 0.55)
        rim_rate = np.clip(0.28 + 0.55 * (ft_rate - 0.22), 0.18, 0.60)
        mid_rate = max(1.0 - three_rate - rim_rate, 0.05)
        # Renormalize.
        total = three_rate + rim_rate + mid_rate
        three_rate /= total
        rim_rate /= total
        mid_rate /= total

        n_three = int(round(three_rate * n_possessions))
        n_rim = int(round(rim_rate * n_possessions))
        n_mid = max(n_possessions - n_three - n_rim, 0)
        while n_three + n_rim + n_mid < n_possessions:
            n_mid += 1

        shot_types: List[ShotType] = (
            [ShotType.ABOVE_BREAK_THREE] * n_three
            + [ShotType.RIM] * n_rim
            + [ShotType.SHORT_MIDRANGE] * n_mid
        )
        if len(shot_types) > n_possessions:
            shot_types = shot_types[:n_possessions]
        if len(shot_types) < n_possessions:
            shot_types.extend([ShotType.SHORT_MIDRANGE] * (n_possessions - len(shot_types)))

        rng.shuffle(shot_types)
        actual_points = self._allocate_points(int(round(row.team_score)), n_possessions, rng)

        out: List[Dict] = []
        for i, shot_type in enumerate(shot_types):
            contested = bool(rng.random() < 0.45)
            xp = float(Possession.calculate_xp(shot_type, is_contested=contested))
            points = int(actual_points[i])
            if points > 0:
                outcome = PossessionOutcome.MADE_SHOT.value
            else:
                outcome = PossessionOutcome.MISSED_SHOT.value
            out.append(
                {
                    "team_id": row.team_id,
                    "shot_type": shot_type.value,
                    "is_contested": contested,
                    "xp": xp,
                    "actual_points": points,
                    "outcome": outcome,
                }
            )
        return out

    def _merge_possessions(self, team1_poss: List[Dict], team2_poss: List[Dict]) -> List[Dict]:
        n_total = len(team1_poss) + len(team2_poss)
        if n_total == 0:
            return []
        merged: List[Dict] = []
        i = 0
        j = 0
        possession_id = 0
        period = 1
        # Alternate possessions to create realistic lead paths.
        while i < len(team1_poss) or j < len(team2_poss):
            if i < len(team1_poss):
                row = dict(team1_poss[i])
                i += 1
                possession_id += 1
                row["possession_id"] = f"poss_{possession_id}"
                row["period"] = period
                row["game_clock"] = max(0.0, 1200.0 - 17.0 * possession_id)
                merged.append(row)
            if j < len(team2_poss):
                row = dict(team2_poss[j])
                j += 1
                possession_id += 1
                row["possession_id"] = f"poss_{possession_id}"
                row["period"] = period
                row["game_clock"] = max(0.0, 1200.0 - 17.0 * possession_id)
                merged.append(row)
            if possession_id >= 70:
                period = 2
        return merged

    @staticmethod
    def _allocate_points(total_points: int, n_possessions: int, rng: np.random.Generator) -> List[int]:
        if n_possessions <= 0:
            return []
        if total_points <= 0:
            return [0] * n_possessions

        points = [0] * n_possessions
        remaining = int(total_points)
        attempts = 0
        max_attempts = max(total_points * 5, 100)
        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            idx = int(rng.integers(0, n_possessions))
            if points[idx] >= 3:
                continue
            points[idx] += 1
            remaining -= 1
        if remaining > 0:
            for idx in range(n_possessions):
                if remaining <= 0:
                    break
                capacity = 3 - points[idx]
                if capacity <= 0:
                    continue
                add = min(capacity, remaining)
                points[idx] += add
                remaining -= add
        return points

    @staticmethod
    def _rng(game_id: str, team_id: str) -> np.random.Generator:
        seed_bytes = hashlib.sha256(f"{game_id}|{team_id}".encode("utf-8")).digest()[:8]
        seed = int.from_bytes(seed_bytes, byteorder="little", signed=False)
        return np.random.default_rng(seed)

    @staticmethod
    def _team_id(name: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in (name or "")).strip("_")

    @staticmethod
    def _to_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
