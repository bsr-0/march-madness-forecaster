"""Player-level roster/metrics scraper for RAPM-ready inputs."""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests


class PlayerMetricsScraper:
    """Fetches player-level team rosters from JSON or CSV endpoints."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            }
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_rosters(self, year: int, source_url: Optional[str] = None, fmt: str = "json") -> Dict:
        cache_name = f"rosters_{year}.json"
        cached = self._load_cache(cache_name)
        if cached and isinstance(cached.get("teams"), list) and cached["teams"]:
            return cached

        url = source_url or os.getenv("PLAYER_METRICS_URL")
        if not url:
            return {}

        response = self.session.get(url, timeout=45)
        response.raise_for_status()

        if fmt.lower() == "csv":
            teams = self._parse_csv(response.text)
            payload = {"teams": teams}
        else:
            raw = response.json()
            payload = self._normalize_json_payload(raw)

        if not isinstance(payload.get("teams"), list) or not payload["teams"]:
            raise ValueError("Player metrics source returned no roster teams")

        if not payload.get("timestamp"):
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        payload["source"] = payload.get("source", "player_metrics")

        self._save_cache(cache_name, payload)
        return payload

    def _normalize_json_payload(self, raw: object) -> Dict:
        if isinstance(raw, list):
            return {"teams": raw}
        if isinstance(raw, dict):
            teams = raw.get("teams")
            if isinstance(teams, list):
                return {
                    "teams": teams,
                    "timestamp": raw.get("timestamp") or raw.get("generated_at") or raw.get("updated_at"),
                    "source": raw.get("source", "player_metrics"),
                }
        return {"teams": []}

    def _parse_csv(self, text: str) -> List[Dict]:
        reader = csv.DictReader(io.StringIO(text))
        by_team: Dict[str, Dict] = {}

        for row in reader:
            team_id = (row.get("team_id") or "").strip()
            team_name = (row.get("team_name") or row.get("team") or team_id).strip()
            if not team_id and not team_name:
                continue
            if not team_id:
                team_id = self._team_id(team_name)

            bucket = by_team.setdefault(
                team_id,
                {
                    "team_id": team_id,
                    "team_name": team_name,
                    "players": [],
                },
            )

            player_name = (row.get("name") or row.get("player_name") or "").strip()
            player_id = (row.get("player_id") or row.get("id") or "").strip() or self._team_player_id(team_id, player_name)
            if not player_name:
                continue

            bucket["players"].append(
                {
                    "player_id": player_id,
                    "name": player_name,
                    "position": row.get("position") or "PG",
                    "minutes_per_game": self._to_float(row.get("minutes_per_game") or row.get("mpg")),
                    "games_played": self._to_int(row.get("games_played") or row.get("gp")),
                    "games_started": self._to_int(row.get("games_started") or row.get("gs")),
                    "rapm_offensive": self._to_float(row.get("rapm_offensive")),
                    "rapm_defensive": self._to_float(row.get("rapm_defensive")),
                    "warp": self._to_float(row.get("warp")),
                    "box_plus_minus": self._to_float(row.get("box_plus_minus") or row.get("bpm")),
                    "usage_rate": self._to_float(row.get("usage_rate") or row.get("usg")),
                    "injury_status": row.get("injury_status") or "healthy",
                    "is_transfer": self._to_bool(row.get("is_transfer")),
                    "transfer_from": row.get("transfer_from"),
                    "eligibility_year": self._to_int(row.get("eligibility_year") or row.get("class_year") or 1),
                }
            )

        return list(by_team.values())

    @staticmethod
    def _team_id(name: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in (name or "")).strip("_")

    @staticmethod
    def _team_player_id(team_id: str, name: str) -> str:
        player = "".join(ch.lower() if ch.isalnum() else "_" for ch in (name or "")).strip("_")
        return f"{team_id}_{player or 'player'}"

    @staticmethod
    def _to_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _to_int(value) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _load_cache(self, filename: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / filename
        if not p.exists():
            return None
        try:
            with open(p, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def _save_cache(self, filename: str, payload: Dict) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / filename
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
