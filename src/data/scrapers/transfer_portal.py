"""Transfer portal scraper utilities."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import requests


class TransferPortalScraper:
    """Fetches transfer portal entries from JSON or CSV endpoints."""

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

    def fetch_entries(self, year: int, source_url: str, fmt: str = "json") -> List[Dict]:
        cache_name = f"transfer_portal_{year}.json"
        cached = self._load_cache(cache_name)
        if cached:
            entries = cached.get("entries", [])
            if entries:
                return entries

        response = self.session.get(source_url, timeout=30)
        response.raise_for_status()

        if fmt.lower() == "csv":
            entries = self._parse_csv(response.text)
        else:
            payload = response.json()
            entries = payload.get("entries", payload)

        if not isinstance(entries, list) or not entries:
            raise ValueError("Transfer portal source returned no entries")

        self._save_cache(cache_name, {"entries": entries})
        return entries

    def _parse_csv(self, csv_text: str) -> List[Dict]:
        reader = csv.DictReader(io.StringIO(csv_text))
        out: List[Dict] = []
        for row in reader:
            out.append(
                {
                    "player_id": row.get("player_id") or row.get("id") or "",
                    "player_name": row.get("player_name") or row.get("name") or "",
                    "source_team_name": row.get("source_team_name") or row.get("from_team") or "",
                    "destination_team_name": row.get("destination_team_name") or row.get("to_team") or "",
                    "entry_date": row.get("entry_date") or "",
                }
            )
        return out

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

    def _save_cache(self, filename: str, data: Dict) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / filename
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
