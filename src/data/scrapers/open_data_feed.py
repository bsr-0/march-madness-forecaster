"""Generic open-data endpoint scraper for JSON/CSV feeds."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import requests


class OpenDataFeedScraper:
    """Fetches open/public endpoint payloads with cache support."""

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

    def fetch_records(
        self,
        cache_name: str,
        source_url: str,
        fmt: str = "json",
        records_key: Optional[str] = None,
    ) -> List[Dict]:
        cached = self._load_cache(cache_name)
        if cached:
            records = cached.get("records", [])
            if isinstance(records, list) and records:
                return [r for r in records if isinstance(r, dict)]

        response = self.session.get(source_url, timeout=45)
        response.raise_for_status()

        if fmt.lower() == "csv":
            records = self._parse_csv(response.text)
        else:
            payload = response.json()
            records = self._extract_records(payload, records_key)

        if not records:
            raise ValueError(f"Open feed returned no records for url={source_url}")

        self._save_cache(cache_name, {"records": records})
        return records

    def _extract_records(self, payload, records_key: Optional[str]) -> List[Dict]:
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        if not isinstance(payload, dict):
            return []

        if records_key:
            rows = payload.get(records_key, [])
            return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

        for key in ("records", "teams", "entries", "games", "polls"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        return []

    def _parse_csv(self, csv_text: str) -> List[Dict]:
        reader = csv.DictReader(io.StringIO(csv_text))
        return [dict(row) for row in reader]

    def _load_cache(self, filename: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / filename
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def _save_cache(self, filename: str, payload: Dict) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / filename
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
