# Torvik Pipeline Interface Specification

**Date**: 2026-03-23
**Purpose**: Defines the exact public interface contract that any replacement scraper must satisfy. Derived from exhaustive code reading of all 90+ consumer files.

---

## 1. Exported Symbols (from `src/data/scrapers/torvik.py`)

Any replacement module at `src/data/scrapers/torvik.py` MUST export these names:

```python
# Re-exported via src/data/scrapers/__init__.py
from .torvik import BartTorvikScraper   # __init__.py:23
from .torvik import TorVikTeam          # Used directly by 15+ files
from .torvik import TorVikValidator     # Used by data_loader.py, test files
from .torvik import TorVikValidationError  # Used by test files, validator
```

---

## 2. TorVikTeam Dataclass

### 2a. Required Fields (all consumers depend on these)

```python
@dataclass
class TorVikTeam:
    # Identity (used by data_loader.py for team matching)
    team_id: str          # snake_case canonical ID (e.g. "duke", "north_carolina")
    name: str             # Display name (e.g. "Duke", "North Carolina")
    conference: str       # Conference abbreviation (e.g. "ACC", "B10")

    # T-Rank ratings (used by data_loader.py → torvik_map)
    t_rank: int           # 1-400 ranking
    barthag: float        # Expected win% vs average team (0.0-1.0)

    # Core efficiency (primary pipeline features)
    adj_offensive_efficiency: float  # Per-100-poss adjusted offense (60-140)
    adj_defensive_efficiency: float  # Per-100-poss adjusted defense (60-140)
    adj_tempo: float                 # Possessions per 40 min (55-90)

    # Four Factors — Offense (fed directly to feature_engineering.py via _FF_FIELD_ORDER)
    effective_fg_pct: float     # eFG% (0.30-0.75), may be math.nan if unavailable
    turnover_rate: float        # TO% (0.05-0.40), may be math.nan
    offensive_reb_rate: float   # ORB% (0.0-0.60), may be math.nan
    free_throw_rate: float      # FTR = FT/FGA (0.0-0.80), may be math.nan

    # Four Factors — Defense
    opp_effective_fg_pct: float   # Opponent eFG% (0.30-0.75), may be math.nan
    opp_turnover_rate: float      # Forced TO% (0.05-0.40), may be math.nan
    defensive_reb_rate: float     # DRB% (0.0-1.0), may be math.nan
    opp_free_throw_rate: float    # Opponent FTR (0.0-0.80), may be math.nan

    # Shooting splits (default 0.0 = "not available")
    two_pt_pct: float = 0.0
    three_pt_pct: float = 0.0
    three_pt_rate: float = 0.0    # % of shots from 3
    ft_pct: float = 0.0

    # Defensive opponent shooting
    opp_two_pt_pct: float = 0.0
    opp_three_pt_pct: float = 0.0
    opp_three_pt_rate: float = 0.0

    # Block/steal rates
    block_pct: float = 0.0
    steal_pct: float = 0.0

    # WAB + Record
    wab: float = 0.0              # Wins Above Bubble
    wins: int = 0
    losses: int = 0
    conf_wins: int = 0
    conf_losses: int = 0
```

### 2b. Required Method

```python
def to_dict(self) -> dict:
    """Convert to dictionary with ALL field names as keys.

    MUST include at minimum these keys (data_loader.py reads them directly):
        team_id, name, conference, t_rank, barthag,
        adj_offensive_efficiency, adj_defensive_efficiency, adj_tempo,
        effective_fg_pct, turnover_rate, offensive_reb_rate, free_throw_rate,
        opp_effective_fg_pct, opp_turnover_rate, defensive_reb_rate, opp_free_throw_rate,
        two_pt_pct, three_pt_pct, three_pt_rate, ft_pct,
        opp_two_pt_pct, opp_three_pt_pct, opp_three_pt_rate,
        block_pct, steal_pct,
        wab, wins, losses, conf_wins, conf_losses
    """
```

### 2c. Convention: math.nan vs 0.0

- **Four Factors** (8 fields): May be `math.nan` when unavailable from CSV fallback. Feature engineering handles this via seed-conditional priors.
- **Shooting/extended fields**: Use `0.0` to mean "not available". Downstream code checks `val != 0.0` or `val or 0`.

---

## 3. BartTorvikScraper Class

### 3a. Constructor

```python
class BartTorvikScraper:
    BASE_URL = "https://barttorvik.com"         # Referenced by tests
    CBBSTAT_API = "https://api.cbbstat.com"     # Referenced by tests

    def __init__(
        self,
        cache_dir: Optional[str] = None,          # Path to cache directory (or None for no cache)
        cache_ttl_seconds: float = 21600,          # 6 hours default (DEFAULT_CACHE_TTL_SECONDS)
        circuit_breaker_state_file: Optional[str] = None,  # For CB persistence
        strict_leakage: bool = False,              # Raise LeakageError after tournament start
    ):
```

**Call sites**:
- `BartTorvikScraper()` — no args (data_loader.py:768, :801; tests)
- `BartTorvikScraper(cache_dir="data/raw")` — with cache dir (historical tests)
- `BartTorvikScraper(str(self.cache_dir))` — collector.py:172
- `BartTorvikScraper(cache_dir=config.data_cache_dir, strict_leakage=True)` — data_loader.py:770
- `BartTorvikScraper(cache_dir=str(tmp_path))` — tests
- `BartTorvikScraper(cache_ttl_seconds=60)` — tests
- `BartTorvikScraper(strict_leakage=True)` — leakage guard tests
- `BartTorvikScraper.__new__(BartTorvikScraper)` — leakage tests (bypasses __init__)

### 3b. Public Methods

#### `fetch_current_rankings(year, strict) → List[TorVikTeam]`

```python
def fetch_current_rankings(self, year: int = 2026, strict: bool = False) -> List[TorVikTeam]:
    """Fetch T-Rank ratings for all D1 teams.

    Strategy chain:
      1. Cache lookup (with TTL + content validation)
      2. cbbstat API (preferred — ratings + Four Factors)
      3. CSV fallback (ratings only, Four Factors = nan)

    Args:
        year: Season year (e.g. 2026)
        strict: If True, raise TorVikValidationError when any team has
                zero/NaN critical fields.

    Returns:
        List of TorVikTeam objects (typically 350-365 teams)
    """
```

**Call sites**:
- `data_loader.py:773` — `BartTorvikScraper(cache_dir=..., strict_leakage=...).fetch_current_rankings(config.year, strict=True)`
- Tests — `scraper.fetch_current_rankings(2026)`

#### `fetch_four_factors(year) → Dict[str, Dict]`

```python
def fetch_four_factors(self, year: int = 2026) -> Dict[str, Dict]:
    """Fetch Four Factors for all teams.

    Returns:
        Dict of team_id → {
            'effective_fg_pct': float,    # 0.30-0.75
            'turnover_rate': float,       # 0.05-0.40
            'offensive_reb_rate': float,  # 0.0-0.60
            'free_throw_rate': float,     # 0.0-0.80
            'opp_effective_fg_pct': float,
            'opp_turnover_rate': float,
            'defensive_reb_rate': float,
            'opp_free_throw_rate': float,
        }
    """
```

**Call sites**:
- `collector.py:173` — `torvik_scraper.fetch_four_factors(year)`
- `scripts/run_conference_predictions.py:50`
- Tests

#### `fetch_shooting_stats(year) → Dict[str, Dict]`

```python
def fetch_shooting_stats(self, year: int = 2026) -> Dict[str, Dict]:
    """Fetch shooting splits (3PT%, FT%) for all teams.

    Returns:
        Dict of team_id → {
            'three_pt_pct': float,  # 0.0-0.60
            'ft_pct': float,        # 0.50-0.90
        }
    """
```

**Call sites**:
- `collector.py:190` — `torvik_scraper.fetch_shooting_stats(year)`
- `scripts/run_conference_predictions.py:65`

#### `load_from_json(filepath) → List[TorVikTeam]`

```python
def load_from_json(self, filepath: str) -> List[TorVikTeam]:
    """Load from JSON file with schema: {"teams": [{field: value, ...}, ...]}

    Returns:
        List of TorVikTeam objects
    """
```

**Call sites**:
- `data_loader.py:768` — `BartTorvikScraper().load_from_json(config.torvik_json)`

#### `_dict_to_team(data) → TorVikTeam`

```python
def _dict_to_team(self, data: dict) -> TorVikTeam:
    """Convert a flat dict (with TorVikTeam field names as keys) to TorVikTeam.

    Uses safe defaults for missing keys:
        adj_offensive_efficiency: 100.0, adj_defensive_efficiency: 100.0,
        adj_tempo: 68.0, barthag: 0.5, t_rank: 999,
        effective_fg_pct: 0.5, turnover_rate: 0.18, offensive_reb_rate: 0.30,
        free_throw_rate: 0.30, opp_effective_fg_pct: 0.5, etc.
    """
```

**Call sites** (NOTE: called as `BartTorvikScraper()._dict_to_team(t)` — technically private but used externally):
- `data_loader.py:801`

#### `data_completeness_report(teams) → Dict[str, float]` (static)

```python
@staticmethod
def data_completeness_report(teams: List[TorVikTeam]) -> Dict[str, float]:
    """Return per-field completeness fraction (0.0-1.0) for diagnostic purposes."""
```

**Call sites**: Tests only

#### `_normalize_team_name_to_id(name) → str` (static)

```python
@staticmethod
def _normalize_team_name_to_id(name: str) -> str:
    """Convert display name to snake_case canonical ID.
    Delegates to src.data.normalize.normalize_team_id()."""
```

**Call sites**: Tests, providers.py (via `_map_barttorvik_row`)

#### `fetch_strategy` property

```python
@property
def fetch_strategy(self) -> Dict[str, str]:
    """Return telemetry dict of which strategy was used per data type.
    e.g. {"rankings": "cbbstat_api", "four_factors": "csv_fallback"}"""
```

### 3c. Internal Methods Referenced Externally

These are nominally private but called from tests or other modules:

| Method | External Caller |
|--------|----------------|
| `_shrink_csv_rate(raw, min_pct, pop)` | `test_torvik_pipeline_rigor.py` (5 tests) |
| `_POP_PRIORS` dict | `test_torvik_pipeline_rigor.py` |
| `_PRIOR_STRENGTH` constant | `test_torvik_pipeline_rigor.py` |
| `_check_tournament_date_guard(year, strict)` | `test_leakage_guards.py` (5 tests) |
| `_cache_has_valid_four_factors(cached)` | `test_torvik_refactor.py` |
| `_cache_has_valid_rankings(cached)` | `test_torvik_refactor.py` |
| `_aggregate_player_csv(csv_text)` | `test_torvik_pipeline_rigor.py` |

---

## 4. TorVikValidator Class

```python
class TorVikValidator:
    RANGES: Dict[str, Tuple[float, float]]  # Field → (min, max) expected ranges
    CRITICAL_FIELDS: Set[str]  # {"adj_offensive_efficiency", "adj_defensive_efficiency", "barthag"}

    @classmethod
    def validate_team(cls, team: TorVikTeam, strict: bool = False) -> List[str]:
        """Validate one team. Returns list of warning strings.
        If strict=True, raises TorVikValidationError when CRITICAL_FIELDS are zero/NaN."""

    @classmethod
    def validate_teams(cls, teams: List[TorVikTeam], strict: bool = False) -> Dict[str, List[str]]:
        """Validate multiple teams. Returns {team_id: [warnings]}."""

    @classmethod
    def validate_four_factors(cls, data: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Validate Four Factors dict. Returns {team_id: [warnings]}."""
```

**Call sites**:
- `data_loader.py:829` — `TorVikValidator.validate_teams(torvik_teams, strict=True)`
- `torvik.py` internal — after fetch_current_rankings, fetch_four_factors
- Tests (6+ test classes)

---

## 5. TorVikValidationError

```python
class TorVikValidationError(Exception):
    """Raised in strict mode when critical data is missing or invalid."""
```

**Call sites**: Validator, tests (`pytest.raises(TorVikValidationError, match=...)`)

---

## 6. JSON File Schemas

### 6a. Rankings file (`torvik_{year}.json`, `torvik_rankings_{year}.json`)

```json
{
  "teams": [
    {
      "team_id": "duke",
      "team_name": "Duke",
      "name": "Duke",
      "conference": "ACC",
      "t_rank": 1,
      "barthag": 0.9813,
      "adj_offensive_efficiency": 128.15,
      "adj_defensive_efficiency": 90.82,
      "adj_tempo": 65.80,
      "effective_fg_pct": 0.0,
      "turnover_rate": 0.0,
      "offensive_reb_rate": 0.0,
      "free_throw_rate": 0.0,
      "opp_effective_fg_pct": 0.4306,
      "opp_turnover_rate": 0.1297,
      "opp_free_throw_rate": 0.1965,
      "two_pt_pct": 0.0,
      "three_pt_pct": 0.0,
      "three_pt_rate": 0.0,
      "ft_pct": 0.0,
      "block_pct": 0.0,
      "steal_pct": 0.0,
      "opp_two_pt_pct": 0.0,
      "opp_three_pt_pct": 0.0,
      "opp_three_pt_rate": 0.0,
      "wab": 0.0,
      "wins": 0,
      "losses": 0,
      "conf_wins": 0,
      "conf_losses": 0,
      "enriched_stats": {}
    }
  ],
  "timestamp": "2026-03-23T14:00:00"
}
```

**Notes**:
- `team_name` and `name` are BOTH present (some consumers read `name`, others `team_name`)
- Four Factors may be `0.0` when CSV fallback was used (enriched later from `torvik_four_factors_{year}.json`)
- `enriched_stats` is optional sub-dict added by defensive FF enrichment

### 6b. Four Factors file (`torvik_four_factors_{year}.json`)

```json
{
  "team_id": {
    "effective_fg_pct": 0.5383,
    "turnover_rate": 0.1703,
    "offensive_reb_rate": 0.2383,
    "free_throw_rate": 0.2961,
    "opp_effective_fg_pct": 0.5417,
    "opp_turnover_rate": 0.154,
    "defensive_reb_rate": 0.6854,
    "opp_free_throw_rate": 0.3906
  }
}
```

**Schema**: Flat dict of `team_id → {8 Four Factors fields}`. No wrapper object. All values are floats (0.0-1.0 range, as fractions not percentages).

### 6c. Shooting file (`torvik_shooting_{year}.json`)

```json
{
  "team_id": {
    "ft_pct": 0.7266,
    "three_pt_pct": 0.3396
  }
}
```

**Schema**: Flat dict of `team_id → {ft_pct, three_pt_pct}`. No wrapper.

### 6d. Cache wrapper format (internal to scraper)

```json
{
  "_cache_schema_version": 3,
  "_cache_timestamp": 1711190400.0,
  "_cache_data": { ... actual data ... }
}
```

Legacy files without wrapper are also accepted (backward compat).

---

## 7. torvik_map Dict Schema

Built by `data_loader.py` from TorVikTeam objects, this dict is passed through the entire pipeline:

```python
torvik_map: Dict[str, Dict] = {
    "duke": {
        # Four Factors (8 fields) — read by feature_engineering.py via _FF_FIELD_ORDER
        "effective_fg_pct": 0.535,
        "turnover_rate": 0.170,
        "offensive_reb_rate": 0.320,
        "free_throw_rate": 0.340,
        "opp_effective_fg_pct": 0.470,
        "opp_turnover_rate": 0.200,
        "defensive_reb_rate": 0.740,
        "opp_free_throw_rate": 0.290,

        # Efficiency (read by data_loader backfill logic)
        "adj_offensive_efficiency": 128.15,
        "adj_defensive_efficiency": 90.82,
        "adj_tempo": 65.80,
        "barthag": 0.981,
        "t_rank": 1,

        # Shooting (read by feature_engineering.py for opp_two_pt_pct fallback)
        "two_pt_pct": 0.52,
        "three_pt_pct": 0.38,
        "three_pt_rate": 0.35,
        "ft_pct": 0.72,
        "opp_two_pt_pct": 0.45,
        "opp_three_pt_pct": 0.31,
        "opp_three_pt_rate": 0.33,

        # Record
        "wab": 5.2,
        "wins": 28,
        "losses": 3,
        "conference": "ACC",
        "conf_wins": 16,
        "conf_losses": 2,

        # Context (added by enrich_tournament_context, NOT from scraper directly)
        "preseason_ap_rank": 3,
        "coach_tournament_appearances": 15,
        "coach_tournament_win_rate": 0.72,
        "coach_deep_run_rate": 0.40,
        "coach_stage_consistency": 0.65,
        "coach_f4_appearances": 6,
        "coach_e8_appearances": 10,
        "coach_s16_appearances": 12,
        "conf_tourney_champion": 1.0,
    }
}
```

**Key**: The context fields (`preseason_ap_rank`, `coach_*`, `conf_tourney_champion`) are injected INTO `torvik_map` by `enrich_tournament_context()` in `data_loader.py`, not by the scraper itself. But `feature_engineering.py` reads them FROM `torvik_data` dict.

---

## 8. Helper Functions (Outside torvik.py)

### 8a. `torvik_to_game_records()` (proprietary_metrics.py:2997)

```python
def torvik_to_game_records(
    torvik_teams: List[Dict],   # List of team dicts (from torvik_payload["teams"])
    ...
) -> List[GameRecord]:
    """Convert Torvik team stats into GameRecord objects for training."""
```

Reads from each team dict: `team_id`, `name`, `adj_offensive_efficiency`, `adj_defensive_efficiency`, `adj_tempo`, `conference`, `wins`, `losses`.

### 8b. `enrich_torvik_teams()` (conference_tournament/data_enrichment.py:53)

```python
def enrich_torvik_teams(
    torvik_data: dict,      # {"teams": [...]}
    data_dir: str,          # Directory containing torvik_four_factors_YYYY.json + torvik_shooting_YYYY.json
    year: int,
) -> dict:
    """Merge Four Factors + shooting stats into the main torvik payload.

    The main torvik_YYYY.json often has zeros for Four Factors.
    This function reads torvik_four_factors_YYYY.json and torvik_shooting_YYYY.json
    and fills in the missing values.

    Returns: Enriched copy of torvik_data.
    """
```

### 8c. `cross_validate_torvik_sources()` (ingestion/validators.py:121)

```python
def cross_validate_torvik_sources(
    torvik_teams: List[Dict],
    four_factors: Dict[str, Dict],
) -> List[str]:
    """Compare key metrics between two Torvik-source payloads.
    Returns list of warning strings."""
```

---

## 9. Constants & Configuration

| Constant | Value | Location | Used By |
|----------|-------|----------|---------|
| `CACHE_SCHEMA_VERSION` | `3` | torvik.py:45 | Cache load/save |
| `DEFAULT_CACHE_TTL_SECONDS` | `21600` (6h) | torvik.py:46 | Constructor default |
| `MIN_TEAMS_THRESHOLD` | `100` | torvik.py:49 | Cache validation |
| `_POP_PRIORS` | `{'orb': 0.295, 'drb': 0.705, 'to': 0.185}` | torvik.py:934 | Bayesian shrinkage |
| `_PRIOR_STRENGTH` | `60.0` | torvik.py:942 | Bayesian shrinkage |
| `_TORVIK_FEATURES` | Set of 12 feature names | feature_engineering.py:911 | Era-aware feature gating |
| `_SOURCE_START_YEARS['torvik']` | `2008` | feature_engineering.py:946 | Feature availability |
| `TORVIK_START` | `2008` | extended_historical_ingest.py:34 | Historical collection |

---

## 10. Config Fields

```python
@dataclass
class SOTAPipelineConfig:
    torvik_json: Optional[str] = None    # Path to torvik_YYYY.json (config.py:361)
```

```python
@dataclass
class CollectorConfig:
    scrape_torvik: bool = True                              # collector.py:79
    torvik_splits_url: Optional[str] = None                 # collector.py:64
    torvik_provider_priority: Optional[List[str]] = None    # collector.py:88
```

```python
@dataclass
class HistoricalPipelineConfig:
    include_torvik: bool = True                             # historical_pipeline.py:55
    torvik_provider_priority: Optional[List[str]] = None    # historical_pipeline.py:57
```

---

## 11. Governance Requirements

Production pipeline (`src/governance/`) requires:

1. `production_validator.py:19` — `"torvik_json"` MUST be present in the required artifact set
2. `production_runner.py:35` — `"torvik_json"` in required artifact paths
3. `pipeline_freeze_2026.json` — Contains SHA256 hash of `src/data/scrapers/torvik.py`
4. `pipeline_monitor.py:34` — `"torvik_*.json": 168` hours freshness requirement

---

## 12. Compatibility Summary

### MUST preserve (breaks pipeline if changed):
- `TorVikTeam` dataclass with all 30 fields + `to_dict()`
- `BartTorvikScraper` constructor signature (4 kwargs)
- `fetch_current_rankings(year, strict)` → `List[TorVikTeam]`
- `fetch_four_factors(year)` → `Dict[str, Dict]` with 8-field schema
- `fetch_shooting_stats(year)` → `Dict[str, Dict]` with 2-field schema
- `load_from_json(filepath)` → `List[TorVikTeam]`
- `_dict_to_team(data)` → `TorVikTeam`
- `TorVikValidator` with `validate_team/validate_teams/validate_four_factors`
- `TorVikValidationError` exception
- JSON file schemas (all 3 types)
- `_normalize_team_name_to_id(name)` static method

### CAN change (internal implementation):
- HTTP fetching strategies (cbbstat API, CSV fallback)
- Circuit breaker internals
- Cache TTL logic
- Player CSV aggregation internals
- Bayesian shrinkage parameters
- Logging messages
- `_fetch_strategy` telemetry dict

### CAN remove (optional/isolated):
- `src/data/scrapers/torvik_r.py` — R wrapper, gracefully degraded everywhere
- `scripts/create_missing_torvik.py` — one-off utility
- Internal methods not referenced externally
