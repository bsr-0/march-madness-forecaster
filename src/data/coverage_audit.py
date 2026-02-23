"""
Shared helpers for auditing team-metrics coverage and resolving game IDs
to metric IDs. Keeps resolver logic consistent across pipeline and scripts.
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Dict, Optional, Callable, List, Tuple

from .features.proprietary_metrics import _load_cbbpy_team_map


# Known aliases: CBBpy abbreviated/alternate game IDs → SR metric IDs.
# These cover common cases where prefix matching fails (e.g., "vcu_rams"
# doesn't start with "virginia_commonwealth").
GAME_ID_ALIASES: Dict[str, str] = {
    "uconn": "connecticut",
    "vcu": "virginia_commonwealth",
    "ole_miss": "mississippi",
    "byu": "brigham_young",
    "unlv": "nevada_las_vegas",
    "lsu": "louisiana_state",
    "smu": "southern_methodist",
    "pitt": "pittsburgh",
    "app_state": "appalachian_state",
    "ul_monroe": "louisiana_monroe",
    "ul_lafayette": "louisiana_lafayette",
    "uic": "illinois_chicago",
    "utsa": "texas_san_antonio",
    "utep": "texas_el_paso",
    "ut_martin": "tennessee_martin",
    "sf_austin": "stephen_f__austin",
    "unc": "north_carolina",
    "unc_asheville": "north_carolina_asheville",
    "unc_greensboro": "north_carolina_greensboro",
    "unc_wilmington": "north_carolina_wilmington",
    "unc_pembroke": "north_carolina_pembroke",
    "uab": "alabama_birmingham",
    "ucf": "central_florida",
    "usc": "southern_california",
    "umbc": "maryland_baltimore_county",
    "umkc": "missouri_kansas_city",
    "umass": "massachusetts",
    "umass_lowell": "massachusetts_lowell",
    "umes": "maryland_eastern_shore",
    "uju": "jacksonville",
    "iupui": "indiana_purdue_indianapolis",
    "iu_indianapolis": "indiana_purdue_indianapolis",
    "hawaii": "hawaii",
    "hawai_i": "hawaii",
    "se_louisiana": "southeastern_louisiana",
    "se_missouri_state": "southeast_missouri_state",
    "fiu": "florida_international",
    "fau": "florida_atlantic",
    "tcu": "texas_christian",
    "wku": "western_kentucky",
    "wsu": "wichita_state",
    "miami": "miami__fl",
    "loyola_chicago": "loyola__il",
    "loyola_il": "loyola__il",
    "loyola_md": "loyola__md",
    "loyola_maryland": "loyola__md",
    "saint_mary_s": "saint_mary_s__ca",
    "st__francis_brooklyn": "st__francis__ny",
    "st__francis__ny": "st__francis__ny",
    "ualbany": "albany__ny",
    "ut_rio_grande_valley": "texas_rio_grande_valley",
    "utrgv": "texas_rio_grande_valley",
    "charleston": "college_of_charleston",
    "texas_am": "texas_a_m",
    "a_m": "texas_a_m",
    "col_of_charleston": "college_of_charleston",
    "uc_davis": "california_davis",
    "uc_irvine": "california_irvine",
    "uc_riverside": "california_riverside",
    "uc_santa_barbara": "california_santa_barbara",
    "uc_san_diego": "california_san_diego",
    "san_jose_state": "san_jose_state",
    "san_jos_state": "san_jose_state",
    "saint_francis_red_flash": "saint_francis__pa",
    "saint_francis_pa": "saint_francis__pa",
    "fairleigh_dickinson": "fairleigh_dickinson",
    "uc_santa_cruz": "california_santa_cruz",
    "njit": "new_jersey_institute_of_technology",
    "siena": "siena",
    "csun": "cal_state_northridge",
    "csuf": "cal_state_fullerton",
    "csub": "cal_state_bakersfield",
    "csulb": "long_beach_state",
    "csus": "sacramento_state",
    "csu_northridge": "cal_state_northridge",
    "csu_fullerton": "cal_state_fullerton",
    "csu_bakersfield": "cal_state_bakersfield",
    "lb_state": "long_beach_state",
    "seattle_u": "seattle",
    "queens_university": "queens__nc",
    "centenary__la": "centenary__la",
    "winston_salem": "winston_salem_state",
    "omaha": "nebraska_omaha",
}

# Torvik-specific ID aliases to metric IDs (canonical/SR).
# Torvik uses some short IDs (e.g., "fdu") that don't match SR/cbbpy.
TORVIK_ID_ALIASES: Dict[str, str] = {
    "fdu": "fairleigh_dickinson",
    "iu_indy": "indiana_purdue_indianapolis",
    "kansas_city": "missouri_kansas_city",
    "loyola__il": "loyola__il",
    "loyola__md": "loyola__md",
    "saint_francis__pa": "saint_francis__pa",
    "saint_peter_s": "saint_peter_s",
    "mount_st__mary_s": "mount_st__mary_s",
    "st__john_s__ny": "st__john_s__ny",
    "st__bonaventure": "st__bonaventure",
    "st__thomas": "st__thomas",
    "william___mary": "william___mary",
}

def build_game_id_resolver(
    team_metrics: Dict[str, Dict],
    games_payload: Dict,
) -> Callable[[str], Optional[str]]:
    """
    Build a resolver that maps game team IDs (mascot-suffixed) to
    metric IDs (school canonical).
    """
    metric_keys = sorted(team_metrics.keys(), key=len, reverse=True)
    _prefix_cache: Dict[str, str] = {}

    # display_name → location (e.g., "UConn Huskies" → "UConn")
    _cbbpy_map = _load_cbbpy_team_map()

    def _loc_to_metric(loc_norm: str) -> Optional[str]:
        # Direct hit
        if loc_norm in team_metrics:
            return loc_norm
        # Alias table (handles uconn→connecticut, lsu→louisiana_state, etc.)
        _alias_target = GAME_ID_ALIASES.get(loc_norm)
        if _alias_target and _alias_target in team_metrics:
            return _alias_target
        # Prefix match (safe for location strings)
        for _mk in metric_keys:
            if loc_norm.startswith(_mk + "_") or _mk.startswith(loc_norm + "_"):
                return _mk
        # Unicode normalization (san_josé_state → san_jose_state)
        try:
            import unicodedata as _ud
            _ascii = "".join(
                c for c in _ud.normalize("NFKD", loc_norm)
                if not _ud.combining(c)
            )
            if _ascii != loc_norm:
                return _loc_to_metric(_ascii)
        except Exception:
            pass
        return None

    # Build display-name → metric_id lookup from game team names.
    _display_to_metric: Dict[str, str] = {}
    _games_raw = games_payload.get("games", [])
    for _g in _games_raw:
        for _id_key, _name_key in (("team1_id", "team1_name"), ("team2_id", "team2_name")):
            _gid = _normalize_team_id(str(_g.get(_id_key, "") or ""))
            _gname = str(_g.get(_name_key, "") or "")
            if _gid and _gname and _gid not in _display_to_metric:
                _loc = _cbbpy_map.get(_gname)
                if _loc:
                    _loc_norm = _normalize_team_id(_loc)
                    _resolved = _loc_to_metric(_loc_norm)
                    if _resolved:
                        _display_to_metric[_gid] = _resolved

    def _resolve_team(game_id: str) -> Optional[str]:
        if game_id in team_metrics:
            return game_id
        if game_id in _prefix_cache:
            return _prefix_cache[game_id]
        # Exact alias match should take precedence over prefix heuristics.
        _alias_target = GAME_ID_ALIASES.get(game_id)
        if _alias_target:
            if _alias_target in team_metrics:
                _prefix_cache[game_id] = _alias_target
                return _alias_target
            for mk in metric_keys:
                if mk.startswith(_alias_target) or _alias_target.startswith(mk):
                    _prefix_cache[game_id] = mk
                    return mk
        # Prefix matching (guarded): exact match or mascot suffix.
        for mk in metric_keys:
            if game_id == mk or game_id.startswith(mk + "_"):
                _prefix_cache[game_id] = mk
                return mk
        # If game_id is shorter than metric key, only accept a unique match.
        prefix_candidates = [mk for mk in metric_keys if mk.startswith(game_id + "_")]
        if len(prefix_candidates) == 1:
            _prefix_cache[game_id] = prefix_candidates[0]
            return prefix_candidates[0]
        # Display-name lookup via cbbpy_team_map
        if game_id in _display_to_metric:
            _prefix_cache[game_id] = _display_to_metric[game_id]
            return _display_to_metric[game_id]
        # Known aliases for abbreviated CBBpy names
        for alias_prefix, metric_id in GAME_ID_ALIASES.items():
            if game_id.startswith(alias_prefix + "_"):
                # Reject ambiguous short aliases (e.g., "a_m", "unc", "miami").
                if len(alias_prefix) < 4:
                    continue
                ambiguous = any(
                    mk.startswith(alias_prefix + "_") and mk != metric_id
                    for mk in metric_keys
                )
                if ambiguous:
                    continue
                if metric_id in team_metrics:
                    _prefix_cache[game_id] = metric_id
                    return metric_id
                for mk in metric_keys:
                    if mk.startswith(metric_id) or metric_id.startswith(mk):
                        _prefix_cache[game_id] = mk
                        return mk
        # Unicode normalization fallback
        try:
            import unicodedata as _udata
            _ascii_id = "".join(
                c for c in _udata.normalize("NFKD", game_id)
                if not _udata.combining(c)
            )
            if _ascii_id != game_id:
                _result = _resolve_team(_ascii_id)
                if _result:
                    _prefix_cache[game_id] = _result
                    return _result
        except Exception:
            pass
        return None

    return _resolve_team


def _normalize_team_id(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def _load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _find_years(historical_dir: str) -> List[int]:
    years = set()
    for fname in os.listdir(historical_dir):
        m = re.match(r"historical_games_(\d{4})\.json$", fname)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def audit_year(games_path: str, metrics_path: str, seeds_path: Optional[str]) -> Dict:
    games_payload = _load_json(games_path)
    metrics_payload = _load_json(metrics_path)

    cbbpy_map = _load_cbbpy_team_map()
    d1_cache: Dict[tuple, bool] = {}
    d1_ids = {_normalize_team_id(loc) for loc in cbbpy_map.values()}

    # Build metrics map
    team_metrics: Dict[str, Dict] = {}
    for tm in metrics_payload.get("teams", []):
        tid = _normalize_team_id(str(tm.get("team_id") or tm.get("name", "")))
        if tid:
            team_metrics[tid] = tm

    team_seeds: Dict[str, int] = {}
    if seeds_path and os.path.exists(seeds_path):
        seeds_payload = _load_json(seeds_path)
        for entry in seeds_payload.get("teams", []):
            tid = _normalize_team_id(str(entry.get("team_id", "")))
            seed = int(entry.get("seed", 0))
            if tid and seed:
                team_seeds[tid] = seed

    resolve_team = build_game_id_resolver(team_metrics, games_payload)

    missing_counts: Dict[str, int] = {}
    missing_names: Dict[str, set] = {}
    missing_canon: Dict[str, str] = {}
    missing_vs_seeded: set = set()
    missing_vs_seeded_wins: set = set()
    missing_probable_d1: set = set()
    missing_probable_non_d1: set = set()
    missing_strict_d1: set = set()

    total_games = 0
    resolved_games = 0
    missing_team_entries = 0

    for game in games_payload.get("games", []):
        raw_t1 = _normalize_team_id(str(game.get("team1_id") or game.get("team1") or ""))
        raw_t2 = _normalize_team_id(str(game.get("team2_id") or game.get("team2") or ""))
        if not raw_t1 or not raw_t2:
            continue
        total_games += 1

        t1 = resolve_team(raw_t1)
        t2 = resolve_team(raw_t2)
        t1_ok = bool(t1 and t1 in team_metrics)
        t2_ok = bool(t2 and t2 in team_metrics)
        if t1_ok and t2_ok:
            resolved_games += 1

        # Missing team bookkeeping
        if not t1_ok:
            missing_counts[raw_t1] = missing_counts.get(raw_t1, 0) + 1
            missing_names.setdefault(raw_t1, set()).add(str(game.get("team1_name") or game.get("team1") or raw_t1))
            missing_team_entries += 1
            if raw_t1 not in missing_canon:
                missing_canon[raw_t1] = _canonical_from_display(
                    str(game.get("team1_name") or game.get("team1") or ""),
                    raw_t1,
                    cbbpy_map,
                )
        if not t2_ok:
            missing_counts[raw_t2] = missing_counts.get(raw_t2, 0) + 1
            missing_names.setdefault(raw_t2, set()).add(str(game.get("team2_name") or game.get("team2") or raw_t2))
            missing_team_entries += 1
            if raw_t2 not in missing_canon:
                missing_canon[raw_t2] = _canonical_from_display(
                    str(game.get("team2_name") or game.get("team2") or ""),
                    raw_t2,
                    cbbpy_map,
                )

        # Seeded-opponent adjacency
        s1 = int(game.get("team1_score", 0))
        s2 = int(game.get("team2_score", 0))
        if not t1_ok and t2_ok and t2 in team_seeds:
            missing_vs_seeded.add(raw_t1)
            if s1 > s2:
                missing_vs_seeded_wins.add(raw_t1)
        if not t2_ok and t1_ok and t1 in team_seeds:
            missing_vs_seeded.add(raw_t2)
            if s2 > s1:
                missing_vs_seeded_wins.add(raw_t2)

    raw_dir = os.path.dirname(os.path.dirname(games_path))
    torvik_metric_ids = _load_torvik_metric_ids_for_year(
        raw_dir,
        os.path.join(raw_dir, "cache"),
        year=_infer_year_from_path(games_path),
    )

    for raw_id, names in missing_names.items():
        name = sorted(list(names))[0] if names else ""
        if _is_probable_d1(name, raw_id, cbbpy_map, d1_ids, d1_cache):
            missing_probable_d1.add(raw_id)
        else:
            missing_probable_non_d1.add(raw_id)
        canon = missing_canon.get(raw_id, _normalize_team_id(raw_id))
        if torvik_metric_ids and canon in torvik_metric_ids:
            missing_strict_d1.add(raw_id)

    top_missing = sorted(missing_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top_missing_fmt = [
        {
            "team_id": tid,
            "count": cnt,
            "names": sorted(list(missing_names.get(tid, set())))[:3],
        }
        for tid, cnt in top_missing
    ]

    return {
        "total_games": total_games,
        "resolved_games": resolved_games,
        "missing_team_entries": missing_team_entries,
        "missing_unique_teams": len(missing_counts),
        "missing_probable_d1": len(missing_probable_d1),
        "missing_probable_non_d1": len(missing_probable_non_d1),
        "missing_probable_d1_list": sorted(list(missing_probable_d1)),
        "missing_strict_d1": len(missing_strict_d1),
        "missing_strict_d1_list": sorted(list(missing_strict_d1)),
        "missing_vs_seeded": len(missing_vs_seeded),
        "missing_vs_seeded_wins": len(missing_vs_seeded_wins),
        "top_missing": top_missing_fmt,
    }


def run_coverage_audit(
    historical_dir: str,
    out_json: str,
    out_csv: str,
) -> Tuple[str, str]:
    years = _find_years(historical_dir)
    if not years:
        raise ValueError(f"No historical_games_YYYY.json files found in {historical_dir}")

    results: Dict[str, Dict] = {}
    summary = {
        "years": len(years),
        "total_games": 0,
        "resolved_games": 0,
        "missing_team_entries": 0,
        "missing_unique_teams": 0,
        "missing_probable_d1": 0,
        "missing_probable_non_d1": 0,
        "missing_strict_d1": 0,
        "missing_vs_seeded": 0,
        "missing_vs_seeded_wins": 0,
    }
    missing_probable_d1_full: set = set()
    missing_strict_d1_full: set = set()

    for year in years:
        games_path = os.path.join(historical_dir, f"historical_games_{year}.json")
        metrics_path = os.path.join(historical_dir, f"team_metrics_{year}.json")
        seeds_path = os.path.join(historical_dir, f"tournament_seeds_{year}.json")

        if not os.path.exists(games_path) or not os.path.exists(metrics_path):
            continue

        year_result = audit_year(games_path, metrics_path, seeds_path)
        results[str(year)] = year_result

        summary["total_games"] += year_result["total_games"]
        summary["resolved_games"] += year_result["resolved_games"]
        summary["missing_team_entries"] += year_result["missing_team_entries"]
        summary["missing_unique_teams"] += year_result["missing_unique_teams"]
        summary["missing_probable_d1"] += year_result["missing_probable_d1"]
        summary["missing_probable_non_d1"] += year_result["missing_probable_non_d1"]
        summary["missing_strict_d1"] += year_result["missing_strict_d1"]
        summary["missing_vs_seeded"] += year_result["missing_vs_seeded"]
        summary["missing_vs_seeded_wins"] += year_result["missing_vs_seeded_wins"]
        missing_probable_d1_full.update(year_result.get("missing_probable_d1_list", []))
        missing_strict_d1_full.update(year_result.get("missing_strict_d1_list", []))

    payload = {
        "summary": summary,
        "missing_probable_d1_full": sorted(list(missing_probable_d1_full)),
        "missing_strict_d1_full": sorted(list(missing_strict_d1_full)),
        "years": results,
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "year",
            "total_games",
            "resolved_games",
            "missing_team_entries",
            "missing_unique_teams",
            "missing_probable_d1",
            "missing_probable_non_d1",
            "missing_strict_d1",
            "missing_vs_seeded",
            "missing_vs_seeded_wins",
        ])
        for year in years:
            yr = results.get(str(year))
            if not yr:
                continue
            writer.writerow([
                year,
                yr["total_games"],
                yr["resolved_games"],
                yr["missing_team_entries"],
                yr["missing_unique_teams"],
                yr["missing_probable_d1"],
                yr["missing_probable_non_d1"],
                yr["missing_strict_d1"],
                yr["missing_vs_seeded"],
                yr["missing_vs_seeded_wins"],
            ])

    return out_json, out_csv


def _is_probable_d1(
    display_name: str,
    raw_id: str,
    cbbpy_map: Dict[str, str],
    d1_ids: set,
    cache: Dict[tuple, bool],
) -> bool:
    """
    Heuristic D1 check:
    - If display name exists in cbbpy_team_map.csv → treat as D1
    - If TeamNameResolver can match with high confidence → treat as D1
    """
    key = (display_name or "", raw_id or "")
    if key in cache:
        return cache[key]
    if display_name and display_name in cbbpy_map:
        cache[key] = True
        return True
    if raw_id and raw_id in d1_ids:
        cache[key] = True
        return True
    cache[key] = False
    return False


def _infer_year_from_path(games_path: str) -> Optional[int]:
    m = re.search(r"historical_games_(\d{4})\.json", games_path)
    if not m:
        return None
    return int(m.group(1))


def _canonical_from_display(display_name: str, raw_id: str, cbbpy_map: Dict[str, str]) -> str:
    """
    Try to get a canonical (school-only) ID for a team. Prefer cbbpy map
    (display name → location), otherwise fall back to raw_id normalization.
    """
    if display_name and display_name in cbbpy_map:
        return _normalize_team_id(cbbpy_map[display_name])
    return _normalize_team_id(raw_id)


def _load_torvik_metric_ids_for_year(*base_dirs: str, year: Optional[int]) -> set:
    """
    Strict D1 list: Torvik team IDs mapped to metric IDs using alias table.
    Looks for torvik_four_factors_{year}.json or torvik_shooting_{year}.json
    in provided base directories.
    """
    if not year:
        return set()
    filenames = [
        f"torvik_four_factors_{year}.json",
        f"torvik_shooting_{year}.json",
    ]
    torvik_ids = set()
    for base_dir in base_dirs:
        if not base_dir or not os.path.isdir(base_dir):
            continue
        for fname in filenames:
            path = os.path.join(base_dir, fname)
            if not os.path.exists(path):
                continue
            try:
                payload = _load_json(path)
                if isinstance(payload, dict):
                    torvik_ids.update([_normalize_team_id(k) for k in payload.keys()])
            except Exception:
                continue
    if not torvik_ids:
        return set()

    # Map Torvik IDs to metric IDs where possible
    metric_ids = set()
    for tid in torvik_ids:
        if tid in TORVIK_ID_ALIASES:
            metric_ids.add(TORVIK_ID_ALIASES[tid])
            continue
        if tid in GAME_ID_ALIASES:
            metric_ids.add(GAME_ID_ALIASES[tid])
            continue
        metric_ids.add(tid)
    return metric_ids
