import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
HIST_ROOT = DATA_ROOT / "historical"
ARTIFACT_ROOT = REPO_ROOT / "artifacts"
BRACKET_PATH = DATA_ROOT / "bracket_2026.json"
OUTPUT_PATH = REPO_ROOT / "docs" / "data" / "dashboard.json"

FEATURE_FILE = REPO_ROOT / "src" / "pipeline" / "sota.py"

DEFAULT_HOLDOUT_YEAR = 2025
PREFERRED_HOLDOUT_YEAR = 2025

GROUP_WEIGHTS = {
    "Core efficiency": 0.22,
    "Four Factors": 0.18,
    "Defensive Four Factors": 0.08,
    "Schedule strength": 0.08,
    "Elo": 0.1,
    "Shooting stability": 0.05,
    "Win percent": 0.06,
    "3pt shooting": 0.07,
    "Experience": 0.05,
    "Absolute level": 0.07,
    "Interaction": 0.04,
}

FEATURE_GROUPS = {
    "diff_adj_off_eff": "Core efficiency",
    "diff_adj_def_eff": "Core efficiency",
    "diff_adj_tempo": "Core efficiency",
    "diff_efg_pct": "Four Factors",
    "diff_to_rate": "Four Factors",
    "diff_orb_rate": "Four Factors",
    "diff_ft_rate": "Four Factors",
    "diff_opp_efg_pct": "Defensive Four Factors",
    "diff_opp_to_rate": "Defensive Four Factors",
    "diff_sos_adj_em": "Schedule strength",
    "diff_elo_rating": "Elo",
    "diff_free_throw_pct": "Shooting stability",
    "diff_win_pct": "Win percent",
    "diff_three_pt_pct": "3pt shooting",
    "diff_three_pt_variance": "3pt shooting",
    "diff_avg_experience": "Experience",
    "diff_roster_continuity": "Experience",
    "abs_adj_off_eff": "Absolute level",
    "abs_adj_def_eff": "Absolute level",
    "abs_sos_adj_em": "Absolute level",
    "seed_interaction": "Interaction",
    "three_pt_seed_interaction": "Interaction",
}

PAIRINGS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def _holdout_candidates() -> List[Path]:
    return sorted(DATA_ROOT.glob("prospective_eval_*_*.json"))


def _extract_holdout_year(payload: Dict) -> int:
    holdout = payload.get("holdout_evaluation", {})
    years = holdout.get("holdout_years") or []
    if years:
        try:
            return int(years[0])
        except Exception:
            pass
    return DEFAULT_HOLDOUT_YEAR


def _select_best_holdout_file(candidates: List[Path]) -> Path:
    parsed: List[Tuple[int, float, Path]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
            year = _extract_holdout_year(payload)
            parsed.append((year, path.stat().st_mtime, path))
        except Exception:
            continue
    if not parsed:
        raise ValueError("No readable holdout evaluation files")

    preferred = [p for p in parsed if p[0] == PREFERRED_HOLDOUT_YEAR]
    if preferred:
        preferred.sort(key=lambda x: x[1], reverse=True)
        return preferred[0][2]

    parsed.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return parsed[0][2]


def _load_holdout_evaluation() -> Tuple[int, Dict[str, float], str, Dict]:
    candidates = _holdout_candidates()
    if not candidates:
        return (
            DEFAULT_HOLDOUT_YEAR,
            {},
            "pending",
            {
                "level": "unknown",
                "note": "No holdout artifact found.",
                "source": None,
            },
        )

    latest = _select_best_holdout_file(candidates)
    payload = json.loads(latest.read_text())
    holdout = payload.get("holdout_evaluation", {})
    holdout_year = _extract_holdout_year(payload)
    per_year = holdout.get("per_year", {}).get(str(holdout_year), {})
    metrics = {
        "games": per_year.get("n_games"),
        "brier": per_year.get("brier_score"),
        "round_weighted_brier": per_year.get("round_weighted_brier"),
        "log_loss": per_year.get("log_loss"),
        "accuracy": per_year.get("accuracy"),
        "brier_skill": per_year.get("brier_skill_score"),
        "estimated_kaggle_rank": per_year.get("estimated_kaggle_rank"),
    }
    status = holdout.get("verdict", "ok")
    integrity = {
        "level": holdout.get("integrity_level", payload.get("integrity_level", "unknown")),
        "note": holdout.get(
            "integrity_note",
            payload.get(
                "integrity_note",
                "Retrospective holdout diagnostics, not true prospective pre-registration.",
            ),
        ),
        "source": latest.name,
    }
    return holdout_year, metrics, status, integrity


def _load_unified_backtest(target_year: int) -> Dict:
    patterns = [
        "*unified_backtest*.json",
        "*backtest_unified*.json",
        "*unified*backtest*.json",
    ]
    candidates: List[Path] = []
    for root in (ARTIFACT_ROOT, DATA_ROOT, REPO_ROOT):
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.glob(pattern))
    if not candidates:
        return {}

    parsed: List[Tuple[float, Dict, Path]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload.get("results"), list):
                parsed.append((path.stat().st_mtime, payload, path))
        except Exception:
            continue
    if not parsed:
        return {}

    with_target_year = [
        item for item in parsed if any(r.get("year") == target_year for r in item[1].get("results", []))
    ]
    if with_target_year:
        with_target_year.sort(key=lambda x: x[0], reverse=True)
        payload, source = with_target_year[0][1], with_target_year[0][2]
    else:
        parsed.sort(key=lambda x: x[0], reverse=True)
        payload, source = parsed[0][1], parsed[0][2]

    year_results = [r for r in payload.get("results", []) if r.get("year") == target_year]
    if not year_results:
        return {}

    cal = next((r for r in year_results if r.get("mode") == "calibration"), {})
    ev_candidates = [r for r in year_results if r.get("mode") == "ev"]
    ev = next((r for r in ev_candidates if r.get("pool_size") == 30), None)
    if ev is None and ev_candidates:
        ev = sorted(ev_candidates, key=lambda r: r.get("pool_size", 10**9))[0]

    return {
        "source": source.name,
        "kaggle": {
            "estimated_rank": cal.get("kaggle_rank"),
            "round_weighted_brier": cal.get("rw_brier"),
            "brier": cal.get("brier"),
            "accuracy": cal.get("accuracy"),
        }
        if cal
        else {},
        "espn_pool": {
            "pool_size": ev.get("pool_size"),
            "rank_position": ev.get("pool_rank"),
            "score": ev.get("pool_score"),
        }
        if ev
        else {},
    }


def _parse_fixed_feature_set() -> List[str]:
    text = FEATURE_FILE.read_text()
    start = text.find("FIXED_FEATURE_SET")
    if start == -1:
        return []
    block = text[start:]
    start_bracket = block.find("[")
    end_bracket = block.find("]")
    if start_bracket == -1 or end_bracket == -1:
        return []
    raw = block[start_bracket : end_bracket + 1]
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    try:
        features = json.loads(cleaned.replace("'", '"'))
        if isinstance(features, list):
            return [f for f in features if isinstance(f, str)]
    except Exception:
        pass
    return []


def _build_feature_importance(features: List[str]) -> List[Dict[str, float]]:
    grouped: Dict[str, List[str]] = {}
    for feat in features:
        group = FEATURE_GROUPS.get(feat, "Other")
        grouped.setdefault(group, []).append(feat)

    importance: List[Dict[str, float]] = []
    for group, feats in grouped.items():
        weight = GROUP_WEIGHTS.get(group, 0.02)
        per = weight / max(len(feats), 1)
        for feat in feats:
            importance.append({"name": feat, "importance": per, "group": group})

    total = sum(item["importance"] for item in importance) or 1
    for item in importance:
        item["importance"] = item["importance"] / total
    return importance


def _summarize_training(holdout_year: int) -> Dict:
    games_per_season = {}
    teams = set()
    total_games = 0
    seasons = []
    min_date = None
    max_date = None

    for path in sorted(HIST_ROOT.glob("historical_games_*.json")):
        season = int(path.stem.split("_")[-1])
        if season >= holdout_year:
            continue
        payload = json.loads(path.read_text())
        games = payload.get("games", [])
        games_per_season[str(season)] = len(games)
        total_games += len(games)
        seasons.append(season)
        for game in games:
            teams.add(game.get("team1_name"))
            teams.add(game.get("team2_name"))
            date = game.get("date")
            if date:
                if not min_date or date < min_date:
                    min_date = date
                if not max_date or date > max_date:
                    max_date = date

    seasons = sorted(seasons)
    feature_names = _parse_fixed_feature_set()
    return {
        "seasons": seasons,
        "total_games": total_games,
        "unique_teams": len([t for t in teams if t]),
        "feature_count": len(feature_names),
        "games_per_season": games_per_season,
        "sources": ["cbbpy", "sportsipy", "barttorvik"],
        "notes": (
            f"Training window spans {min_date} to {max_date}. Holdout season {holdout_year} excluded from training."
        ),
    }


def _seed_win_prob(seed1: int, seed2: int, slope: float = 0.175) -> float:
    diff = seed2 - seed1
    return 1.0 / (1.0 + math.exp(-slope * diff))


def _build_predictions() -> Dict:
    if not BRACKET_PATH.exists():
        return {"matchups": [], "champion_probs": [], "notes": "Bracket file missing."}
    bracket = json.loads(BRACKET_PATH.read_text())
    teams = bracket.get("teams", [])
    by_region: Dict[str, Dict[int, Dict]] = {}
    for team in teams:
        region = team.get("region")
        seed = int(team.get("seed", 0))
        if not region or not seed:
            continue
        by_region.setdefault(region, {})[seed] = team

    matchups = []
    for region, seeds in by_region.items():
        for seed1, seed2 in PAIRINGS:
            t1 = seeds.get(seed1)
            t2 = seeds.get(seed2)
            if not t1 or not t2:
                continue
            win_prob = _seed_win_prob(seed1, seed2)
            matchups.append(
                {
                    "round": "R64",
                    "region": region,
                    "team1": {"name": t1.get("team_name") or t1.get("name"), "seed": seed1},
                    "team2": {"name": t2.get("team_name") or t2.get("name"), "seed": seed2},
                    "team1_win_prob": win_prob,
                    "predicted_winner": (t1.get("team_name") or t1.get("name"))
                    if win_prob >= 0.5
                    else (t2.get("team_name") or t2.get("name")),
                }
            )

    champion_scores = []
    for team in teams:
        seed = int(team.get("seed", 0)) or 16
        strength = math.exp(-0.15 * seed)
        champion_scores.append((team.get("team_name") or team.get("name"), strength))

    total = sum(score for _, score in champion_scores) or 1
    champion_probs = [
        {"team": name, "prob": score / total}
        for name, score in sorted(champion_scores, key=lambda x: x[1], reverse=True)
    ]

    return {
        "season": bracket.get("season"),
        "method": "seed_prior_baseline",
        "matchups": matchups,
        "champion_probs": champion_probs,
        "notes": "Seed-based baseline probabilities derived from bracket seeds.",
    }


def main() -> None:
    holdout_year, backtest_metrics, backtest_status, integrity = _load_holdout_evaluation()
    unified = _load_unified_backtest(holdout_year)
    training_summary = _summarize_training(holdout_year)
    feature_names = _parse_fixed_feature_set()
    feature_importance = _build_feature_importance(feature_names)
    predictions = _build_predictions()

    dashboard = {
        "metadata": {
            "model_name": "SOTA Ensemble (dashboard export)",
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "training_seasons": training_summary.get("seasons", []),
            "holdout_season": holdout_year,
            "notes": "Exported for GitHub Pages dashboard.",
        },
        "training_summary": training_summary,
        "feature_importance": {
            "method": "domain_prior",
            "features": feature_importance,
            "notes": "Weights reflect pre-registered feature priors unless model gain importances are supplied.",
        },
        "predictions": predictions,
        "backtest": {
            "season": holdout_year,
            "status": backtest_status,
            "metrics": backtest_metrics,
            "kaggle": unified.get("kaggle", {}),
            "espn_pool": unified.get("espn_pool", {}),
            "integrity": integrity,
            "notes": (
                "Holdout evaluation summary (latest available, prefers 2025 when present). "
                "Historical holdout should be interpreted as retrospective unless explicitly pre-registered."
            ),
            "sources": {
                "holdout_eval": integrity.get("source"),
                "unified_backtest": unified.get("source"),
            },
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dashboard, indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
