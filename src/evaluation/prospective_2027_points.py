"""Frozen walk-forward selector and source gate for the 2027 ESPN-points release."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "frozen" / "prospective_2027_v4_points.json"
RESULT_PATH = REPO / "artifacts" / "prospective_2027" / "selector_result.json"
RULE_IDS = ("blend_region_35", "ev_optimal", "highest_p1")


def load_spec(path: Path = SPEC_PATH) -> Dict[str, Any]:
    spec = json.loads(path.read_text())
    from src.governance.frozen_spec import canonical_hash

    body = {key: value for key, value in spec.items() if key != "spec_hash"}
    if spec.get("spec_hash") != canonical_hash(body):
        raise RuntimeError(f"{path}: frozen selector spec hash is invalid")
    if set(spec.get("selector", {}).get("candidate_rules", {})) != set(RULE_IDS):
        raise RuntimeError(f"{path}: frozen candidate rules differ from the v4 implementation")
    return spec


def candidate_bracket(artifact: Mapping[str, Any], rule_id: str) -> list[list[int]]:
    """Return a frozen rule's bracket as artifact team indices."""
    from src.product.selection import select_diverse

    if rule_id in ("blend_region_35", "ev_optimal"):
        try:
            bracket = artifact["named_strategies"][rule_id]["w"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"candidate artifact has no required rule {rule_id!r}") from exc
    elif rule_id == "highest_p1":
        try:
            index = select_diverse(dict(artifact), objective="p1", k=1)[0]
            bracket = artifact["candidates"][index]["w"]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("candidate artifact cannot supply the highest_p1 rule") from exc
    else:
        raise ValueError(f"unknown frozen candidate rule {rule_id!r}")
    return _validate_index_bracket(bracket, len(artifact.get("teams", [])))


def _validate_index_bracket(bracket: Any, team_count: int) -> list[list[int]]:
    if not isinstance(bracket, list) or len(bracket) != 6:
        raise ValueError("bracket must contain six rounds")
    expected_lengths = (32, 16, 8, 4, 2, 1)
    result: list[list[int]] = []
    for round_index, (round_teams, expected_length) in enumerate(zip(bracket, expected_lengths)):
        if not isinstance(round_teams, list) or len(round_teams) != expected_length:
            raise ValueError(f"bracket round {round_index} must contain {expected_length} team indices")
        if any(type(team_index) is not int or not 0 <= team_index < team_count for team_index in round_teams):
            raise ValueError(f"bracket round {round_index} contains an invalid team index")
        result.append(list(round_teams))
    return result


def build_seed_only_bracket(artifact: Mapping[str, Any], year: int) -> list[list[int]]:
    """Construct the deterministic walk-forward seed baseline on an artifact's field."""
    from scripts.mc_pool_backtest import ESPN_SCORING, ROUND_NAMES
    from src.optimization.bracket_construction import construct_bracket
    from src.prediction.seed_probabilities import build_seed_round_probabilities
    from src.simulation import bracket_topology as topology

    teams = artifact.get("teams")
    first_round_indices = artifact.get("first_round")
    if not isinstance(teams, list) or len(teams) < 64:
        raise ValueError(f"{year}: seed baseline requires at least 64 artifact teams")
    if not isinstance(first_round_indices, list) or len(first_round_indices) != 64:
        raise ValueError(f"{year}: artifact first_round must contain 64 team indices")
    if any(not isinstance(team, Mapping) for team in teams):
        raise ValueError(f"{year}: every artifact team must be an object")
    ids = [team.get("id") for team in teams]
    if any(not isinstance(team_id, str) or not team_id for team_id in ids) or len(set(ids)) != len(teams):
        raise ValueError(f"{year}: artifact team IDs must be unique non-empty strings")
    if any(type(index) is not int or not 0 <= index < len(teams) for index in first_round_indices):
        raise ValueError(f"{year}: artifact first_round has an invalid team index")
    if len(set(first_round_indices)) != 64:
        raise ValueError(f"{year}: artifact first_round must resolve to 64 distinct teams")

    draw_teams = [teams[index] for index in first_round_indices]
    seeds = {team["id"]: team.get("seed") for team in draw_teams}
    regions = {team["id"]: team.get("region") for team in draw_teams}
    if any(type(seed) is not int or not 1 <= seed <= 16 for seed in seeds.values()):
        raise ValueError(f"{year}: artifact team seeds must be integers from 1 through 16")
    if any(not isinstance(region, str) or not region for region in regions.values()):
        raise ValueError(f"{year}: artifact teams must carry regions")

    first_round = [ids[index] for index in first_round_indices]
    region_order = topology.region_order_from_first_round(first_round, regions)
    round_probabilities = build_seed_round_probabilities(seeds, as_of=year)
    picks, *_ = construct_bracket(
        mode="forward_greedy",
        seeds=seeds,
        regions=regions,
        round_probs=round_probabilities,
        public_picks={},
        risk_level=0.0,
        pool_size=30,
        scoring_system=ESPN_SCORING,
        region_order=region_order,
    )
    winners = topology.picks_to_winners_by_round(picks, first_round)
    team_index = {team_id: index for index, team_id in enumerate(ids)}
    indexed = [[team_index[team_id] for team_id in rnd] for rnd in winners]
    if tuple(map(len, indexed)) != (32, 16, 8, 4, 2, 1):
        raise RuntimeError(f"{year}: seed baseline construction returned invalid round lengths")
    if tuple(ROUND_NAMES) != ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
        raise RuntimeError("ESPN scoring round order no longer matches the frozen selector")
    return indexed


def _validate_score_rows(
    points_by_year: Mapping[int, Mapping[str, Any]], target_years: Sequence[int]
) -> Dict[int, Dict[str, float]]:
    if set(points_by_year) != set(target_years):
        missing = sorted(set(target_years) - set(points_by_year))
        extra = sorted(set(points_by_year) - set(target_years))
        raise ValueError(f"paired score years must match the frozen target set; missing={missing}, extra={extra}")
    validated: Dict[int, Dict[str, float]] = {}
    expected_keys = {"seed_only", *RULE_IDS}
    for year in target_years:
        row = points_by_year[year]
        if set(row) != expected_keys:
            raise ValueError(f"{year}: expected score columns {sorted(expected_keys)}, got {sorted(row)}")
        checked: Dict[str, float] = {}
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{year}: {key} score must be a finite number")
            checked[key] = float(value)
        validated[year] = checked
    return validated


def _pick_rule(
    target_year: int,
    gains_by_year: Mapping[int, Mapping[str, float]],
    target_years: Sequence[int],
) -> str | None:
    prior_years = [year for year in target_years if year < target_year]
    if not prior_years:
        return None
    means = {
        rule_id: float(np.mean([gains_by_year[year][rule_id] for year in prior_years]))
        for rule_id in RULE_IDS
    }
    return min(RULE_IDS, key=lambda rule_id: (-means[rule_id], rule_id))


def _confidence_interval(gains: np.ndarray, *, resamples: int, seed: int) -> Dict[str, float]:
    if gains.ndim != 1 or gains.size < 2 or not np.isfinite(gains).all():
        raise ValueError("confidence intervals require at least two finite paired season gains")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, gains.size, size=(resamples, gains.size))
    bootstrap_means = gains[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    mean = float(gains.mean())
    t_critical = float(student_t.ppf(0.975, gains.size - 1))
    t_margin = t_critical * float(gains.std(ddof=1)) / math.sqrt(gains.size)
    return {
        "mean_gain": mean,
        "bootstrap_95_lower": float(low),
        "bootstrap_95_upper": float(high),
        "paired_t_95_lower": mean - t_margin,
        "paired_t_95_upper": mean + t_margin,
        "resamples": resamples,
        "seed": seed,
    }


def evaluate_nested_scores(
    points_by_year: Mapping[int, Mapping[str, Any]], spec: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    """Apply frozen inner selection, outer walk-forward scoring, and promotion gate."""
    spec = dict(spec or load_spec())
    targets = tuple(spec["release"]["historical_gate"]["target_seasons"])
    scores = _validate_score_rows(points_by_year, targets)
    gains = {
        year: {rule_id: scores[year][rule_id] - scores[year]["seed_only"] for rule_id in RULE_IDS}
        for year in targets
    }

    folds: Dict[str, Dict[str, Any]] = {}
    outer_gains = []
    for year in targets:
        selected = _pick_rule(year, gains, targets)
        if selected is None:
            fold_gain = 0.0
            selected_points = scores[year]["seed_only"]
        else:
            fold_gain = gains[year][selected]
            selected_points = scores[year][selected]
        outer_gains.append(fold_gain)
        folds[str(year)] = {
            "selected_rule": selected or "seed_only",
            "seed_only_points": scores[year]["seed_only"],
            "selected_points": selected_points,
            "paired_gain": fold_gain,
            "training_seasons": [prior for prior in targets if prior < year],
        }

    interval_cfg = spec["release"]["historical_gate"]["bootstrap"]
    interval = _confidence_interval(
        np.asarray(outer_gains, dtype=float),
        resamples=int(interval_cfg["resamples"]),
        seed=int(interval_cfg["seed"]),
    )
    bootstrap_pass = interval["mean_gain"] > 0 and interval["bootstrap_95_lower"] > 0
    t_pass = interval["mean_gain"] > 0 and interval["paired_t_95_lower"] > 0
    if bootstrap_pass != t_pass:
        gate = "INDETERMINATE"
    elif bootstrap_pass:
        gate = "PASS"
    else:
        gate = "FAIL"

    production_means = {
        rule_id: float(np.mean([gains[year][rule_id] for year in targets]))
        for rule_id in RULE_IDS
    }
    production_rule = min(RULE_IDS, key=lambda rule_id: (-production_means[rule_id], rule_id))
    release_rule = production_rule if gate == "PASS" else "seed_only"
    return {
        "nested_folds": folds,
        "paired_gains": outer_gains,
        "promotion_gate": {"status": gate, **interval},
        "production_rule_means": production_means,
        "production_rule": production_rule,
        "release": {
            "selected_rule": release_rule,
            "fallback": release_rule == "seed_only",
            "reason": None if release_rule != "seed_only" else f"promotion gate is {gate}",
        },
    }


def _source_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def _collect_source_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("file"), str) and isinstance(value.get("sha256"), str):
            yield value
        for child in value.values():
            yield from _collect_source_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _collect_source_records(child)


def _source_record_issues(record: Any, label: str) -> list[Dict[str, str]]:
    if not isinstance(record, Mapping):
        return [{"status": "INDETERMINATE", "reason": f"{label} source record is absent"}]
    raw_path = record.get("file")
    raw_digest = record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return [{"status": "INDETERMINATE", "reason": f"{label} source path is absent"}]
    if not isinstance(raw_digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", raw_digest) is None:
        return [{"status": "INDETERMINATE", "reason": f"{label} source SHA-256 is absent or invalid"}]
    path = _source_path(raw_path)
    if not path.is_file():
        return [{"status": "INDETERMINATE", "reason": f"{label} source is missing: {raw_path}"}]
    try:
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return [{"status": "INDETERMINATE", "reason": f"{label} source cannot be read: {raw_path}"}]
    if actual_digest != raw_digest.lower():
        return [{"status": "FAIL", "reason": f"{label} source hash differs: {raw_path}"}]
    return []


def _source_group_issues(
    records: Any,
    label: str,
    *,
    minimum: int = 1,
    exact: int | None = None,
) -> list[Dict[str, str]]:
    if not isinstance(records, list) or len(records) < minimum or (exact is not None and len(records) != exact):
        expected = f"exactly {exact}" if exact is not None else f"at least {minimum}"
        return [{"status": "INDETERMINATE", "reason": f"{label} must contain {expected} source records"}]
    issues: list[Dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        issues.extend(_source_record_issues(record, f"{label} input {index}"))
    return issues


def _rating_source_issues(ratings: Any, optional_sources: Sequence[str]) -> list[Dict[str, str]]:
    if not isinstance(ratings, Mapping):
        return [{"status": "INDETERMINATE", "reason": "rating-source provenance is absent"}]
    used = ratings.get("used")
    inputs = ratings.get("inputs")
    if (
        not isinstance(used, list)
        or any(not isinstance(source, str) for source in used)
        or not isinstance(inputs, Mapping)
    ):
        return [{"status": "INDETERMINATE", "reason": "rating-source usage or input records are incomplete"}]
    if "torvik" not in used:
        return [{"status": "FAIL", "reason": "mandatory Torvik rating source is absent"}]
    unknown_sources = set(used) - {"torvik", *optional_sources}
    if unknown_sources or len(used) != len(set(used)):
        return [{"status": "FAIL", "reason": "candidate artifact uses an unregistered or duplicate rating source"}]

    issues: list[Dict[str, str]] = []
    for source in ("torvik", *optional_sources):
        record = inputs.get(source)
        if source in used:
            issues.extend(_source_record_issues(record, f"{source} rating"))
        elif (
            not isinstance(record, Mapping)
            or not isinstance(record.get("file"), str)
            or not record["file"].strip()
            or record.get("status") != "unavailable"
            or record.get("sha256") is not None
        ):
            if isinstance(record, Mapping) and isinstance(record.get("sha256"), str):
                issues.extend(_source_record_issues(record, f"{source} rating"))
            else:
                issues.append({
                    "status": "INDETERMINATE",
                    "reason": f"{source} rating-source availability is not recorded",
                })
    return issues


def inspect_historical_sources(spec: Mapping[str, Any], candidates_dir: Path) -> Dict[str, Any]:
    """Verify availability, stored hashes, and public-pick timestamps for all target seasons."""
    from src.data.season_calendar import TOURNAMENT_START_DATES, get_round_of_64_tip

    target_years = spec["release"]["historical_gate"]["target_seasons"]
    issues: list[Dict[str, str]] = []
    indeterminate = False
    artifacts: Dict[str, Dict[str, Any]] = {}

    for year in target_years:
        path = candidates_dir / f"candidates_{year}.json"
        sidecar = path.with_suffix(".sha256")
        if not path.is_file() or not sidecar.is_file():
            indeterminate = True
            issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "candidate artifact or SHA-256 sidecar is missing"})
            continue
        artifact_bytes = path.read_bytes()
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        expected_hash = sidecar.read_text().split()[0] if sidecar.read_text().split() else ""
        if expected_hash != artifact_hash:
            issues.append({"year": str(year), "status": "FAIL", "reason": "candidate artifact SHA-256 disagrees with its sidecar"})
            continue
        try:
            artifact = json.loads(artifact_bytes)
        except json.JSONDecodeError:
            issues.append({"year": str(year), "status": "FAIL", "reason": "candidate artifact is not valid JSON"})
            continue
        if not isinstance(artifact, Mapping):
            issues.append({"year": str(year), "status": "FAIL", "reason": "candidate artifact root must be an object"})
            continue
        metadata = artifact.get("meta")
        pool_settings = metadata.get("pool_settings") if isinstance(metadata, Mapping) else None
        if pool_settings != {"pool_size": 30, "scoring_id": "espn_standard"}:
            issues.append({"year": str(year), "status": "FAIL", "reason": "candidate artifact does not use the frozen pool/scoring settings"})
            continue

        provenance = artifact.get("provenance")
        if not isinstance(provenance, Mapping):
            indeterminate = True
            issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "artifact has no source provenance"})
            continue
        field_provenance = provenance.get("field")
        seed_provenance = provenance.get("seed_head_to_head")
        seed_inputs = seed_provenance.get("input_files") if isinstance(seed_provenance, Mapping) else None
        public = provenance.get("public_picks")
        torvik = provenance.get("torvik")
        ratings = provenance.get("rating_sources")
        rating_config = spec["candidate_generation"]["rating_sources"]
        evidence_issues = [
            *_source_group_issues(
                field_provenance.get("input_files") if isinstance(field_provenance, Mapping) else None,
                "field/results",
                minimum=2,
            ),
            *_source_group_issues(seed_inputs, "seed-referee", exact=2),
            *_source_record_issues(public, "public-picks"),
            *_source_record_issues(torvik, "Torvik"),
            *_rating_source_issues(ratings, rating_config["optional"]),
        ]
        for evidence_issue in evidence_issues:
            issues.append({"year": str(year), **evidence_issue})
            indeterminate |= evidence_issue["status"] == "INDETERMINATE"
        if isinstance(seed_provenance, Mapping) and seed_provenance.get("point_in_time") is False:
            issues.append({"year": str(year), "status": "FAIL", "reason": "seed referee is explicitly marked as not point-in-time"})
        if (
            not isinstance(seed_provenance, Mapping)
            or seed_provenance.get("point_in_time") is not True
            or type(seed_provenance.get("as_of")) is not int
        ):
            indeterminate = True
            issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "seed-referee point-in-time or as_of provenance is incomplete"})
        elif seed_provenance["as_of"] != year:
            issues.append({"year": str(year), "status": "FAIL", "reason": "seed-referee as_of does not equal the target season"})

        if not isinstance(public, Mapping) or public.get("capture_time_verified") is not True:
            indeterminate = True
            issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "public-picks capture time is not verified"})
        else:
            raw_timestamp = None
            public_file = public.get("file")
            if not isinstance(public_file, str) or not _source_path(public_file).is_file():
                indeterminate = True
                issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "public-picks source file is missing"})
            else:
                try:
                    public_payload = json.loads(_source_path(public_file).read_text())
                except json.JSONDecodeError:
                    public_payload = {}
                    issues.append({"year": str(year), "status": "FAIL", "reason": "public-picks source file is not valid JSON"})
                if isinstance(public_payload, Mapping):
                    raw_timestamp = public_payload.get("captured_at") or public_payload.get("timestamp")
            captured_raw = public.get("captured_at")
            if not isinstance(captured_raw, str) or not isinstance(raw_timestamp, str):
                indeterminate = True
                issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "verified public-picks timestamp is absent"})
            else:
                try:
                    captured_at = datetime.fromisoformat(captured_raw.replace("Z", "+00:00"))
                    source_captured_at = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
                except ValueError:
                    captured_at = None
                    source_captured_at = None
                if (
                    captured_at is None
                    or captured_at.tzinfo is None
                    or source_captured_at is None
                    or source_captured_at.tzinfo is None
                ):
                    indeterminate = True
                    issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "public-picks timestamp is not timezone-aware ISO-8601"})
                elif captured_at != source_captured_at:
                    issues.append({"year": str(year), "status": "FAIL", "reason": "public-picks provenance timestamp differs from the source archive"})
                else:
                    tip = get_round_of_64_tip(year)
                    if captured_at > tip:
                        issues.append({"year": str(year), "status": "FAIL", "reason": "public-picks capture occurred after the R64 lock"})
                    declared_cutoff = public.get("declared_cutoff")
                    if declared_cutoff:
                        try:
                            cutoff = datetime.fromisoformat(str(declared_cutoff).replace("Z", "+00:00"))
                        except ValueError:
                            cutoff = None
                        if cutoff is None or cutoff.tzinfo is None:
                            indeterminate = True
                            issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "declared public-picks cutoff is invalid"})
                        elif captured_at > cutoff:
                            issues.append({"year": str(year), "status": "FAIL", "reason": "public-picks capture occurred after its declared cutoff"})

        if not isinstance(torvik, Mapping) or not isinstance(torvik.get("cutoff_date"), str):
            indeterminate = True
            issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "Torvik cutoff provenance is absent"})
        else:
            try:
                cutoff_date = date.fromisoformat(torvik["cutoff_date"])
            except ValueError:
                cutoff_date = None
            start_date = TOURNAMENT_START_DATES.get(year)
            if cutoff_date is None or start_date is None:
                indeterminate = True
                issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "Torvik cutoff or authoritative tournament date is invalid"})
            elif cutoff_date >= start_date:
                issues.append({"year": str(year), "status": "FAIL", "reason": "Torvik cutoff is not before tournament start"})
            elif torvik.get("tournament_start") != start_date.isoformat():
                issues.append({"year": str(year), "status": "FAIL", "reason": "Torvik provenance declares the wrong tournament start"})
            else:
                torvik_file = torvik.get("file")
                if not isinstance(torvik_file, str) or not _source_path(torvik_file).is_file():
                    indeterminate = True
                    issues.append({"year": str(year), "status": "INDETERMINATE", "reason": "Torvik source file is missing"})
                else:
                    try:
                        torvik_payload = json.loads(_source_path(torvik_file).read_text())
                    except json.JSONDecodeError:
                        torvik_payload = {}
                        issues.append({"year": str(year), "status": "FAIL", "reason": "Torvik source file is not valid JSON"})
                    if not isinstance(torvik_payload, Mapping) or (
                        torvik_payload.get("data_type") != "pre_tournament"
                        or torvik_payload.get("cutoff_date") != cutoff_date.isoformat()
                        or torvik_payload.get("tournament_start") != start_date.isoformat()
                    ):
                        issues.append({"year": str(year), "status": "FAIL", "reason": "Torvik provenance differs from the source snapshot"})

        if not any(issue["year"] == str(year) and issue["status"] == "FAIL" for issue in issues):
            try:
                display_path = path.relative_to(REPO).as_posix()
            except ValueError:
                display_path = str(path)
            artifacts[str(year)] = {
                "path": display_path,
                "sha256": artifact_hash,
                "source_hashes": {
                    str(source["file"]): source["sha256"] for source in _collect_source_records(provenance)
                },
            }

    failed = any(issue["status"] == "FAIL" for issue in issues)
    status = "FAIL" if failed else "INDETERMINATE" if indeterminate else "PASS"
    return {"status": status, "issues": issues, "artifacts": artifacts}


def validate_prospective_artifact(
    artifact: Mapping[str, Any],
    artifact_path: Path,
    year: int,
    spec: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Check the released season artifact against the frozen generation and source contract."""
    from src.data.season_calendar import TOURNAMENT_START_DATES, get_round_of_64_tip
    from src.simulation import bracket_topology as topology

    spec = dict(spec or load_spec())
    if year != spec["release"]["prospective_season"]:
        raise ValueError(f"prospective artifact validator is only for {spec['release']['prospective_season']}")
    failures: list[str] = []
    unknowns: list[str] = []

    if not artifact_path.is_file():
        unknowns.append("candidate artifact is missing")
    else:
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        sidecar = artifact_path.with_suffix(".sha256")
        declared = sidecar.read_text().split() if sidecar.is_file() else []
        if not declared:
            unknowns.append("candidate artifact SHA-256 sidecar is missing")
        elif declared[0] != digest:
            failures.append("candidate artifact SHA-256 disagrees with its sidecar")

    meta = artifact.get("meta")
    expected_defaults = spec["candidate_generation"]["defaults"]
    expected_metadata = {
        "n_sims": expected_defaults["n_sims_total"],
        "candidate_target": spec["candidate_generation"]["candidate_count_target"],
        "p1_trials": expected_defaults["p1_trials"],
        "rng_seed": expected_defaults["rng_seed"],
    }
    if not isinstance(meta, Mapping):
        unknowns.append("candidate artifact metadata is absent")
    else:
        for artifact_key, expected in expected_metadata.items():
            if artifact_key not in meta:
                unknowns.append(f"candidate artifact metadata {artifact_key} is absent")
            elif meta[artifact_key] != expected:
                failures.append(f"candidate artifact {artifact_key}={meta[artifact_key]!r}, expected {expected!r}")
        candidate_total = meta.get("n_candidates")
        if type(candidate_total) is not int:
            unknowns.append("candidate artifact total candidate count is absent")
        elif candidate_total < spec["candidate_generation"]["candidate_count_target"]:
            failures.append("candidate artifact contains fewer candidates than the frozen target")
        if meta.get("pool_settings") != {
            "pool_size": expected_defaults["pool_size"],
            "scoring_id": expected_defaults["scoring_id"],
        }:
            failures.append("candidate artifact pool settings differ from the frozen v4 settings")
        if meta.get("p1_pool_size") != expected_defaults["pool_size"]:
            failures.append("candidate artifact P(1st) pool size differs from the frozen v4 setting")
        generated_at = meta.get("generated_at")
        try:
            generated = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
        except ValueError:
            generated = None
        if generated is None or generated.tzinfo is None:
            unknowns.append("candidate artifact generation timestamp is missing or timezone-naive")
        else:
            selection_sunday = date.fromisoformat(spec["source_cutoffs"]["field"]["selection_sunday"])
            if generated.date() < selection_sunday:
                failures.append("candidate artifact was generated before the frozen Selection Sunday")

    teams = artifact.get("teams")
    first_round = artifact.get("first_round")
    if not isinstance(teams, list) or not isinstance(first_round, list) or len(teams) < 64 or len(first_round) != 64:
        failures.append("candidate artifact does not contain a resolved 64-team main draw")
    elif (
        any(type(index) is not int or not 0 <= index < len(teams) for index in first_round)
        or len(set(first_round)) != 64
    ):
        failures.append("candidate artifact first-round order does not resolve to 64 distinct field teams")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        unknowns.append("candidate artifact provenance is absent")
    else:
        public = provenance.get("public_picks")
        torvik = provenance.get("torvik")
        seed_source = provenance.get("seed_head_to_head")
        field = provenance.get("field")
        ratings = provenance.get("rating_sources")
        rating_config = spec["candidate_generation"]["rating_sources"]
        source_issues = [
            *_source_record_issues(public, "public-picks"),
            *_source_record_issues(torvik, "Torvik"),
            *_source_group_issues(
                field.get("input_files") if isinstance(field, Mapping) else None,
                "field/results",
                minimum=2,
            ),
            *_source_group_issues(
                seed_source.get("input_files") if isinstance(seed_source, Mapping) else None,
                "seed-referee",
                exact=2,
            ),
            *_rating_source_issues(ratings, rating_config["optional"]),
        ]
        for source_issue in source_issues:
            if source_issue["status"] == "FAIL":
                failures.append(source_issue["reason"])
            else:
                unknowns.append(source_issue["reason"])

        if not isinstance(public, Mapping) or public.get("capture_time_verified") is not True:
            unknowns.append("prospective public-picks capture is not verified")
        else:
            raw_captured = public.get("captured_at")
            public_file = public.get("file")
            public_payload: Mapping[str, Any] = {}
            if not isinstance(public_file, str) or not _source_path(public_file).is_file():
                unknowns.append("prospective public-picks source file is missing")
            else:
                try:
                    source_payload = json.loads(_source_path(public_file).read_text())
                except json.JSONDecodeError:
                    source_payload = {}
                    failures.append("prospective public-picks source file is not valid JSON")
                if isinstance(source_payload, Mapping):
                    public_payload = source_payload
            raw_source_time = public_payload.get("captured_at") or public_payload.get("timestamp")
            try:
                captured = datetime.fromisoformat(str(raw_captured).replace("Z", "+00:00"))
                source_captured = datetime.fromisoformat(str(raw_source_time).replace("Z", "+00:00"))
                declared_cutoff = datetime.fromisoformat(
                    spec["source_cutoffs"]["public_picks"]["capture_at_or_before"]
                )
                recorded_cutoff = datetime.fromisoformat(str(public.get("declared_cutoff")).replace("Z", "+00:00"))
            except (KeyError, TypeError, ValueError):
                captured = source_captured = declared_cutoff = recorded_cutoff = None
            if (
                captured is None
                or source_captured is None
                or declared_cutoff is None
                or recorded_cutoff is None
                or captured.tzinfo is None
                or source_captured.tzinfo is None
                or declared_cutoff.tzinfo is None
                or recorded_cutoff.tzinfo is None
            ):
                unknowns.append("prospective public-picks cutoff evidence is incomplete")
            elif captured != source_captured:
                failures.append("prospective public-picks timestamp differs from the archive")
            elif recorded_cutoff != declared_cutoff:
                failures.append("prospective public-picks provenance does not declare the frozen cutoff")
            elif captured > declared_cutoff or captured > get_round_of_64_tip(year):
                failures.append("prospective public-picks capture is after a frozen cutoff")

        if not isinstance(torvik, Mapping):
            unknowns.append("prospective Torvik provenance is absent")
        else:
            start = TOURNAMENT_START_DATES.get(year)
            try:
                cutoff_date = date.fromisoformat(str(torvik.get("cutoff_date")))
            except ValueError:
                cutoff_date = None
            if start is None or cutoff_date is None:
                unknowns.append("prospective Torvik cutoff evidence is incomplete")
            elif (
                cutoff_date >= start
                or torvik.get("data_type") != "pre_tournament"
                or torvik.get("tournament_start") != start.isoformat()
            ):
                failures.append("prospective Torvik snapshot violates the frozen tournament cutoff")
            else:
                torvik_file = torvik.get("file")
                if not isinstance(torvik_file, str) or not _source_path(torvik_file).is_file():
                    unknowns.append("prospective Torvik source file is missing")
                else:
                    try:
                        torvik_payload = json.loads(_source_path(torvik_file).read_text())
                    except json.JSONDecodeError:
                        torvik_payload = {}
                        failures.append("prospective Torvik source file is not valid JSON")
                    if not isinstance(torvik_payload, Mapping) or (
                        torvik_payload.get("data_type") != "pre_tournament"
                        or torvik_payload.get("cutoff_date") != cutoff_date.isoformat()
                        or torvik_payload.get("tournament_start") != start.isoformat()
                    ):
                        failures.append("prospective Torvik provenance differs from the source snapshot")

        if not isinstance(seed_source, Mapping) or seed_source.get("point_in_time") is not True:
            unknowns.append("prospective seed referee point-in-time provenance is absent")
        elif seed_source.get("as_of") != year:
            failures.append("prospective seed referee was not built with the target season as_of")

        pairing = provenance.get("f4_pairing")
        if (
            not isinstance(field, Mapping)
            or field.get("main_draw") != 64
            or type(field.get("play_in_games")) is not int
            or type(field.get("slots_resolved")) is not int
            or field.get("play_in_games") != field.get("slots_resolved")
            or field.get("play_in_games", 0) < 1
            or not isinstance(pairing, list)
            or len(pairing) != 4
            or len(set(pairing)) != 4
        ):
            failures.append("prospective field is unresolved or lacks an explicit Final Four pairing")
        elif isinstance(teams, list) and isinstance(first_round, list):
            team_regions = {
                team.get("id"): team.get("region")
                for team in teams
                if isinstance(team, Mapping)
            }
            try:
                resolved_pairing = list(
                    topology.region_order_from_first_round(
                        [teams[index]["id"] for index in first_round], team_regions
                    )
                )
            except (KeyError, TypeError, ValueError):
                failures.append("prospective artifact topology cannot reproduce its Final Four pairing")
            else:
                if resolved_pairing != pairing:
                    failures.append("prospective artifact Final Four pairing differs from its bracket topology")

        if (
            not isinstance(ratings, Mapping)
            or not isinstance(ratings.get("missing_torvik_team_ratings"), list)
            or not isinstance(ratings.get("used"), list)
        ):
            unknowns.append("prospective rating-source availability or missing-team list is absent")
        elif "torvik" not in ratings["used"]:
            failures.append("mandatory Torvik rating source is absent")
        else:
            missing_team_ids = ratings["missing_torvik_team_ratings"]
            team_ids = {team.get("id") for team in teams if isinstance(team, Mapping)} if isinstance(teams, list) else set()
            if any(not isinstance(team_id, str) or team_id not in team_ids for team_id in missing_team_ids):
                failures.append("missing Torvik rating list contains a team outside the resolved field")
            rating_inputs = ratings.get("inputs")
            optional_sources = spec["candidate_generation"]["rating_sources"]["optional"]
            if not isinstance(rating_inputs, Mapping) or any(
                source not in rating_inputs for source in optional_sources
            ):
                unknowns.append("optional rating-source availability is not fully recorded")

    status = "FAIL" if failures else "INDETERMINATE" if unknowns else "PASS"
    return {"status": status, "failures": failures, "unknowns": unknowns}


def _outcome_hash(actual: Mapping[str, set[str]]) -> str:
    from src.simulation.pool_competition import ROUND_NAMES

    canonical = {round_name: sorted(actual[round_name]) for round_name in ROUND_NAMES}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def _outcome_source_path(year: int) -> Path:
    from scripts._common import HIST_DIR

    context = HIST_DIR / f"tournament_context_{year}.json"
    if context.is_file():
        payload = json.loads(context.read_text())
        if "results" in payload:
            return context
    legacy = HIST_DIR / f"tournament_results_{year}.json"
    return legacy


def _points_for_w(
    artifact: Mapping[str, Any], bracket: Sequence[Sequence[int]], outcome_games: list[dict], year: int
) -> float:
    from scripts import build_track_record
    from src.simulation.pool_competition import actual_winners_by_round, score_brackets_team_identity
    from src.simulation import bracket_topology as topology

    build_track_record._require_complete_outcome(year, outcome_games)
    actual = actual_winners_by_round(outcome_games)
    team_ids = [team["id"] for team in artifact["teams"]]
    winner_ids = [[team_ids[index] for index in rnd] for rnd in bracket]
    first_round = [team_ids[index] for index in artifact["first_round"]]
    row = topology.winner_sets_to_bool_vector(winner_ids, first_round).reshape(1, 63)
    score = score_brackets_team_identity(row, actual, first_round, build_track_record.ESPN_SCORING)
    if score.shape != (1,) or not np.isfinite(score[0]):
        raise RuntimeError(f"{year}: canonical ESPN scorer returned an invalid score")
    return float(score[0])


def build_repository_result(
    *,
    spec_path: Path = SPEC_PATH,
    candidates_dir: Path = REPO / "artifacts" / "candidates",
) -> Dict[str, Any]:
    """Build an auditable evaluation report, refusing scores until source evidence passes."""
    spec = load_spec(spec_path)
    source_gate = inspect_historical_sources(spec, candidates_dir)
    report: Dict[str, Any] = {
        "schema": 1,
        "kind": "prospective_2027_points_selector",
        "spec_hash": spec["spec_hash"],
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_gate": source_gate,
        "inputs": {"candidate_artifacts": source_gate["artifacts"]},
        "promotion_gate": {"status": "INDETERMINATE"},
        "release": {"selected_rule": "seed_only", "fallback": True, "reason": "historical source gate is not PASS"},
    }
    try:
        report["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        report["git_commit"] = None
    report["python_version"] = sys.version.split()[0]
    dependency_manifest = REPO / "pyproject.toml"
    report["dependency_manifest_sha256"] = hashlib.sha256(dependency_manifest.read_bytes()).hexdigest()
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout
        report["git_worktree_clean"] = not dirty.strip()
    except (OSError, subprocess.CalledProcessError):
        report["git_worktree_clean"] = False

    if source_gate["status"] != "PASS":
        report["status"] = source_gate["status"]
        report["reason"] = "No paired score comparison was run because source eligibility is not PASS."
        return report

    from scripts._common import load_tournament_results

    outcomes: Dict[int, list[dict]] = {}
    artifacts: Dict[int, Dict[str, Any]] = {}
    outcomes_hashes: Dict[str, str] = {}
    outcomes_sources: Dict[str, Dict[str, str]] = {}
    for year in spec["release"]["historical_gate"]["target_seasons"]:
        artifact_path = candidates_dir / f"candidates_{year}.json"
        artifact = json.loads(artifact_path.read_text())
        from scripts.build_track_record import _require_complete_outcome
        from src.simulation.pool_competition import actual_winners_by_round

        try:
            source_path = _outcome_source_path(year)
        except json.JSONDecodeError as exc:
            report["status"] = "FAIL"
            report["outcome_gate"] = {"status": "FAIL", "reason": f"{year}: outcome source is not valid JSON"}
            report["reason"] = str(exc)
            return report
        if not source_path.is_file():
            report["status"] = "INDETERMINATE"
            report["outcome_gate"] = {"status": "INDETERMINATE", "reason": f"{year}: outcome source file is missing"}
            report["reason"] = "No paired score comparison was run because a scored outcome source is missing."
            return report
        try:
            outcomes_for_year = load_tournament_results(year)
        except FileNotFoundError as exc:
            report["status"] = "INDETERMINATE"
            report["outcome_gate"] = {"status": "INDETERMINATE", "reason": f"{year}: tournament result file is missing"}
            report["reason"] = str(exc)
            return report
        try:
            _require_complete_outcome(year, outcomes_for_year)
        except RuntimeError as exc:
            report["status"] = "FAIL"
            report["outcome_gate"] = {"status": "FAIL", "reason": str(exc)}
            report["reason"] = f"{year}: tournament outcome is incomplete or malformed"
            return report
        actual = actual_winners_by_round(outcomes_for_year)
        outcomes_hashes[str(year)] = _outcome_hash(actual)
        source_path = _outcome_source_path(year)
        if not source_path.is_file():
            report["status"] = "INDETERMINATE"
            report["outcome_gate"] = {"status": "INDETERMINATE", "reason": f"{year}: outcome source file is missing"}
            report["reason"] = "No paired score comparison was run because a scored outcome source is missing."
            return report
        outcomes_sources[str(year)] = {
            "path": source_path.relative_to(REPO).as_posix(),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
        artifacts[year] = artifact
        outcomes[year] = outcomes_for_year

    points_by_year: Dict[int, Dict[str, float]] = {}
    for year in spec["release"]["historical_gate"]["target_seasons"]:
        artifact = artifacts[year]
        baseline = build_seed_only_bracket(artifact, year)
        row = {"seed_only": _points_for_w(artifact, baseline, outcomes[year], year)}
        for rule_id in RULE_IDS:
            row[rule_id] = _points_for_w(
                artifact, candidate_bracket(artifact, rule_id), outcomes[year], year
            )
        points_by_year[year] = row

    selection = evaluate_nested_scores(points_by_year, spec)
    report.update(selection)
    report["source_gate"] = source_gate
    report["status"] = selection["promotion_gate"]["status"]
    report["inputs"]["outcomes_sha256"] = outcomes_hashes
    report["inputs"]["outcome_sources"] = outcomes_sources
    report["inputs"]["paired_points"] = {
        str(year): points_by_year[year] for year in spec["release"]["historical_gate"]["target_seasons"]
    }
    report["release"]["reason"] = (
        None if not report["release"]["fallback"] else f"promotion gate is {report['promotion_gate']['status']}"
    )
    if report["promotion_gate"]["status"] == "PASS" and not report["git_worktree_clean"]:
        report["release"] = {
            "selected_rule": "seed_only",
            "fallback": True,
            "reason": "The frozen release contract requires a clean committed Git tree.",
        }
        report["release_gate"] = {"status": "INDETERMINATE", "reason": report["release"]["reason"]}
        report["status"] = "INDETERMINATE"
    else:
        report["release_gate"] = {
            "status": "PASS" if report["release"]["selected_rule"] != "seed_only" else report["promotion_gate"]["status"],
            "reason": report["release"]["reason"],
        }
    return report


def write_repository_result(
    result: Mapping[str, Any], output_path: Path = RESULT_PATH
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True).encode()
    output_path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n"
    )
    return digest


def load_current_result(
    *,
    result_path: Path = RESULT_PATH,
    spec: Mapping[str, Any] | None = None,
    candidates_dir: Path = REPO / "artifacts" / "candidates",
) -> Dict[str, Any] | None:
    """Load only a hash-consistent PASS result whose frozen inputs remain unchanged."""
    if not result_path.is_file():
        return None
    sidecar = result_path.with_suffix(result_path.suffix + ".sha256")
    if not sidecar.is_file():
        return None
    payload = result_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    declared = sidecar.read_text().split()
    if not declared or declared[0] != digest:
        return None
    result = json.loads(payload)
    spec = dict(spec or load_spec())
    if (
        result.get("kind") != "prospective_2027_points_selector"
        or result.get("spec_hash") != spec["spec_hash"]
        or result.get("source_gate", {}).get("status") != "PASS"
        or result.get("promotion_gate", {}).get("status") != "PASS"
    ):
        return None
    if inspect_historical_sources(spec, candidates_dir).get("status") != "PASS":
        return None
    for year in spec["release"]["historical_gate"]["target_seasons"]:
        record = result.get("inputs", {}).get("candidate_artifacts", {}).get(str(year))
        artifact_path = candidates_dir / f"candidates_{year}.json"
        if not record or not artifact_path.is_file():
            return None
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != record.get("sha256"):
            return None
        outcome_hash = result.get("inputs", {}).get("outcomes_sha256", {}).get(str(year))
        if not outcome_hash:
            return None
        try:
            from scripts._common import load_tournament_results
            from scripts.build_track_record import _require_complete_outcome
            from src.simulation.pool_competition import actual_winners_by_round

            games = load_tournament_results(year)
            _require_complete_outcome(year, games)
            current_outcome = _outcome_hash(actual_winners_by_round(games))
        except (FileNotFoundError, KeyError, TypeError, RuntimeError):
            return None
        if current_outcome != outcome_hash:
            return None
        source = result.get("inputs", {}).get("outcome_sources", {}).get(str(year))
        if not source:
            return None
        source_path = _source_path(source["path"])
        if not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest() != source.get("sha256"):
            return None
    if result.get("git_worktree_clean") is not True:
        return None
    return result
