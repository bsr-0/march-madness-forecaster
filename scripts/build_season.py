"""Build one season's bracket bank and the UI payload, in one command.

    python -m scripts.build_season --year 2027

What this replaces: a manual, undocumented three-step sequence (refresh
team stats, build the candidate artifact, rebuild the UI payload) that used
to live only in a runbook document that was deleted in the 2026-08-18
FINDINGS.md consolidation and, when recovered from git history, turned out
to describe a pipeline that no longer exists (a single CLI-selected bracket
manually typed into ESPN, not the browser-filterable candidate bank the
site ships today). See AUDIT_INDEPENDENT_EVALUATOR_2027.md recommendation 8.

Runs, in order:
  1. scripts/generate_team_stats_table.py   -- refreshes docs/data/team_stats_by_year.json
                                                for every season (cheap; always current).
  2. scripts/experiments/build_candidate_artifact.py --year YEAR
                                             -- writes artifacts/candidates/candidates_{YEAR}.json
                                                (~150k simulations; several minutes).
  3. scripts/build_ui_payload.py            -- rebuilds every docs/data/season_*.json and
                                                seasons.json from whatever candidate artifacts
                                                exist on disk. Rerunning it costs nothing and
                                                keeps every season's payload current, not just
                                                the one just built.

Pass `--generated-at` only for an artifact replay; it pins the original
timezone-aware timestamp so the candidate JSON checksum can match.

Deliberately does NOT commit or push. docs/data/*.json is the only output
this script changes that belongs in git (artifacts/candidates/*.json is
gitignored except its .sha256), and putting a live pool bracket in front of
users is exactly the kind of action that should require a human to look at
the printed validation numbers first. See RUNBOOK_2027.md for the review
checklist and the deploy step (a plain `git push` of docs/data/ triggers
.github/workflows/deploy-docs-on-push.yml -- no separate deploy command).

Step 2's own guard refuses to rebuild an official season's artifact once it
exists (PROSPECTIVE_2027 CHECKPOINT 2's immutability rule) -- see
`_refuse_to_overwrite` in build_candidate_artifact.py. This script does not
work around that; pass --force to ask step 2 to, and note it will still
refuse outright for an official season no matter what.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, args: list[str]) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    t0 = time.monotonic()
    result = subprocess.run([sys.executable, "-m", *args], cwd=PROJECT_ROOT)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        raise SystemExit(
            f"\n{label} failed (exit {result.returncode}) after {elapsed:.0f}s -- stopping. "
            f"Nothing after this step ran, and docs/data/ is unchanged for {label}'s output."
        )
    print(f"-- {label} done in {elapsed:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True, help="season to build, e.g. 2027")
    ap.add_argument("--n-sims", type=int, default=150_000)
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument(
        "--generated-at",
        help="timezone-aware ISO timestamp to preserve when replaying an existing artifact",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="allow overwriting an existing candidate artifact for this year "
        "(still refused outright for an official prospective season)",
    )
    ap.add_argument(
        "--skip-stats",
        action="store_true",
        help="skip step 1 (team_stats_by_year.json is already current)",
    )
    args = ap.parse_args()

    if not args.skip_stats:
        _run("1/3  Refreshing team_stats_by_year.json", ["scripts.generate_team_stats_table"])
    else:
        print("1/3  Skipped (--skip-stats)")

    build_cmd = [
        "scripts.experiments.build_candidate_artifact",
        "--year",
        str(args.year),
        "--n-sims",
        str(args.n_sims),
        "--target",
        str(args.target),
        "--trials",
        str(args.trials),
        "--seed",
        str(args.seed),
    ]
    if args.generated_at is not None:
        build_cmd.extend(["--generated-at", args.generated_at])
    if args.force:
        build_cmd.append("--force")
    _run(f"2/3  Building candidate artifact for {args.year}", build_cmd)

    _run("3/3  Rebuilding docs/data/season_*.json + seasons.json", ["scripts.build_ui_payload"])

    print(
        f"\n{'=' * 78}\n"
        f"Done. Read the validation block printed by step 2/3 above before doing anything else.\n"
        f"Next: review docs/data/season_{args.year}.json (or open docs/index.html locally), then\n"
        f"  git add docs/data/season_{args.year}.json docs/data/seasons.json docs/data/team_stats_by_year.json\n"
        f"  git commit -m 'Build {args.year} bracket bank'\n"
        f"  git push\n"
        f"which triggers deploy-docs-on-push.yml -- no separate deploy step.\n"
        f"{'=' * 78}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
