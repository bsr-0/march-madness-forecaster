"""
MCP server for the March Madness forecaster.

Exposes five tools for Claude to query without reading raw JSON files or
shelling out to Python scripts:

  get_leverage_picks      — ranked leverage/fade picks from a pool report
  get_sensitivity_report  — strategy stability under ±5% public pick shifts
  get_backtest_summary    — LOYO model accuracy results
  get_production_config   — current production config parameters
  run_pool_optimization   — fresh optimizer run (requires project deps)

Register with Claude Code by adding mcpServers to .claude/settings.json,
or run directly:  python mcp_server.py
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Optional

import requests
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).parent

mcp = FastMCP("march-madness-forecaster")

_GITHUB_REPO = "bsr-0/march-madness-forecaster"
_GITHUB_API = "https://api.github.com"


def _gh(method: str, path: str, **kwargs) -> requests.Response:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable is not set")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    return requests.request(method, f"{_GITHUB_API}{path}", headers=headers, **kwargs)


# Pool report files by mode
_POOL_REPORTS: dict[str, str] = {
    "blend": "pool_report_blend.json",
    "seed": "pool_report_seed.json",
    "noseed": "pool_report_noseed.json",
    "torvik": "pool_report_n31_torvik.json",
    "n31_blend": "pool_report_n31_blend.json",
    "n31_seed": "pool_report_n31_seed.json",
    "n31_noseed": "pool_report_n31_noseed.json",
    "default": "pool_report.json",
}

# Latest backtest artifacts, newest first
_BACKTEST_FILES = [
    "artifacts/unified_backtest_2025.json",
    "artifacts/unified_backtest_2024.json",
    "artifacts/backtest_result.json",
    "backtest_2024.json",
]


def _load_pool_report(mode: str) -> dict:
    filename = _POOL_REPORTS.get(mode, _POOL_REPORTS["default"])
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"No pool report for mode '{mode}'. Available modes: {list(_POOL_REPORTS.keys())}")
    with open(path) as f:
        return json.load(f)


def _load_latest_backtest() -> tuple[dict, str]:
    for relative in _BACKTEST_FILES:
        path = ROOT / relative
        if path.exists():
            with open(path) as f:
                return json.load(f), relative
    raise FileNotFoundError("No backtest result files found in expected locations.")


@mcp.tool()
def get_leverage_picks(
    mode: str = "blend",
    top_n: int = 10,
    round_filter: Optional[str] = None,
) -> str:
    """
    Return ranked leverage picks (teams to target) and fade picks (teams to avoid)
    from the most recent pool optimization.

    Args:
        mode: Pool report mode — one of blend, seed, noseed, torvik, n31_blend,
              n31_seed, n31_noseed, default.
        top_n: Number of top picks to return per category (default 10).
        round_filter: Optional round name to filter by, e.g. "R64", "R32", "S16",
                      "E8", "F4", "CHAMP".

    Returns:
        Formatted summary of leverage picks, fade picks, recommended strategy,
        and EV comparison between chalk and contrarian play styles.
    """
    report = _load_pool_report(mode)

    leverage = report.get("leverage_picks", [])
    fade = report.get("fade_picks", [])
    strategy_evs = report.get("strategy_evs", {})
    recommended = report.get("recommended_strategy", "unknown")
    year = report.get("year", "unknown")
    pool_size = report.get("pool_size", "unknown")

    if round_filter:
        leverage = [p for p in leverage if p.get("round") == round_filter]
        fade = [p for p in fade if p.get("round") == round_filter]

    leverage = leverage[:top_n]
    fade = fade[:top_n]

    def fmt_pick(p: dict) -> str:
        team = p.get("team_name", p.get("team_id", "?"))
        rnd = p.get("round", "?")
        model_p = p.get("model_probability", 0)
        public_p = p.get("public_pick_percentage", 0)
        ratio = p.get("leverage_ratio", 0)
        ev_diff = p.get("ev_differential", 0)
        return (
            f"  {team} ({rnd}): "
            f"model={model_p:.1%}  public={public_p:.1%}  "
            f"leverage={ratio:.2f}x  ev_diff={ev_diff:+.4f}"
        )

    lines = [
        f"Pool Report — mode={mode}  year={year}  pool_size={pool_size}",
        f"Recommended strategy: {recommended}",
        "",
        "Strategy EVs:",
        f"  chalk_ev      = {strategy_evs.get('chalk_ev', 'n/a'):.4f}"
        if isinstance(strategy_evs.get("chalk_ev"), float)
        else f"  chalk_ev      = {strategy_evs.get('chalk_ev', 'n/a')}",
        f"  contrarian_ev = {strategy_evs.get('contrarian_ev', 'n/a'):.4f}"
        if isinstance(strategy_evs.get("contrarian_ev"), float)
        else f"  contrarian_ev = {strategy_evs.get('contrarian_ev', 'n/a')}",
        f"  leverage_ratio = {strategy_evs.get('leverage_ratio', 'n/a')}",
        "",
        f"Top {len(leverage)} leverage picks (TARGET — undervalued by public):",
    ]
    lines += [fmt_pick(p) for p in leverage] if leverage else ["  (none)"]
    lines += ["", f"Top {len(fade)} fade picks (AVOID — overvalued by public):"]
    lines += [fmt_pick(p) for p in fade] if fade else ["  (none)"]

    return "\n".join(lines)


@mcp.tool()
def get_sensitivity_report(mode: str = "blend") -> str:
    """
    Return the pool strategy sensitivity analysis — does the recommendation change
    if public pick percentages shift by ±5%?

    Args:
        mode: Pool report mode (blend, seed, noseed, torvik, etc.)

    Returns:
        Stability flag (STABLE / HIGH_STRATEGY_UNCERTAINTY), champion under each
        scenario, Final Four changes, and what that means for the submission.
    """
    report = _load_pool_report(mode)
    s = report.get("sensitivity", {})

    if not s:
        return f"No sensitivity data in pool report for mode='{mode}'."

    flag = s.get("flag", "NOT_EVALUATED")
    is_stable = s.get("is_stable", None)
    shift_pct = s.get("shift_pct", 0.05)

    baseline_champ = s.get("baseline_champion", "?")
    up_champ = s.get("shifted_up_champion", "?")
    down_champ = s.get("shifted_down_champion", "?")

    baseline_f4 = s.get("baseline_final_four", [])
    up_f4 = s.get("shifted_up_final_four", [])
    down_f4 = s.get("shifted_down_final_four", [])

    champ_stable = baseline_champ == up_champ == down_champ
    f4_changes_up = sorted(set(up_f4) - set(baseline_f4))
    f4_changes_down = sorted(set(down_f4) - set(baseline_f4))

    lines = [
        f"Sensitivity Report — mode={mode}  shift=±{shift_pct:.0%}",
        f"Flag: {flag}",
        f"Stable: {is_stable}",
        "",
        f"Champion — baseline: {baseline_champ}",
        f"           public +{shift_pct:.0%}: {up_champ}  {'(same)' if up_champ == baseline_champ else '*** CHANGED ***'}",
        f"           public -{shift_pct:.0%}: {down_champ}  {'(same)' if down_champ == baseline_champ else '*** CHANGED ***'}",
        "",
        "Final Four — baseline: " + ", ".join(baseline_f4) if baseline_f4 else "Final Four — baseline: (not recorded)",
    ]
    if f4_changes_up:
        lines.append(f"  Changes (+shift): in={f4_changes_up}")
    if f4_changes_down:
        lines.append(f"  Changes (-shift): in={f4_changes_down}")

    if flag == "HIGH_STRATEGY_UNCERTAINTY":
        lines += [
            "",
            "WARNING: HIGH_STRATEGY_UNCERTAINTY — the recommendation is fragile.",
            "A 5% shift in public sentiment changes the optimal champion or 2+ F4 picks.",
            "Consider hedging across multiple brackets rather than committing to one.",
        ]
    elif flag == "STABLE":
        lines += [
            "",
            "Strategy is stable — public pick shifts do not change the core recommendation.",
        ]

    return "\n".join(lines)


@mcp.tool()
def get_backtest_summary() -> str:
    """
    Return the LOYO (Leave-One-Year-Out) backtest results — Brier scores per mode,
    accuracy, and regression gate status.

    Returns:
        Per-mode Brier score summary, year coverage, and whether the model
        passes the regression gate vs seed baseline (threshold: 0.190).
    """
    data, source = _load_latest_backtest()

    modes = data.get("modes", [])
    years = data.get("years", [])
    summary = data.get("summary_by_mode", {})

    BRIER_GATE = 0.190
    SEED_BASELINE = 0.230

    lines = [
        f"Backtest Summary (source: {source})",
        f"Years: {years}  Modes: {modes}",
        f"Seed baseline Brier: {SEED_BASELINE}  Gate threshold: {BRIER_GATE}",
        "",
    ]

    if summary:
        lines.append("Per-mode results:")
        for mode_name, stats in summary.items():
            if not isinstance(stats, dict):
                continue
            mean_brier = stats.get("mean_brier", "n/a")
            std_brier = stats.get("std_brier", "n/a")
            mean_rw = stats.get("mean_rw_brier", "n/a")
            accuracy = stats.get("mean_accuracy", "n/a")
            n_years = stats.get("n_years", "n/a")

            gate_status = ""
            if isinstance(mean_brier, float):
                gate_status = "PASS" if mean_brier <= BRIER_GATE else "FAIL"
                bss = (SEED_BASELINE - mean_brier) / SEED_BASELINE
                gate_status += f"  BSS={bss:+.3f}"

            brier_str = (
                f"{mean_brier:.4f}±{std_brier:.4f}"
                if isinstance(mean_brier, float) and isinstance(std_brier, float)
                else str(mean_brier)
            )
            rw_str = f"{mean_rw:.4f}" if isinstance(mean_rw, float) else str(mean_rw)
            acc_str = f"{accuracy:.1%}" if isinstance(accuracy, float) else str(accuracy)

            lines.append(
                f"  {mode_name:20s}  brier={brier_str}  rw_brier={rw_str}  "
                f"accuracy={acc_str}  n={n_years}  {gate_status}"
            )
    else:
        # Fall back to raw results structure
        results = data.get("results", {})
        lines.append(f"Raw results keys: {list(results.keys())[:10]}")
        lines.append("(No summary_by_mode found; inspect the source file directly)")

    return "\n".join(lines)


@mcp.tool()
def get_production_config() -> str:
    """
    Return the current production configuration parameters — model hyperparameters,
    calibration settings, simulation parameters, and governance flags.

    Returns:
        Formatted production config from configs/production_2026.json.
    """
    config_path = ROOT / "configs" / "production_2026.json"
    if not config_path.exists():
        # Try to find any production config
        candidates = sorted(ROOT.glob("configs/production_*.json"))
        if not candidates:
            return "No production config found in configs/."
        config_path = candidates[-1]

    with open(config_path) as f:
        config = json.load(f)

    # Format key sections
    lines = [f"Production Config: {config_path.name}", ""]

    key_sections = [
        "model",
        "calibration",
        "simulation",
        "optimization",
        "governance",
        "features",
        "scoring",
    ]

    for section in key_sections:
        val = config.get(section)
        if val is not None:
            lines.append(f"[{section}]")
            if isinstance(val, dict):
                for k, v in val.items():
                    lines.append(f"  {k} = {v}")
            else:
                lines.append(f"  {val}")
            lines.append("")

    # Catch anything not in known sections
    extra = {k: v for k, v in config.items() if k not in key_sections}
    if extra:
        lines.append("[other]")
        for k, v in extra.items():
            if not isinstance(v, (dict, list)):
                lines.append(f"  {k} = {v}")

    return "\n".join(lines)


@mcp.tool()
def run_pool_optimization(
    pool_size: int = 100,
    payout_structure: str = "winner_take_all",
    mode: str = "blend",
) -> str:
    """
    Run a fresh pool optimization and return bracket recommendations.
    Loads current-year model probabilities and public pick data, then
    runs the optimizer with the given pool parameters.

    Requires project dependencies (numpy, scipy, scikit-learn) to be installed.

    Args:
        pool_size: Number of entrants in your pool (default 100).
        payout_structure: One of winner_take_all, top_3, top_10pct, tiered.
        mode: Model blend mode — one of blend, seed, noseed, torvik.

    Returns:
        Recommended strategy, top leverage picks, champion, Final Four,
        and AssumptionsManifest.
    """
    try:
        import sys

        sys.path.insert(0, str(ROOT))

        from src.optimization.pool_optimizer import PoolEnvironment, PoolOptimizer
        from src.pipeline.pipeline import load_current_year_data
    except ImportError as e:
        return (
            f"Cannot run optimizer: missing dependency — {e}\n"
            f"Install project dependencies first: pip install -r requirements.txt\n\n"
            f"Alternatively, use get_leverage_picks(mode='{mode}') to read the "
            f"most recent pre-computed optimization from disk."
        )

    try:
        data = load_current_year_data(mode=mode)
        env = PoolEnvironment(
            pool_size=pool_size,
            scoring_rules={"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320},
            payout_structure=payout_structure,
            public_pick_distribution=data["public_picks"],
        )
        optimizer = PoolOptimizer(
            probabilities=data["pairwise_probs"],
            environment=env,
            model_round_probs=data.get("round_probs"),
        )
        result = optimizer.optimize()
        sens = optimizer.sensitivity_analysis()

        lines = [
            f"Fresh Optimization — pool_size={pool_size}  payout={payout_structure}  mode={mode}",
            f"Recommended strategy: {result.recommended_strategy}",
            f"Sensitivity flag: {result.manifest.sensitivity_flag}",
            "",
            "Strategy EVs:",
        ]
        for strat, ev in result.strategy_evs.items():
            if isinstance(ev, float):
                lines.append(f"  {strat}: {ev:.4f}")
        lines += [
            "",
            "Top leverage picks:",
        ]
        for p in result.leverage_picks[:8]:
            team = p.get("team_name", p.get("team_id", "?"))
            rnd = p.get("round", "?")
            ratio = p.get("leverage_ratio", 0)
            lines.append(f"  {team} ({rnd}) — leverage={ratio:.2f}x")

        lines += [
            "",
            f"Manifest: pool_size={result.manifest.pool_size}  "
            f"payout={result.manifest.payout_structure}  "
            f"timestamp={result.manifest.timestamp}",
        ]
        return "\n".join(lines)

    except Exception as e:
        return (
            f"Optimizer run failed: {e}\n\n"
            f"Tip: use get_leverage_picks(mode='{mode}') to read the last "
            f"pre-computed result instead."
        )


# ---------------------------------------------------------------------------
# GitHub tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_branches(pattern: Optional[str] = None) -> str:
    """
    List remote branches for this repository, optionally filtered by a prefix.

    Args:
        pattern: Optional substring to filter branch names (e.g. "claude/").

    Returns:
        Newline-separated list of matching branch names.
    """
    r = _gh("GET", f"/repos/{_GITHUB_REPO}/branches", params={"per_page": 100})
    r.raise_for_status()
    names = [b["name"] for b in r.json()]
    if pattern:
        names = [n for n in names if pattern in n]
    return "\n".join(names) if names else "No branches found."


@mcp.tool()
def delete_branch(branch_name: str, confirm: bool = False) -> str:
    """
    Delete a remote branch. Restricted to claude/* branches only.

    Args:
        branch_name: Branch to delete — must start with "claude/".
        confirm: Must be True to execute; prevents accidental deletion.

    Returns:
        Success or refusal message.
    """
    if not branch_name.startswith("claude/"):
        return f"Refused: only claude/* branches may be deleted. Got '{branch_name}'."
    if not confirm:
        return f"Refused: set confirm=True to delete '{branch_name}'."
    r = _gh("DELETE", f"/repos/{_GITHUB_REPO}/git/refs/heads/{branch_name}")
    if r.status_code == 204:
        return f"Deleted branch '{branch_name}'."
    return f"Failed ({r.status_code}): {r.text}"


@mcp.tool()
def list_pull_requests(state: str = "open") -> str:
    """
    List pull requests for this repository.

    Args:
        state: Filter by state — "open" (default), "closed", or "all".

    Returns:
        Formatted list of PRs with number, title, branches, and author.
    """
    r = _gh("GET", f"/repos/{_GITHUB_REPO}/pulls", params={"state": state, "per_page": 30})
    r.raise_for_status()
    prs = r.json()
    if not prs:
        return f"No {state} pull requests."
    lines = [
        f"PR #{pr['number']} [{pr['state']}] {pr['title']} "
        f"— {pr['head']['ref']} → {pr['base']['ref']} (@{pr['user']['login']})"
        for pr in prs
    ]
    return "\n".join(lines)


@mcp.tool()
def get_workflow_runs(workflow: Optional[str] = None, limit: int = 10) -> str:
    """
    List recent CI workflow runs for this repository.

    Args:
        workflow: Optional workflow filename or ID to filter by (e.g. "auto-merge-claude.yml").
        limit: Max runs to return (default 10).

    Returns:
        Formatted list of runs with ID, name, status, conclusion, branch, and timestamp.
    """
    params = {"per_page": limit}
    if workflow:
        r = _gh("GET", f"/repos/{_GITHUB_REPO}/actions/workflows/{workflow}/runs", params=params)
    else:
        r = _gh("GET", f"/repos/{_GITHUB_REPO}/actions/runs", params=params)
    r.raise_for_status()
    runs = r.json().get("workflow_runs", [])
    if not runs:
        return "No workflow runs found."
    lines = [
        f"#{run['id']}  {run['name']}  [{run['status']}/{run['conclusion']}]"
        f"  branch={run['head_branch']}  {run['created_at']}"
        for run in runs
    ]
    return "\n".join(lines)


@mcp.tool()
def create_pull_request(
    title: str,
    head: str,
    body: str = "",
    base: str = "main",
) -> str:
    """
    Create a pull request from a head branch into base (default: main).

    Args:
        title: PR title.
        head: Source branch name.
        body: Optional PR description (markdown supported).
        base: Target branch (default "main").

    Returns:
        PR number and URL on success, or error message.
    """
    r = _gh(
        "POST",
        f"/repos/{_GITHUB_REPO}/pulls",
        json={"title": title, "body": body, "head": head, "base": base},
    )
    if r.status_code == 201:
        pr = r.json()
        return f"Created PR #{pr['number']}: {pr['html_url']}"
    return f"Failed ({r.status_code}): {r.json().get('message', r.text)}"


@mcp.tool()
def get_workflow_run_logs(run_id: int) -> str:
    """
    Fetch logs for a CI workflow run. Returns the last 20 000 characters.

    Args:
        run_id: Numeric workflow run ID (visible in get_workflow_runs output).

    Returns:
        Concatenated log content from all jobs, truncated to last 20 000 chars.
    """
    # GitHub returns a 302 redirect to a pre-signed zip download URL.
    r = _gh(
        "GET",
        f"/repos/{_GITHUB_REPO}/actions/runs/{run_id}/logs",
        allow_redirects=False,
    )
    if r.status_code == 302:
        r = requests.get(r.headers["Location"])
    if r.status_code != 200:
        return f"Failed to fetch logs ({r.status_code}): {r.text}"

    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            parts = [
                f"=== {name} ===\n{zf.read(name).decode('utf-8', errors='replace')}" for name in sorted(zf.namelist())
            ]
        combined = "\n\n".join(parts)
        max_chars = 20_000
        if len(combined) > max_chars:
            return combined[-max_chars:] + "\n\n[truncated — showing last 20 000 chars]"
        return combined
    except Exception as e:
        return f"Log parse error: {e}"


@mcp.tool()
def merge_pull_request(pr_number: int, merge_method: str = "rebase") -> str:
    """
    Merge a pull request. Defaults to rebase to preserve linear history.

    Args:
        pr_number: PR number to merge.
        merge_method: "rebase" (default), "squash", or "merge".

    Returns:
        Success message or error details.
    """
    if merge_method not in ("rebase", "squash", "merge"):
        return f"Invalid merge_method '{merge_method}'. Use: rebase, squash, merge."
    r = _gh(
        "PUT",
        f"/repos/{_GITHUB_REPO}/pulls/{pr_number}/merge",
        json={"merge_method": merge_method},
    )
    if r.status_code == 200:
        return f"PR #{pr_number} merged via {merge_method}."
    return f"Failed ({r.status_code}): {r.json().get('message', r.text)}"


@mcp.tool()
def create_branch(branch_name: str, from_ref: str = "main") -> str:
    """
    Create a new remote branch from an existing ref.

    Args:
        branch_name: Name for the new branch.
        from_ref: Branch, tag, or commit SHA to branch from (default "main").

    Returns:
        Success message with the new branch SHA, or error details.
    """
    # Resolve the ref to a SHA first
    r = _gh("GET", f"/repos/{_GITHUB_REPO}/git/ref/heads/{from_ref}")
    if r.status_code != 200:
        # Try as a tag
        r = _gh("GET", f"/repos/{_GITHUB_REPO}/git/ref/tags/{from_ref}")
    if r.status_code != 200:
        return f"Could not resolve ref '{from_ref}' ({r.status_code}): {r.json().get('message', r.text)}"
    sha = r.json()["object"]["sha"]

    r = _gh(
        "POST",
        f"/repos/{_GITHUB_REPO}/git/refs",
        json={"ref": f"refs/heads/{branch_name}", "sha": sha},
    )
    if r.status_code == 201:
        return f"Created branch '{branch_name}' at {sha[:7]} (from {from_ref})."
    return f"Failed ({r.status_code}): {r.json().get('message', r.text)}"


@mcp.tool()
def trigger_workflow(workflow_id: str, ref: str = "main", inputs: Optional[dict] = None) -> str:
    """
    Trigger a workflow_dispatch event for a GitHub Actions workflow.

    Args:
        workflow_id: Workflow filename (e.g. "data-ingestion.yml") or numeric ID.
        ref: Branch or tag to run the workflow on (default "main").
        inputs: Optional dict of workflow input parameters.

    Returns:
        Success message or error details.
    """
    r = _gh(
        "POST",
        f"/repos/{_GITHUB_REPO}/actions/workflows/{workflow_id}/dispatches",
        json={"ref": ref, "inputs": inputs or {}},
    )
    if r.status_code == 204:
        return f"Triggered workflow '{workflow_id}' on ref '{ref}'."
    return f"Failed ({r.status_code}): {r.json().get('message', r.text)}"


@mcp.tool()
def get_pr_review_comments(pr_number: int) -> str:
    """
    Fetch review comments (inline code comments) on a pull request.

    Args:
        pr_number: PR number.

    Returns:
        Formatted list of review comments with author, file, line, and body.
    """
    r = _gh("GET", f"/repos/{_GITHUB_REPO}/pulls/{pr_number}/comments", params={"per_page": 50})
    r.raise_for_status()
    comments = r.json()
    if not comments:
        return f"No review comments on PR #{pr_number}."
    lines = [f"PR #{pr_number} review comments ({len(comments)} total)", ""]
    for c in comments:
        path = c.get("path", "?")
        line = c.get("line") or c.get("original_line", "?")
        author = c["user"]["login"]
        body = c["body"].strip()
        lines.append(f"@{author}  {path}:{line}")
        lines.append(f"  {body}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def add_pr_comment(pr_number: int, body: str) -> str:
    """
    Post a top-level comment on a pull request.

    Args:
        pr_number: PR number.
        body: Comment text (markdown supported).

    Returns:
        URL of the created comment, or error details.
    """
    r = _gh(
        "POST",
        f"/repos/{_GITHUB_REPO}/issues/{pr_number}/comments",
        json={"body": body},
    )
    if r.status_code == 201:
        return f"Comment posted: {r.json()['html_url']}"
    return f"Failed ({r.status_code}): {r.json().get('message', r.text)}"


@mcp.tool()
def get_file_contents(path: str, ref: str = "main") -> str:
    """
    Read a file from the repository without a local checkout.

    Args:
        path: File path relative to repo root (e.g. "configs/production_2026.json").
        ref: Branch, tag, or commit SHA (default "main").

    Returns:
        Decoded file contents, truncated to 20 000 chars if large.
    """
    import base64

    r = _gh("GET", f"/repos/{_GITHUB_REPO}/contents/{path}", params={"ref": ref})
    if r.status_code == 404:
        return f"File not found: '{path}' on ref '{ref}'."
    r.raise_for_status()
    data = r.json()
    if data.get("type") != "file":
        return f"'{path}' is not a file (type={data.get('type')})."
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    max_chars = 20_000
    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n[truncated — file is {len(content)} chars]"
    return content


@mcp.tool()
def list_issues(state: str = "open") -> str:
    """
    List issues for this repository.

    Args:
        state: Filter by state — "open" (default), "closed", or "all".

    Returns:
        Formatted list of issues with number, title, labels, and author.
    """
    r = _gh("GET", f"/repos/{_GITHUB_REPO}/issues", params={"state": state, "per_page": 30})
    r.raise_for_status()
    # GitHub issues endpoint also returns PRs; filter them out
    issues = [i for i in r.json() if "pull_request" not in i]
    if not issues:
        return f"No {state} issues."
    lines = []
    for i in issues:
        labels = ", ".join(lb["name"] for lb in i.get("labels", []))
        label_str = f"  [{labels}]" if labels else ""
        lines.append(f"#{i['number']} {i['title']}{label_str} (@{i['user']['login']})")
    return "\n".join(lines)


@mcp.tool()
def run_python(code: str, timeout: int = 30) -> str:
    """
    Execute Python code in the project environment and return stdout + stderr.

    Runs in a subprocess with cwd set to the repo root so imports like
    `from src.optimization.pool_optimizer import ...` work out of the box.
    Output is capped at 20 000 chars. File writes are not blocked but the
    subprocess has no elevated permissions beyond the current user.

    Args:
        code: Python source code to execute.
        timeout: Max seconds before the subprocess is killed (default 30).

    Returns:
        Combined stdout and stderr output, or a timeout/error message.
    """
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + "--- stderr ---\n" + result.stderr
        if not output:
            return f"(no output, exit code {result.returncode})"
        max_chars = 20_000
        if len(output) > max_chars:
            return output[:max_chars] + f"\n\n[truncated — {len(output)} chars total]"
        return output
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout}s."
    except Exception as e:
        return f"Failed to run subprocess: {e}"


if __name__ == "__main__":
    mcp.run()
