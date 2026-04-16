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

import json
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).parent

mcp = FastMCP("march-madness-forecaster")

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


if __name__ == "__main__":
    mcp.run()
