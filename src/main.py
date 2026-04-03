"""Main CLI interface for March Madness forecaster.

All command implementations live in src/cli/*_cmds.py modules.
This module is the thin router: it builds the argument parser,
registers every command group, and dispatches via args.func(args).
"""

import argparse
import sys

from .cli import (
    pipeline_cmds,
    data_cmds,
    scrape_cmds,
    eval_cmds,
    export_cmds,
    research_cmds,
    ops_cmds,
    live_protocol_cmds,
    pool_cmds,
)

# ---------------------------------------------------------------------------
# Backward-compatible re-exports used by tests
# ---------------------------------------------------------------------------
from .cli._helpers import (  # noqa: F401
    _resolve_multi_year_dir,
    _parse_year_list,
    _parse_float_list,
    _build_pipeline_config,
    _guard_production_2026,
)
from .cli.pipeline_cmds import run_production_2026_cmd  # noqa: F401


# ---------------------------------------------------------------------------
# All command-group modules in registration order
# ---------------------------------------------------------------------------
_COMMAND_MODULES = [
    pipeline_cmds,
    data_cmds,
    scrape_cmds,
    eval_cmds,
    export_cmds,
    research_cmds,
    ops_cmds,
    live_protocol_cmds,
    pool_cmds,
]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="March Madness Bracket Forecaster - Mathematically robust tournament predictions"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    for module in _COMMAND_MODULES:
        module.register(subparsers)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
