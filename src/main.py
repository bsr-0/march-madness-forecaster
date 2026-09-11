"""Main CLI interface for March Madness forecaster.

All command implementations live in src/cli/*_cmds.py modules.
This module is the thin router: it builds the argument parser,
registers every command group, and dispatches via args.func(args).

The ``forecast`` / ``run-production`` / ``backtest-harness`` family of
commands drove the ML pipeline, removed on 2026-09-11 (audit H10: it lost to
the site's fitted model in every season and to the seed table in seven of
nine). What remains is data ingestion, scraping and pool optimisation.
"""

import argparse
import sys

from .cli import (
    data_cmds,
    scrape_cmds,
    pool_cmds,
)

# ---------------------------------------------------------------------------
# All command-group modules in registration order
# ---------------------------------------------------------------------------
_COMMAND_MODULES = [
    data_cmds,
    scrape_cmds,
    pool_cmds,
]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="March Madness Bracket Forecaster - pool-aware bracket optimisation"
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
