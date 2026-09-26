"""Audit historical source eligibility and run the frozen v4 selector when eligible."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.prospective_2027_points import (  # noqa: E402
    RESULT_PATH,
    build_repository_result,
    write_repository_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULT_PATH)
    parser.add_argument("--candidates-dir", type=Path, default=REPO / "artifacts" / "candidates")
    args = parser.parse_args()

    result = build_repository_result(candidates_dir=args.candidates_dir)
    digest = write_repository_result(result, args.out)
    print(json.dumps({
        "status": result["status"],
        "source_gate": result["source_gate"]["status"],
        "promotion_gate": result["promotion_gate"]["status"],
        "selected_rule": result["release"]["selected_rule"],
        "result": str(args.out),
        "sha256": digest,
        "reason": result.get("reason"),
    }, indent=2))
    if result["status"] == "PASS":
        return 0
    return 2 if result["status"] == "INDETERMINATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
