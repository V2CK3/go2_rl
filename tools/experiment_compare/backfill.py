#!/usr/bin/env python3
"""Backfill config.json / template eval_results.json for existing training runs."""

from __future__ import annotations

import argparse
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _TOOLS_DIR)

from experiment_compare.scanner import (  # noqa: E402
    DEFAULT_EVAL_RESULTS,
    DEFAULT_LOGS_DIR,
    backfill_all,
    discover_runs,
    save_eval_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing config.json")
    parser.add_argument(
        "--eval-template",
        action="store_true",
        help="Write eval_results.json template where missing",
    )
    args = parser.parse_args()

    written = backfill_all(args.logs_dir, force=args.force)
    print(f"Wrote {len(written)} config.json file(s):")
    for path in written:
        print(f"  {path}")

    if args.eval_template:
        for run in discover_runs(args.logs_dir, load_metrics=False):
            path = os.path.join(run.run_dir, "eval_results.json")
            if os.path.isfile(path) and not args.force:
                continue
            save_eval_results(run.run_dir, DEFAULT_EVAL_RESULTS)
            print(f"  eval template: {path}")


if __name__ == "__main__":
    main()
