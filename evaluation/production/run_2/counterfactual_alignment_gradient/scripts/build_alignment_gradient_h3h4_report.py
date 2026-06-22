from __future__ import annotations

"""CLI wrapper for the branch-local alignment-gradient H3/H4 report builder.

The implementation lives in `alignment_gradient_report/`; this file keeps the launcher-facing command stable.
"""

import argparse
import json

from alignment_gradient_report.pipeline import build_report


def parse_args() -> argparse.Namespace:
    """Parse the stable branch report CLI arguments."""
    parser = argparse.ArgumentParser(description="Build the alignment-gradient H3/H4 branch report")
    parser.add_argument("--branch-root", required=True)
    parser.add_argument("--run-id", default="run_2_alignment_gradient")
    parser.add_argument(
        "--report-mode",
        choices=["h3h4", "branch_network", "full"],
        default="full",
        help="h3h4 builds the core hypothesis figures; branch_network builds only adapted branch network figures; full builds both.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the requested report and print its manifest as JSON."""
    args = parse_args()
    manifest = build_report(args.branch_root, args.run_id, args.report_mode)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
