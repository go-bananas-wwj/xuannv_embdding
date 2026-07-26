#!/usr/bin/env python3
"""Create a write-once admission record for one paired V5 bootstrap comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.report import admit_registered_results as admission


def parse_args() -> argparse.Namespace:
    """Parse every identity needed to admit one derived paired comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-release-admission", type=Path, required=True)
    parser.add_argument("--candidate-release-admission", type=Path, required=True)
    parser.add_argument("--bootstrap-report", type=Path, required=True)
    parser.add_argument("--baseline-family", required=True)
    parser.add_argument("--candidate-family", required=True)
    return parser.parse_args()


def main() -> None:
    """Write and print the immutable comparison admission identity."""
    args = parse_args()
    admission.validate_comparison_admission_output_path(
        args.output,
        baseline_family=args.baseline_family,
        candidate_family=args.candidate_family,
    )
    payload = admission.write_comparison_admission(
        output_path=args.output,
        baseline_release_admission_path=args.baseline_release_admission,
        candidate_release_admission_path=args.candidate_release_admission,
        bootstrap_report_path=args.bootstrap_report,
        baseline_family=args.baseline_family,
        candidate_family=args.candidate_family,
    )
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "sha256": payload["bootstrap_report"]["sha256"]}
        )
    )


if __name__ == "__main__":
    main()
