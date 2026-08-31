#!/usr/bin/env python3
"""Compare only canonical CONTROL versus HopPAIR development reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from support_orbit_musique.backend import CANONICAL_RECEIPT, validate_launch_receipt
from support_orbit_musique.evaluation import compare_production


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--launch-receipt", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    receipt = validate_launch_receipt(args.launch_receipt, purpose="generation")
    if Path(args.launch_receipt).expanduser().resolve() != CANONICAL_RECEIPT:
        raise ValueError("comparison accepts only the canonical launch receipt")
    comparison, destination = compare_production(receipt, write=True)
    print(
        json.dumps(
            {
                "comparison": str(destination),
                "decision": comparison["decision"],
                "paired_orbits": comparison["counts"]["paired_orbits"],
                "roles": comparison["production"]["roles"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
