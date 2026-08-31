#!/usr/bin/env python3
"""Evaluate one canonical receipt-bound development generation arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from support_orbit_musique.backend import CANONICAL_RECEIPT, validate_launch_receipt
from support_orbit_musique.evaluation import GENERATION_ARMS, evaluate_production_arm


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--launch-receipt", type=Path, required=True)
    value.add_argument("--generation-manifest", type=Path, required=True)
    value.add_argument("--arm", required=True, choices=GENERATION_ARMS)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # This must remain the first body-reading operation: the backend verifies
    # canonical receipt status, sidecar, complete source lock, protocol, and
    # public provenance before the generation manifest or dev body is opened.
    receipt = validate_launch_receipt(args.launch_receipt, purpose="generation")
    if Path(args.launch_receipt).expanduser().resolve() != CANONICAL_RECEIPT:
        raise ValueError("evaluation accepts only the canonical launch receipt")
    report, destination = evaluate_production_arm(
        receipt,
        arm=args.arm,
        generation_manifest_path=args.generation_manifest,
        write=True,
    )
    print(
        json.dumps(
            {
                "arm": args.arm,
                "evaluation": str(destination),
                "orbits": report["counts"]["orbits"],
                "parse_rate": report["orbit_metrics"]["parse_rate"],
                "run_integrity": report["run_integrity"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
