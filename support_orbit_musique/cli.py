"""Command-line entry point for the frozen MuSiQue builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DEFAULT_OUTPUT, EXPECTED_SOURCE_PATH, BuildConfig, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic C/D/M orbits from the official MuSiQue train split."
    )
    parser.add_argument("--source", type=Path, default=EXPECTED_SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_dataset(args.source, args.output, BuildConfig())
    print(json.dumps(result.manifest["counts"], ensure_ascii=False, sort_keys=True))
    print(f"release_status={result.manifest['release_gate']['status']}")


if __name__ == "__main__":
    main()
