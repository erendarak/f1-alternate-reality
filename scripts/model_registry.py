"""Check every model manifest against its artifact and the lap dataset; write the registry page.

Usage:
    uv run python scripts/model_registry.py            # check + write docs/MODEL_REGISTRY.md
    uv run python scripts/model_registry.py --check    # check only (exit 1 on a problem)
"""

import argparse
import sys
from pathlib import Path

from f1ar.calibration.registry import load_registry, registry_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="don't write the page")
    args = parser.parse_args()
    entries = load_registry(ROOT)
    bad = 0
    for e in entries:
        problems = e.problems
        bad += bool(problems)
        state = "; ".join(problems) or "ok"
        engine = e.used_by_engine or "-"
        print(f"{e.model_name:<24} {e.model_version:<3} engine: {engine:<38} {state}")
    if not args.check:
        (ROOT / "docs" / "MODEL_REGISTRY.md").write_text(
            registry_markdown(entries), encoding="utf-8", newline="\n"
        )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
