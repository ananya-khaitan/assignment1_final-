"""Retired legacy-diagram hook.

The prior schematics described interval-gated trading and per-IMF behavior
that differed from the causal implementation. New figures must be generated
from, and listed in, a successful causal-run manifest.
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Legacy methodology diagrams are disabled. Generate new figures from "
        "canonical causal forecasts before adding them to a report."
    )


if __name__ == "__main__":
    main()
