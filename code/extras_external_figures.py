"""Retired live-data figure hook.

The prior implementation queried Crossref at execution time and generated
non-versioned figures unrelated to a specific empirical run. It is disabled
until a versioned literature-data snapshot and a causal-figure manifest exist.
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Live external figure generation is disabled for reproducibility. "
        "Use a versioned source snapshot and register generated figures in "
        "outputs/figures/causal_figure_manifest.txt."
    )


if __name__ == "__main__":
    main()
