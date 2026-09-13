"""Retired legacy-table hook.

The old static tables describe ApEn/interval-gated methodology that no longer
matches the causal runner. Current tables are emitted directly by
``run_experiment.py`` and included by ``export_docx.py``.
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "This legacy table generator is disabled. Run the causal experiment; "
        "its current CSV outputs are the sole report source."
    )


if __name__ == "__main__":
    main()
