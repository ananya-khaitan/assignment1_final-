"""Single explicit entrypoint for the reproducible empirical pipeline."""

from __future__ import annotations

import runpy
from pathlib import Path

from evaluation.run_model_selection_tests import run_tests
from run_experiment import main as run_experiment
from run_fixed_models import run_fixed_models
from utils.legacy import snapshot_pre_causal_outputs


def main() -> None:
    code_root = Path(__file__).resolve().parent
    snapshot = snapshot_pre_causal_outputs()
    if snapshot is not None:
        print(f"Archived pre-refactor generated outputs to {snapshot}")
    # build_master is intentionally executed first so derived data cannot be
    # mistaken for current raw-data output.
    runpy.run_path(str(code_root / "build_master.py"), run_name="__main__")
    run_experiment()
    run_fixed_models()
    run_tests()


if __name__ == "__main__":
    main()
