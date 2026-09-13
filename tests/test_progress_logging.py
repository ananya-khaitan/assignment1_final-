import json
import tempfile
from pathlib import Path

from code.utils.progress import CompanyProgressLogger
from code.watch_run_progress import render


def test_progress_logger_writes_observed_rate_and_eta():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "glencore.jsonl"
        logger = CompanyProgressLogger(path, run_id="run_test", company="Glencore", total_origins=10)
        logger.started(calibration_total=2, test_total=8)
        logger.record(
            "running",
            partition="test",
            partition_completed=3,
            partition_total=8,
            overall_completed=5,
            target_date="2024-01-10",
            refit=False,
        )
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert events[0]["status"] == "started"
        latest = events[-1]
        assert latest["overall_completed"] == 5
        assert latest["overall_total"] == 10
        assert latest["origins_per_second"] is not None
        assert latest["eta_seconds"] is not None
        assert latest["estimated_finish_utc"] is not None


def test_progress_view_renders_company_and_overall_completion():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "glencore.jsonl"
        logger = CompanyProgressLogger(path, run_id="run_test", company="Glencore", total_origins=10)
        logger.record("completed", overall_completed=10)
        view = render(Path(directory))
        assert "Glencore" in view
        assert "10/10" in view
        assert "Overall:" in view
