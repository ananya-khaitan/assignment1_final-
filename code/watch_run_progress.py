"""Render live ETA snapshots from a causal experiment's JSONL progress logs.

Examples (from ``code/``)::

    python watch_run_progress.py --once
    python watch_run_progress.py --follow --refresh-seconds 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:  # Support both `python watch_run_progress.py` and package-based tests.
    from .config import OUT_ROOT, ORDER
except ImportError:  # pragma: no cover - direct script execution from code/
    from config import OUT_ROOT, ORDER


def _latest_event(path: Path) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            latest = json.loads(line)
        except json.JSONDecodeError:
            # A process may be appending while this reader opens the file.
            continue
    return latest


def _format_seconds(value: float | int | None) -> str:
    if value is None:
        return "calculating"
    seconds = max(0, int(round(float(value))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes:d}m {seconds:02d}s"


def _resolve_log_dir(run_id: str | None) -> Path:
    root = Path(OUT_ROOT) / "run_logs"
    if run_id:
        directory = root / run_id
    else:
        candidates = sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No run logs found in {root}")
        directory = candidates[-1]
    if not directory.exists():
        raise FileNotFoundError(f"Run-log directory does not exist: {directory}")
    return directory


def render(log_dir: Path) -> str:
    events = {path.stem: _latest_event(path) for path in log_dir.glob("*.jsonl")}
    lines = [f"Run progress: {log_dir.name}", ""]
    total_done = total_work = 0
    for company in ORDER:
        key = company.lower().replace(" ", "_").replace("-", "_")
        event = events.get(key)
        if event is None:
            lines.append(f"{company:<28} waiting")
            continue
        completed = int(event["overall_completed"])
        total = int(event["overall_total"])
        total_done += completed
        total_work += total
        lines.append(
            f"{company:<28} {event['status']:<9} {completed:>3}/{total:<3} "
            f"ETA {_format_seconds(event.get('eta_seconds')):<10} "
            f"finish {event.get('estimated_finish_utc') or 'calculating'}"
        )
    if total_work:
        lines.extend(["", f"Overall: {total_done}/{total_work} origins recorded ({total_done / total_work * 100:.1f}%)."])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show live causal-experiment progress and ETA.")
    parser.add_argument("--run-id", help="Run directory name under outputs/run_logs; defaults to the latest.")
    parser.add_argument("--follow", action="store_true", help="Refresh until interrupted.")
    parser.add_argument("--refresh-seconds", type=float, default=20.0)
    parser.add_argument("--once", action="store_true", help="Print a single snapshot (the default).")
    args = parser.parse_args()
    if args.refresh_seconds <= 0:
        raise ValueError("--refresh-seconds must be positive")
    log_dir = _resolve_log_dir(args.run_id)
    while True:
        print(render(log_dir), flush=True)
        if not args.follow:
            break
        time.sleep(args.refresh_seconds)


if __name__ == "__main__":
    main()
