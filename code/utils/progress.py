"""Durable, process-safe progress events for long empirical runs.

Each company writes to its own JSONL file, avoiding cross-process locks.  A
record includes an observed throughput and an ETA derived only from work that
has already completed in that company process.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class CompanyProgressLogger:
    """Append concise, immediately visible progress records for one asset."""

    def __init__(self, path: str | Path, *, run_id: str, company: str, total_origins: int) -> None:
        if total_origins < 1:
            raise ValueError("total_origins must be positive")
        self.path = Path(path)
        self.run_id = run_id
        self.company = company
        self.total_origins = total_origins
        self.started_at_utc = utc_timestamp()
        self.started_monotonic = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record(
        self,
        status: str,
        *,
        partition: str | None = None,
        partition_completed: int = 0,
        partition_total: int = 0,
        overall_completed: int = 0,
        target_date: str | None = None,
        refit: bool | None = None,
        detail: str | None = None,
    ) -> None:
        elapsed_seconds = max(0.0, time.monotonic() - self.started_monotonic)
        rate = overall_completed / elapsed_seconds if overall_completed > 0 and elapsed_seconds > 0 else None
        eta_seconds = (self.total_origins - overall_completed) / rate if rate and overall_completed < self.total_origins else 0.0
        estimated_finish = (
            datetime.fromtimestamp(time.time() + eta_seconds, tz=timezone.utc).isoformat()
            if rate is not None
            else None
        )
        self._write(
            {
                "timestamp_utc": utc_timestamp(),
                "run_id": self.run_id,
                "company": self.company,
                "pid": os.getpid(),
                "status": status,
                "partition": partition,
                "partition_completed": partition_completed,
                "partition_total": partition_total,
                "overall_completed": overall_completed,
                "overall_total": self.total_origins,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "origins_per_second": round(rate, 6) if rate is not None else None,
                "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
                "estimated_finish_utc": estimated_finish,
                "target_date": target_date,
                "refit": refit,
                "detail": detail,
            }
        )

    def started(self, *, calibration_total: int, test_total: int) -> None:
        self.record(
            "started",
            detail=f"{calibration_total} calibration + {test_total} final-test one-step origins",
        )

    def failed(self, detail: str, overall_completed: int = 0) -> None:
        self.record("failed", overall_completed=overall_completed, detail=detail)

