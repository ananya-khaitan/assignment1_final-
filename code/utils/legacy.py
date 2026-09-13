"""Non-destructive archiving of pre-refactor generated artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

from config import OUT_DATA, OUT_FIG, OUT_ROOT, OUT_TAB


def snapshot_pre_causal_outputs() -> Path | None:
    """Copy old outputs once before the causal run is allowed to overwrite them."""
    destination = Path(OUT_ROOT) / "legacy" / "pre_causal_refactor"
    marker = destination / "SNAPSHOT_COMPLETE.txt"
    if marker.exists():
        return None
    destination.mkdir(parents=True, exist_ok=True)
    for source in (Path(OUT_DATA), Path(OUT_FIG), Path(OUT_TAB)):
        if source.exists():
            shutil.copytree(source, destination / source.name, dirs_exist_ok=True)
    marker.write_text(
        "Copied automatically before the first causal run. These files are archival and not current evidence.\n",
        encoding="utf-8",
    )
    return destination
