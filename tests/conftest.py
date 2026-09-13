"""Pytest compatibility for the project's historical ``code`` package name.

Python 3.13 imports the standard-library :mod:`code` module through pytest's
debugging plugin before test collection.  Expose the project directory as a
submodule search location on that already-loaded module so imports such as
``code.models`` remain usable without replacing standard-library attributes
required by :mod:`pdb`.
"""

from __future__ import annotations

import code as stdlib_code
from pathlib import Path


PROJECT_CODE = Path(__file__).resolve().parents[1] / "code"
search_locations = list(getattr(stdlib_code, "__path__", []))
if str(PROJECT_CODE) not in search_locations:
    search_locations.append(str(PROJECT_CODE))
stdlib_code.__path__ = search_locations
if stdlib_code.__spec__ is not None:
    stdlib_code.__spec__.submodule_search_locations = search_locations

