"""Shared pytest configuration.

Ensures ``src/`` is importable without requiring the package to be
installed, and provides asyncio test support via pytest-asyncio in
"auto" mode (see pyproject.toml's [tool.pytest.ini_options]).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
