"""Root conftest for platform tests."""

from __future__ import annotations

import sys

sys.modules.setdefault("platform.conftest", sys.modules[__name__])
