from __future__ import annotations

import pytest

from msg_embedding.core.config import MSGSettings, reset_settings


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path):
    """Provide each test with a fresh MSGSettings pointing at a temp directory."""
    settings = MSGSettings(
        repo_root=tmp_path,
        db_url=f"sqlite:///{tmp_path / 'test.db'}",
        redis_url="redis://localhost:6379/15",
        log_level="DEBUG",
        log_format="console",
    )
    reset_settings(settings)
    yield settings
    reset_settings(None)
