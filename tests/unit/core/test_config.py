from __future__ import annotations

from pathlib import Path

from msg_embedding.core.config import MSGSettings, get_settings, reset_settings


class TestMSGSettings:
    def test_default_settings(self, _isolated_settings):
        s = get_settings()
        assert isinstance(s.repo_root, Path)
        assert s.api_port == 8000
        assert s.log_level == "DEBUG"
        assert s.log_format == "console"

    def test_data_path_property(self, _isolated_settings):
        s = get_settings()
        assert s.data_path == s.repo_root / "data"

    def test_artifacts_path_property(self, _isolated_settings):
        s = get_settings()
        assert s.artifacts_path == s.repo_root / "artifacts"

    def test_ensure_dirs_creates_directories(self, _isolated_settings):
        s = get_settings()
        s.ensure_dirs()
        assert s.data_path.exists()
        assert s.artifacts_path.exists()
        assert s.reports_path.exists()
        assert s.bridge_out_path.exists()

    def test_reset_settings(self):
        custom = MSGSettings(repo_root=Path("/tmp/test"), api_port=9999)
        reset_settings(custom)
        assert get_settings().api_port == 9999
        reset_settings(None)

    def test_configs_path(self, _isolated_settings):
        s = get_settings()
        assert s.configs_path == s.repo_root / "configs"
