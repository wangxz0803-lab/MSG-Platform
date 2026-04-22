from __future__ import annotations

from msg_embedding.core.logging import get_logger, setup_logging


class TestLogging:
    def test_get_logger_returns_bound_logger(self):
        logger = get_logger("test_module")
        assert logger is not None

    def test_logger_can_log_without_error(self, capsys):
        logger = get_logger("test_module")
        logger.info("test_event", key="value")

    def test_setup_logging_idempotent(self):
        setup_logging(level="DEBUG", log_format="console")
        setup_logging(level="INFO", log_format="json")

    def test_logger_with_initial_context(self):
        logger = get_logger("test_module", request_id="abc-123")
        assert logger is not None
