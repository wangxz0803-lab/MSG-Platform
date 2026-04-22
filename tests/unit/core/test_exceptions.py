from __future__ import annotations

import pytest

from msg_embedding.core.exceptions import (
    ConfigError,
    DataSourceError,
    FeatureExtractionError,
    InferenceError,
    JobError,
    MATLABError,
    ModelError,
    MSGError,
    PlatformError,
    TrainingError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_msg_error(self):
        for exc_cls in (
            ConfigError,
            DataSourceError,
            MATLABError,
            FeatureExtractionError,
            ModelError,
            TrainingError,
            InferenceError,
            PlatformError,
            JobError,
        ):
            assert issubclass(exc_cls, MSGError)

    def test_matlab_error_is_data_source_error(self):
        assert issubclass(MATLABError, DataSourceError)

    def test_job_error_is_platform_error(self):
        assert issubclass(JobError, PlatformError)

    def test_raise_and_catch_by_base(self):
        with pytest.raises(MSGError, match="test message"):
            raise ConfigError("test message")

    def test_raise_matlab_caught_as_data_source(self):
        with pytest.raises(DataSourceError):
            raise MATLABError("MATLAB timed out")

    def test_exception_message_preserved(self):
        err = TrainingError("NaN loss at epoch 5")
        assert str(err) == "NaN loss at epoch 5"
