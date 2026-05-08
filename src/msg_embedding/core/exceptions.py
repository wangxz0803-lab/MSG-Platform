from __future__ import annotations


class MSGError(Exception):
    """Base for all ChannelHub application errors."""


class ConfigError(MSGError):
    """Configuration loading or validation failure."""


class DataSourceError(MSGError):
    """Data source initialization or iteration failure."""


class MATLABError(DataSourceError):
    """MATLAB subprocess failure (timeout, bad exit code, missing exe)."""


class FeatureExtractionError(MSGError):
    """Feature extraction pipeline failure (SVD, normalization, etc.)."""


class ModelError(MSGError):
    """Model loading, checkpoint, or forward-pass failure."""


class TrainingError(MSGError):
    """Training loop failure (NaN loss, OOM, checkpoint save)."""


class InferenceError(MSGError):
    """Inference or export failure."""


class PlatformError(MSGError):
    """Platform infrastructure error (DB, queue, dispatch)."""


class JobError(PlatformError):
    """Job lifecycle error (dispatch, cancel, timeout)."""
