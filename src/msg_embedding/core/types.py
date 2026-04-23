from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

NdArrayFloat = npt.NDArray[np.floating[Any]]
NdArrayComplex = npt.NDArray[np.complexfloating[Any, Any]]
NdArrayInt = npt.NDArray[np.integer[Any]]
NdArray = npt.NDArray[Any]


class LinkType(str, Enum):
    UL = "UL"
    DL = "DL"


class ChannelEstMode(str, Enum):
    LS = "LS"
    MMSE = "MMSE"
    IDEAL = "IDEAL"


class SourceType(str, Enum):
    INTERNAL_SIM = "internal_sim"
    SIONNA_RT = "sionna_rt"
    QUADRIGA_REAL = "quadriga_real"
    FIELD = "field"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    SIMULATE = "simulate"
    CONVERT = "convert"
    BRIDGE = "bridge"
    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"
    EXPORT = "export"
    REPORT = "report"
