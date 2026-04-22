from __future__ import annotations

from typing import Any

import torch

from msg_embedding.core.protocol_spec import PROTOCOL_SPEC

_NORM_EPS: float = 1e-8


def db2lin(x: torch.Tensor) -> torch.Tensor:
    return 10 ** (x / 10.0)


def lin2db(x: torch.Tensor, eps: float = _NORM_EPS) -> torch.Tensor:
    return 10 * torch.log10(x + eps)


class ProtocolNormalizer:
    """Min-max normalizer based on 3GPP protocol value ranges.

    Supports dB-domain and linear-domain normalization, mapping physical
    values to the [-1, 1] range and back.
    """

    def __init__(self, method: str = "minmax") -> None:
        self.method = method
        self.spec: dict[str, dict[str, Any]] = PROTOCOL_SPEC

    def _get_linear_range(self, key: str) -> tuple[float, float]:
        spec = self.spec.get(key, {})
        process = spec.get("process", "db")

        if process == "linear" and "linear_range" in spec:
            return spec["linear_range"]
        elif process == "linear":
            db_min, db_max = spec["min"], spec["max"]
            return (10 ** (db_min / 10.0), 10 ** (db_max / 10.0))
        else:
            return (spec["min"], spec["max"])

    def normalize(self, value: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.spec:
            return value

        spec = self.spec[key]
        process = spec.get("process", "db")

        if process == "linear":
            linear_value = db2lin(value) if spec["unit"] == "dB" else value

            lin_min, lin_max = self._get_linear_range(key)
            linear_value = torch.clamp(linear_value, lin_min, lin_max)

            if self.method == "minmax":
                normalized = 2.0 * (linear_value - lin_min) / (lin_max - lin_min + _NORM_EPS) - 1.0
                return torch.clamp(normalized, -1.0, 1.0)

        else:
            min_val, max_val = spec["min"], spec["max"]

            if self.method == "minmax":
                normalized = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
                return torch.clamp(normalized, -1.0, 1.0)

        return value

    def denormalize(self, normalized: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.spec:
            return normalized

        spec = self.spec[key]
        process = spec.get("process", "db")

        if process == "linear":
            lin_min, lin_max = self._get_linear_range(key)

            if self.method == "minmax":
                linear_value = (normalized + 1.0) / 2.0 * (lin_max - lin_min) + lin_min

                if spec["unit"] == "dB":
                    return lin2db(linear_value)
                return linear_value

        else:
            min_val, max_val = spec["min"], spec["max"]

            if self.method == "minmax":
                physical = (normalized + 1.0) / 2.0 * (max_val - min_val) + min_val

                if spec.get("type") == "discrete":
                    physical = torch.round(physical)

                return physical

        return normalized
