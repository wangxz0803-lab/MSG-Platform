"""3GPP protocol specification tables.

PROTOCOL_SPEC is kept in Python (rather than YAML) because it has deeply nested
heterogeneous schema (mixed discrete/continuous fields, tuple ranges, optional
process/linear_range keys) and is consumed programmatically by the normalization
and feature-extraction pipelines.

Source specs:
    * 3GPP TS 38.214 -- Physical layer procedures for data
    * 3GPP TS 38.133 -- Requirements for support of radio resource management
    * 3GPP TS 38.213 -- Physical layer procedures for control (TA definition)
"""

from __future__ import annotations

from typing import Any

PROTOCOL_SPEC: dict[str, dict[str, Any]] = {
    "cqi": {"min": 0, "max": 15, "unit": "index", "type": "discrete"},
    "srs_sinr": {
        "min": -20.0,
        "max": 20.0,
        "unit": "dB",
        "process": "linear",
        "linear_range": (0.01, 100.0),
    },
    "rsrp": {"min": -160.0, "max": -60.0, "unit": "dBm", "process": "db"},
    "rsrp_srs": {"min": -160.0, "max": -60.0, "unit": "dBm", "process": "db"},
    "rsrp_cb": {"min": -160.0, "max": -60.0, "unit": "dBm", "process": "db"},
    "cell_rsrp": {"min": -160.0, "max": -60.0, "unit": "dBm", "process": "db"},
    "path_loss": {"min": 40.0, "max": 160.0, "unit": "dB", "process": "db"},
    "tx_ant_num": {"min": 4, "max": 256, "unit": "count", "type": "discrete"},
    "rx_ant_num": {"min": 1, "max": 8, "unit": "count", "type": "discrete"},
    "ue_tx_ant_num": {"min": 1, "max": 8, "unit": "count", "type": "discrete"},
    "srs_hopping_rb": {"min": 0, "max": 272, "unit": "RB", "type": "discrete"},
    "pdp": {"min": 0.0, "max": 1.0, "unit": "linear", "process": "linear"},
    "srs_w1": {"min": 0.0, "max": 1.0, "unit": "linear", "process": "linear"},
    "srs_w2": {"min": 0.0, "max": 1.0, "unit": "linear", "process": "linear"},
    "srs_w3": {"min": 0.0, "max": 1.0, "unit": "linear", "process": "linear"},
    "srs_w4": {"min": 0.0, "max": 1.0, "unit": "linear", "process": "linear"},
    "srs_cb_sinr": {
        "min": -20.0,
        "max": 20.0,
        "unit": "dB",
        "process": "linear",
        "linear_range": (0.01, 100.0),
    },
    "ta": {"min": -128.0, "max": 128.0, "unit": "1/16Ts", "type": "discrete"},
}

__all__ = ["PROTOCOL_SPEC"]
