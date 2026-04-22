"""5G NR reference-signal generators per 3GPP TS 38.211.

Public API:
    * Zadoff-Chu & low-PAPR base sequences — :mod:`zc`
    * Length-31 Gold pseudo-random sequence — :mod:`gold`
    * PSS / SSS / PBCH-DMRS                 — :mod:`ssb`
    * PDSCH / PUSCH DMRS                    — :mod:`dmrs`
    * SRS                                   — :mod:`srs`
    * NZP-CSI-RS                            — :mod:`csi_rs`
"""

from __future__ import annotations

from .csi_rs import (
    CDM_WF,
    CDM_WT,
    CSIRSPortInfo,
    csi_rs_port_info,
    csi_rs_sequence,
)
from .dmrs import (
    TYPE1_CDM_GROUPS,
    TYPE1_WF,
    TYPE1_WT,
    TYPE2_CDM_GROUPS,
    TYPE2_WF,
    TYPE2_WT,
    DMRSConfig,
    dmrs_re_map,
    dmrs_sequence,
)
from .gold import NC, pseudo_random
from .srs import (
    SRS_BW_TABLE,
    SRSBandwidthRow,
    srs_base_sequence,
    srs_cyclic_shift,
    srs_group_number,
    srs_sequence,
)
from .ssb import PSS_LEN, SSS_LEN, pbch_dmrs, pci, pss, sss
from .zc import (
    PHI_TABLE_M6,
    PHI_TABLE_M12,
    PHI_TABLE_M18,
    PHI_TABLE_M24,
    r_uv_long,
    r_uv_short,
    zadoff_chu,
)

__all__ = [
    # zc
    "zadoff_chu",
    "r_uv_long",
    "r_uv_short",
    "PHI_TABLE_M6",
    "PHI_TABLE_M12",
    "PHI_TABLE_M18",
    "PHI_TABLE_M24",
    # gold
    "pseudo_random",
    "NC",
    # ssb
    "pss",
    "sss",
    "pci",
    "pbch_dmrs",
    "PSS_LEN",
    "SSS_LEN",
    # dmrs
    "DMRSConfig",
    "dmrs_sequence",
    "dmrs_re_map",
    "TYPE1_CDM_GROUPS",
    "TYPE1_WF",
    "TYPE1_WT",
    "TYPE2_CDM_GROUPS",
    "TYPE2_WF",
    "TYPE2_WT",
    # srs
    "srs_sequence",
    "srs_base_sequence",
    "srs_cyclic_shift",
    "srs_group_number",
    "SRSBandwidthRow",
    "SRS_BW_TABLE",
    # csi_rs
    "csi_rs_sequence",
    "csi_rs_port_info",
    "CSIRSPortInfo",
    "CDM_WF",
    "CDM_WT",
]
