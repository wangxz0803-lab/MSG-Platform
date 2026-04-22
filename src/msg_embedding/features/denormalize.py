from __future__ import annotations

from typing import Any

import torch

from msg_embedding.features.normalizer import ProtocolNormalizer

_NORM_EPS: float = 1e-8


def denormalize_reconstruction(
    recon_feat: dict[str, torch.Tensor],
    normalizer: ProtocolNormalizer,
    norm_stats: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    physical_feat: dict[str, torch.Tensor] = {}

    if "pdp_crop" in recon_feat:
        physical_feat["pdp_crop"] = (recon_feat["pdp_crop"] + 1.0) / 2.0

    for k in [
        "srs1",
        "srs2",
        "srs3",
        "srs4",
        "pmi1",
        "pmi2",
        "pmi3",
        "pmi4",
        "dft1",
        "dft2",
        "dft3",
        "dft4",
    ]:
        if k not in recon_feat:
            continue
        r = recon_feat[k]
        if norm_stats and k in norm_stats:
            rms = norm_stats[k]["rms"].unsqueeze(-1)
            physical_feat[k] = r * (rms + _NORM_EPS)
        else:
            physical_feat[k] = r

    for k in ["rsrp_srs", "rsrp_cb"]:
        if k not in recon_feat:
            continue
        r_db = normalizer.denormalize(recon_feat[k], k)
        physical_feat[k] = torch.clamp(r_db, -160.0, -60.0)

    if "cell_rsrp" in recon_feat:
        r_db = normalizer.denormalize(recon_feat["cell_rsrp"], "cell_rsrp")
        physical_feat["cell_rsrp"] = torch.clamp(r_db, -160.0, -60.0)

    return physical_feat
