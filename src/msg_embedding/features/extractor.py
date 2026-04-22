from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from msg_embedding.features.normalizer import ProtocolNormalizer, db2lin

_NORM_EPS: float = 1e-8

_DEFAULTS: dict[str, Any] = {
    "tx_ant_num_max": 64,
    "cell_rsrp_dim": 16,
    "token_dim": 128,
    "seq_len": 16,
}


class FeatureExtractor(nn.Module):
    """Multi-modal feature extractor with protocol normalization and quality gating.

    Token layout (16 slots):
        [0]=pdp, [1-4]=srs1-4, [5-8]=pmi1-4, [9-12]=dft1-4,
        [13]=rsrp_srs, [14]=rsrp_cb, [15]=cell_rsrp
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        _c = cfg or {}

        self._tx_ant_num_max: int = _c.get("tx_ant_num_max", _DEFAULTS["tx_ant_num_max"])
        self._cell_rsrp_dim: int = _c.get("cell_rsrp_dim", _DEFAULTS["cell_rsrp_dim"])
        self._token_dim: int = _c.get("token_dim", _DEFAULTS["token_dim"])
        self._seq_len: int = _c.get("seq_len", _DEFAULTS["seq_len"])

        self.normalizer = ProtocolNormalizer(method="minmax")

        self.complex_embed = nn.Linear(2 * self._tx_ant_num_max, self._token_dim)
        self.pdp_embed = nn.Linear(64, self._token_dim)
        self.real_embed = nn.Linear(self._tx_ant_num_max, self._token_dim)
        self.cell_embed = nn.Linear(self._cell_rsrp_dim, self._token_dim)

        self.quality_gate_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.energy_gate_proj = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 4),
            nn.Sigmoid(),
        )
        self.pmi_gate_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _infer_batch_info(feat: dict[str, torch.Tensor]) -> tuple[int, torch.device]:
        probe = [
            "rsrp_srs",
            "rsrp_cb",
            "cqi",
            "srs_sinr",
            "pdp_crop",
            "srs1",
            "pmi1",
            "dft1",
            "cell_rsrp",
        ]
        for k in probe:
            if k in feat:
                t = feat[k]
                return t.shape[0], t.device
        raise ValueError("FeatureExtractor: feat dict is empty or has no recognized fields")

    def normalize_features(
        self, feat: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor]:
        norm_feat: dict[str, torch.Tensor] = {}
        norm_stats: dict[str, Any] = {}
        field_mask: dict[str, bool] = {}

        B, device = self._infer_batch_info(feat)

        if "cqi" in feat:
            norm_feat["cqi"] = self.normalizer.normalize(feat["cqi"].float(), "cqi")
            field_mask["cqi"] = True
        else:
            norm_feat["cqi"] = torch.zeros(B, device=device)
            field_mask["cqi"] = False

        if "srs_sinr" in feat:
            sinr_db = feat["srs_sinr"]
            norm_feat["srs_sinr"] = self.normalizer.normalize(sinr_db, "srs_sinr")
            sinr_linear = db2lin(sinr_db)
            field_mask["srs_sinr"] = True
        else:
            norm_feat["srs_sinr"] = torch.zeros(B, device=device)
            sinr_linear = torch.ones(B, device=device)
            field_mask["srs_sinr"] = False

        if "srs_cb_sinr" in feat:
            cb_sinr_db = feat["srs_cb_sinr"]
            norm_feat["srs_cb_sinr"] = self.normalizer.normalize(cb_sinr_db, "srs_cb_sinr")
            field_mask["srs_cb_sinr"] = True
        else:
            field_mask["srs_cb_sinr"] = False

        if "rsrp_srs" in feat:
            norm_feat["rsrp_srs"] = self.normalizer.normalize(feat["rsrp_srs"], "rsrp_srs")
            field_mask["rsrp_srs"] = True
        else:
            field_mask["rsrp_srs"] = False

        if "rsrp_cb" in feat:
            norm_feat["rsrp_cb"] = self.normalizer.normalize(feat["rsrp_cb"], "rsrp_cb")
            field_mask["rsrp_cb"] = True
        else:
            field_mask["rsrp_cb"] = False

        for k in ["srs_w1", "srs_w2", "srs_w3", "srs_w4"]:
            if k in feat:
                norm_feat[k] = self.normalizer.normalize(feat[k], k)

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
            if k not in feat:
                continue
            v = feat[k]
            power = torch.abs(v) ** 2
            rms = torch.sqrt(power.mean(dim=-1, keepdim=True) + _NORM_EPS)
            v_norm = v / (rms + _NORM_EPS)
            norm_feat[k] = v_norm
            norm_stats[k] = {"rms": rms.squeeze(-1)}

        if "pdp_crop" in feat:
            pdp = torch.clamp(feat["pdp_crop"], 0.0, 1.0)
            norm_feat["pdp_crop"] = pdp * 2.0 - 1.0

        if "cell_rsrp" in feat:
            cell_rsrp = feat["cell_rsrp"]
            if cell_rsrp.shape[-1] < self._cell_rsrp_dim:
                pad = torch.full(
                    (cell_rsrp.shape[0], self._cell_rsrp_dim - cell_rsrp.shape[-1]),
                    -160.0,
                    device=cell_rsrp.device,
                    dtype=cell_rsrp.dtype,
                )
                cell_rsrp = torch.cat([cell_rsrp, pad], dim=-1)
            else:
                cell_rsrp = cell_rsrp[:, : self._cell_rsrp_dim]
            norm_feat["cell_rsrp"] = self.normalizer.normalize(cell_rsrp, "cell_rsrp")
            field_mask["cell_rsrp"] = True
        else:
            field_mask["cell_rsrp"] = False

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
            "pdp_crop",
            "srs_w1",
            "srs_w2",
            "srs_w3",
            "srs_w4",
        ]:
            field_mask[k] = k in feat

        norm_stats["field_mask"] = field_mask
        return norm_feat, norm_stats, sinr_linear

    def forward(self, feat: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        B, _ = self._infer_batch_info(feat)
        norm_feat, norm_stats, sinr_linear = self.normalize_features(feat)

        tokens: list[torch.Tensor] = []
        token_present: list[bool] = []
        _dev = norm_feat["cqi"].device
        zero_tok = torch.zeros(B, 1, self._token_dim, device=_dev)

        def _emit(tok: torch.Tensor, present: bool) -> None:
            tokens.append(tok)
            token_present.append(present)

        sinr_robust = torch.tanh(norm_feat["srs_sinr"] * 2.5).unsqueeze(-1)
        quality_gate = self.quality_gate_proj(sinr_robust)
        quality_gate = quality_gate * 0.3 + 0.7

        energy_sc = torch.stack(
            [
                norm_feat.get("srs_w1", torch.zeros(B, device=_dev)),
                norm_feat.get("srs_w2", torch.zeros(B, device=_dev)),
                norm_feat.get("srs_w3", torch.zeros(B, device=_dev)),
                norm_feat.get("srs_w4", torch.zeros(B, device=_dev)),
            ],
            dim=-1,
        )
        energy_gates = self.energy_gate_proj(energy_sc)
        energy_gates = energy_gates * 0.3 + 0.7
        srs_gates = torch.minimum(quality_gate.expand(-1, 4), energy_gates)

        cqi_robust = torch.tanh(norm_feat["cqi"] * 2.0).unsqueeze(-1).float()
        pmi_gate = self.pmi_gate_proj(cqi_robust)
        pmi_gate = pmi_gate * 0.3 + 0.7

        # PDP token
        if "pdp_crop" in norm_feat:
            weight = (sinr_linear - 0.01) / (100.0 - 0.01)
            weight = torch.clamp(weight, 0.0, 1.0)
            pdp_weighted = norm_feat["pdp_crop"] * weight.view(B, 1)
            _emit(self.pdp_embed(pdp_weighted).unsqueeze(1), True)
        else:
            _emit(zero_tok, False)

        # Complex feature tokens
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
            if k not in norm_feat:
                _emit(zero_tok, False)
                continue
            v = norm_feat[k]

            if k.startswith("srs"):
                flow_idx = int(k[-1]) - 1
                gate = srs_gates[:, flow_idx].view(B, 1)
                v_gated = v * gate
            elif k.startswith("pmi"):
                gate = pmi_gate.view(B, 1)
                v_gated = v * gate
            else:
                v_gated = v

            if v_gated.shape[-1] < self._tx_ant_num_max:
                pad_size = self._tx_ant_num_max - v_gated.shape[-1]
                v_padded = F.pad(v_gated, (0, pad_size), value=0.0)
            else:
                v_padded = v_gated[:, : self._tx_ant_num_max]

            cat = torch.cat([v_padded.real, v_padded.imag], dim=-1)
            _emit(self.complex_embed(cat).unsqueeze(1), True)

        # RSRP tokens
        srs_sinr_gate = torch.tanh(norm_feat["srs_sinr"] * 2.5)
        srs_sinr_gate = srs_sinr_gate * 0.3 + 0.7

        if "srs_cb_sinr" in norm_feat:
            cb_sinr_gate = torch.tanh(norm_feat["srs_cb_sinr"] * 2.5)
            cb_sinr_gate = cb_sinr_gate * 0.3 + 0.7
        else:
            cb_sinr_gate = srs_sinr_gate

        rsrp_gates: dict[str, torch.Tensor] = {}
        for k in ["rsrp_srs", "rsrp_cb"]:
            if k not in norm_feat:
                _emit(zero_tok, False)
                continue
            v = norm_feat[k]
            gate = (srs_sinr_gate if k == "rsrp_srs" else cb_sinr_gate).view(B, 1)
            rsrp_gates[k] = gate.squeeze(-1)
            v_gated = v * gate

            if v_gated.shape[-1] < self._tx_ant_num_max:
                v_gated = F.pad(v_gated, (0, self._tx_ant_num_max - v_gated.shape[-1]), value=0.0)
            else:
                v_gated = v_gated[:, : self._tx_ant_num_max]
            _emit(self.real_embed(v_gated).unsqueeze(1), True)

        norm_stats["rsrp_gates"] = rsrp_gates

        # Cell RSRP token
        if "cell_rsrp" in norm_feat:
            _emit(self.cell_embed(norm_feat["cell_rsrp"]).unsqueeze(1), True)
        else:
            _emit(zero_tok, False)

        # Assemble fixed-length sequence
        out = torch.cat(tokens, dim=1)
        token_mask = (
            torch.tensor([not p for p in token_present], dtype=torch.bool, device=_dev)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )

        if out.shape[1] > self._seq_len:
            out = out[:, : self._seq_len]
            token_mask = token_mask[:, : self._seq_len]
        elif out.shape[1] < self._seq_len:
            pad_n = self._seq_len - out.shape[1]
            pad = torch.zeros(B, pad_n, self._token_dim, device=out.device)
            out = torch.cat([out, pad], dim=1)
            tail_mask = torch.ones(B, pad_n, dtype=torch.bool, device=_dev)
            token_mask = torch.cat([token_mask, tail_mask], dim=1)

        norm_stats["token_mask"] = token_mask

        return out, norm_stats
