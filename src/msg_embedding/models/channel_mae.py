from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from msg_embedding.models.adapters import LatentAdapter

_DEFAULTS: dict[str, Any] = {
    "token_dim": 128,
    "seq_len": 16,
    "latent_dim": 256,
    "n_heads": 8,
    "encoder_layers": 4,
    "decoder_layers": 2,
    "ff_dim": 512,
    "dropout_p": 0.1,
    "tx_ant_num_max": 64,
    "cell_rsrp_dim": 16,
    "latent_out_dim": 16,
    "use_snr_condition": True,
    "perturb_gauss": 1e-3,
    "perturb_shift": 0.02,
}


class ChannelMAE(nn.Module):
    """Masked Autoencoder for 5G NR channel representation learning.

    Architecture: Transformer encoder -> MLP bottleneck -> Transformer decoder
    with energy-weighted pooling into a 16-D L2-normalized latent space.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg or {}
        _c = self.cfg

        self._token_dim: int = _c.get("token_dim", _DEFAULTS["token_dim"])
        self._seq_len: int = _c.get("seq_len", _DEFAULTS["seq_len"])
        self._latent_dim: int = _c.get("latent_dim", _DEFAULTS["latent_dim"])
        self._n_heads: int = _c.get("n_heads", _DEFAULTS["n_heads"])
        self._encoder_layers: int = _c.get("encoder_layers", _DEFAULTS["encoder_layers"])
        self._decoder_layers: int = _c.get("decoder_layers", _DEFAULTS["decoder_layers"])
        self._ff_dim: int = _c.get("ff_dim", _DEFAULTS["ff_dim"])
        self._dropout_p: float = _c.get("dropout_p", _DEFAULTS["dropout_p"])
        self._tx_ant_num_max: int = _c.get("tx_ant_num_max", _DEFAULTS["tx_ant_num_max"])
        self._cell_rsrp_dim: int = _c.get("cell_rsrp_dim", _DEFAULTS["cell_rsrp_dim"])
        self._latent_out_dim: int = _c.get("latent_out_dim", _DEFAULTS["latent_out_dim"])
        self._use_snr_condition: bool = _c.get("use_snr_condition", _DEFAULTS["use_snr_condition"])
        self._perturb_gauss: float = _c.get("perturb_gauss", _DEFAULTS["perturb_gauss"])
        self._perturb_shift: float = _c.get("perturb_shift", _DEFAULTS["perturb_shift"])

        self.input_proj = nn.Linear(self._token_dim, self._token_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, self._seq_len, self._token_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self._token_dim))

        self.snr_encoder: nn.Module | None = (
            nn.Sequential(
                nn.Linear(1, 64),
                nn.GELU(),
                nn.Linear(64, self._token_dim),
            )
            if self._use_snr_condition
            else None
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self._token_dim,
                nhead=self._n_heads,
                dim_feedforward=self._ff_dim,
                dropout=self._dropout_p,
                batch_first=True,
            ),
            num_layers=self._encoder_layers,
        )

        self.mlp = nn.Sequential(
            nn.Linear(self._token_dim, self._latent_dim),
            nn.GELU(),
            nn.Dropout(self._dropout_p),
            nn.Linear(self._latent_dim, self._token_dim),
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                self._token_dim,
                nhead=self._n_heads,
                dim_feedforward=self._ff_dim,
                dropout=self._dropout_p,
                batch_first=True,
            ),
            num_layers=self._decoder_layers,
        )

        self.pdp_embed_back = nn.Linear(self._token_dim, 64)
        self.complex_embed_back = nn.Linear(self._token_dim, 2 * self._tx_ant_num_max)
        self.real_embed_back = nn.Linear(self._token_dim, self._tx_ant_num_max)
        self.cell_embed_back = nn.Linear(self._token_dim, self._cell_rsrp_dim)

        self.latent_proj = nn.Sequential(
            nn.LayerNorm(self._token_dim),
            nn.Linear(self._token_dim, self._token_dim // 2),
            nn.GELU(),
            nn.Dropout(self._dropout_p),
            nn.LayerNorm(self._token_dim // 2),
            nn.Linear(self._token_dim // 2, self._latent_out_dim),
        )
        self.proj_shortcut = nn.Linear(self._token_dim, self._latent_out_dim, bias=False)

        self.latent_adapter = LatentAdapter(self._latent_out_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(
        self,
        tokens: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        perturb_type: str | None = None,
        mask_ratio: float | None = None,
    ) -> torch.Tensor:
        B, L, _D = tokens.shape
        x = self.input_proj(tokens) + self.pos_emb

        if self.snr_encoder is not None and snr_db is not None:
            snr_emb = self.snr_encoder(snr_db.unsqueeze(-1).float())
            x = x + snr_emb.unsqueeze(1)

        if self.training and perturb_type is not None:
            if perturb_type == "gauss":
                x = x + torch.randn_like(x) * self._perturb_gauss
            elif perturb_type == "shift":
                x = x + torch.randn_like(x) * self._perturb_shift

        if mask_ratio is not None and mask_ratio > 0:
            num_unmasked = int(L * (1 - mask_ratio))
            mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
            for i in range(B):
                unmasked_idx = torch.randperm(L, device=x.device)[:num_unmasked]
                mask[i, unmasked_idx] = False
            x = x * (~mask).unsqueeze(-1)

        return self.encoder(x)

    def pool_and_project(self, enc: torch.Tensor, use_adapter: bool = False) -> torch.Tensor:
        mlp_out = self.mlp(enc)
        norm_energy = torch.norm(mlp_out, dim=-1, keepdim=True)
        weight = torch.softmax(norm_energy, dim=1)
        pooled = torch.sum(mlp_out * weight, dim=1)

        proj = self.latent_proj(pooled) + self.proj_shortcut(pooled)
        if use_adapter:
            proj = self.latent_adapter(proj)
        return F.normalize(proj, dim=-1)

    def get_latent(
        self,
        tokens: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        is_infer: bool = False,
        perturb_type: str | None = None,
        use_adapter: bool = False,
    ) -> torch.Tensor:
        enc = self.encode(tokens, snr_db, perturb_type=(None if is_infer else perturb_type))
        return self.pool_and_project(enc, use_adapter=use_adapter)

    def forward(
        self,
        tokens: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        mask_ratio: float | None = None,
        perturb_type: str | None = None,
    ) -> torch.Tensor:
        """MAE reconstruction forward: encoder -> bottleneck -> decoder."""
        B = tokens.shape[0]
        enc = self.encode(tokens, snr_db, perturb_type=perturb_type, mask_ratio=mask_ratio)
        mem = self.mlp(enc)
        query = self.pos_emb.expand(B, -1, -1)
        return self.decoder(query, mem)

    def reconstruct(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode per-token features from decoder output.

        Token layout: [0]=pdp, [1-4]=srs, [5-8]=pmi, [9-12]=dft,
                      [13]=rsrp_srs, [14]=rsrp_cb, [15]=cell_rsrp
        """
        f: dict[str, torch.Tensor] = {}
        f["pdp_crop"] = torch.tanh(self.pdp_embed_back(tokens[:, 0]))

        feat_list = [
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
        ]
        idx = 1
        for feat in feat_list:
            if idx >= tokens.shape[1]:
                break
            c = torch.tanh(self.complex_embed_back(tokens[:, idx]))
            f[feat] = torch.complex(c[..., : self._tx_ant_num_max], c[..., self._tx_ant_num_max :])
            idx += 1

        for feat in ("rsrp_srs", "rsrp_cb"):
            if idx >= tokens.shape[1]:
                break
            f[feat] = torch.tanh(self.real_embed_back(tokens[:, idx]))
            idx += 1

        if idx < tokens.shape[1]:
            f["cell_rsrp"] = torch.tanh(self.cell_embed_back(tokens[:, idx]))

        return f
