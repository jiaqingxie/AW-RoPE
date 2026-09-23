"""AW-RoPE inside the unmodified-width official Performer attention shell."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from performer_pytorch import SelfAttention

from .aw_rope import AnalyticWalkRoPE, build_reverse_edge_index


def _cfg_value(config, name: str, default):
    return getattr(config, name, default) if config is not None else default


class OfficialPerformerAWRoPE(SelfAttention):
    """Official ``SelfAttention`` with only a sparse AW Q/K transport added.

    Calling ``super().__init__`` preserves the paper baseline's projections,
    default ``dim_head=64``, FAVOR+ implementation, output projection and
    attention dropout.  AW acts after the original Q/K projections and before
    dense batching; every non-positional component remains the paper baseline.
    """

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dropout: float,
        aw_cfg,
        method: str = "aw",
    ) -> None:
        super().__init__(
            dim=dim,
            heads=heads,
            dropout=dropout,
            causal=False,
        )
        if method not in {"aw", "aw-nb", "aw-nb-ms"}:
            raise ValueError(f"official Performer AW received unsupported method {method!r}")
        self.model_dim = int(dim)
        self.inner_dim = int(self.to_q.out_features)
        self.method = method
        self.aw_rezero = bool(_cfg_value(aw_cfg, "rezero", False))
        if self.inner_dim != heads * 64:
            raise RuntimeError(
                f"paper Performer width drifted: expected {heads}x64, got {self.inner_dim}"
            )
        # B/ReZero must be an exact paired-NoPE initialization.  Building the
        # AW-only module must therefore not advance the RNG seen by later
        # layers in the shared Performer backbone.
        paired_backbone_rng = (
            torch.random.get_rng_state() if self.aw_rezero else None
        )
        max_displacement = _cfg_value(aw_cfg, "max_displacement", math.pi)
        if max_displacement is not None and float(max_displacement) <= 0:
            max_displacement = None
        self.aw_rope = AnalyticWalkRoPE(
            dim=self.inner_dim,
            field_node_dim=self.model_dim,
            method=method,
            num_steps=int(_cfg_value(aw_cfg, "num_steps", 8)),
            initial_z=float(_cfg_value(aw_cfg, "z", 0.8)),
            z_values=tuple(_cfg_value(aw_cfg, "z_values", (0.2, 0.4, 0.6, 0.8))),
            field_hidden_dim=int(_cfg_value(aw_cfg, "field_hidden_dim", 32)),
            max_displacement=max_displacement,
            frequency_base=float(_cfg_value(aw_cfg, "frequency_base", 10_000.0)),
            learnable_frequencies=bool(_cfg_value(aw_cfg, "learnable_frequencies", True)),
            learnable_z=bool(_cfg_value(aw_cfg, "learnable_z", True)),
            normalize_resolvent=bool(_cfg_value(aw_cfg, "normalize_resolvent", False)),
            residual_mix=float(_cfg_value(aw_cfg, "residual_mix", 1.0)),
            preserve_input_norm=bool(_cfg_value(aw_cfg, "preserve_input_norm", False)),
            norm_group_size=int(_cfg_value(aw_cfg, "norm_group_size", 0)),
            field_type=str(_cfg_value(aw_cfg, "field_type", "local-antisymmetric")),
            position_dim=int(_cfg_value(aw_cfg, "position_dim", 3)),
        )
        if paired_backbone_rng is not None:
            torch.random.set_rng_state(paired_backbone_rng)
            self.aw_rezero_gate = nn.Parameter(torch.zeros(()))

    @property
    def paper_contract(self) -> dict[str, object]:
        return {
            "attention_class": "SelfAttention+AW",
            "model_dim": self.model_dim,
            "heads": int(self.heads),
            "dim_head": self.inner_dim // int(self.heads),
            "inner_dim": self.inner_dim,
            "qkv_bias": self.to_q.bias is not None,
            "attention_dropout": float(self.dropout.p),
            "performer_kernel": "softmax-favor+",
            "aw_method": self.method,
        }

    def forward(self, batch) -> torch.Tensor:
        if not hasattr(batch, "edge_index") or batch.edge_index is None:
            raise ValueError("official Performer AW requires batch.edge_index")
        node_features = batch.x
        query = self.to_q(node_features)
        key = self.to_k(node_features)
        value = self.to_v(node_features)
        reverse_edge = getattr(batch, "_aw_reverse_edge", None)
        if reverse_edge is None and self.method in {"aw-nb", "aw-nb-ms"}:
            reverse_edge = build_reverse_edge_index(batch.edge_index, node_features.shape[0])
            batch._aw_reverse_edge = reverse_edge
        transported_query, transported_key = self.aw_rope(
            query,
            key,
            node_features,
            batch.edge_index,
            edge_weight=getattr(batch, "edge_weight", None),
            reverse_edge=reverse_edge,
            positions=getattr(batch, "pos", None),
            precomputed_displacement=getattr(
                batch, "aw_static_edge_displacement", None
            ),
        )
        if self.aw_rezero:
            gate = self.aw_rezero_gate.to(dtype=query.dtype)
            query = query + gate * (transported_query - query)
            key = key + gate * (transported_key - key)
        else:
            query, key = transported_query, transported_key
        query, real_nodes = to_dense_batch(query, batch.batch)
        key, _ = to_dense_batch(key, batch.batch)
        value, _ = to_dense_batch(value, batch.batch)
        batch_size, nodes, _ = query.shape
        query = query.view(batch_size, nodes, self.heads, -1).transpose(1, 2)
        key = key.view(batch_size, nodes, self.heads, -1).transpose(1, 2)
        value = value.view(batch_size, nodes, self.heads, -1).transpose(1, 2)

        # Preserve performer-pytorch SelfAttention's exact padding contract:
        # only V is masked before FAVOR+; Q/K padding comes from bias-free zero
        # projections and participates in the same denominator as the paper.
        value = value.masked_fill(~real_nodes[:, None, :, None], 0.0)
        output = self.fast_attention(query, key, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, nodes, self.inner_dim)
        output = self.dropout(self.to_out(output))
        return output[real_nodes]
