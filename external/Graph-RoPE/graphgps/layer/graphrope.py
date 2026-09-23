import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.utils import to_dense_batch

from .aw_rope import (
    AnalyticWalkRoPE,
    LowRankComplexRandomFeatureAWRoPE,
    build_reverse_edge_index,
)
from .exact_holonomy import ExactHolonomyRoPE, LearnableExactHolonomyRoPE


_POSITIONAL_METHOD_ALIASES = {
    "none": "none",
    "nope": "none",
    "wire": "wire",
    "graphrope": "wire",
    "aw": "aw",
    "aw-rope": "aw",
    "aw_nb": "aw-nb",
    "aw-nb": "aw-nb",
    "nb": "aw-nb",
    "aw_nb_ms": "aw-nb-ms",
    "aw-nb-ms": "aw-nb-ms",
    "multiscale-nb": "aw-nb-ms",
    "lr-aw": "lr-aw",
    "aw-lr": "lr-aw",
    "low-rank-aw": "lr-aw",
    "lr-crf-aw-rope": "lr-aw",
    "exact": "exact-holonomy",
    "full-holonomy": "exact-holonomy",
    "exact-holonomy": "exact-holonomy",
    "holonomy-rope": "exact-holonomy",
}


def resolve_positional_method(method: str | None, enable: bool) -> str:
    """Resolve new method names while preserving the official enable flag.

    Existing WIRE configs do not contain ``method``.  For those configs,
    ``enable=True`` still selects WIRE and ``enable=False`` still selects
    NoPE, exactly as in the original implementation.
    """
    if method is None or str(method).strip() == "":
        return "wire" if enable else "none"
    key = str(method).strip().lower()
    if key not in _POSITIONAL_METHOD_ALIASES:
        choices = ", ".join(sorted(set(_POSITIONAL_METHOD_ALIASES.values())))
        raise ValueError(f"unknown positional method {method!r}; expected one of {choices}")
    return _POSITIONAL_METHOD_ALIASES[key]


def _cfg_value(config, name: str, default):
    return getattr(config, name, default) if config is not None else default


def init_omega_matrix(omega_matrix, init_omega: str, d: int, k: int):
    """
    Initialize an Omega matrix according to the specified strategy.
    
    Args:
        omega_matrix: The nn.Linear layer to initialize
        init_omega: Initialization strategy
        d: Total feature dimension 
        k: Rotational feature dimension
    """
    with torch.no_grad():
        match init_omega:
            case "zero":
                nn.init.zeros_(omega_matrix.weight)
            
            case "exponential":
                # Initialize with random frequencies
                rand_freqs = torch.rand(d//2, k, device=omega_matrix.weight.device)
                # Apply exponential decay
                decay_factors = torch.tensor([[10000**(2*i/(d//2)) for _ in range(k)] 
                                        for i in range(d//2)], device=omega_matrix.weight.device)
                omega_matrix.weight.copy_(rand_freqs / decay_factors)
            
            case "uniform":
                pass

            case "orthogonal":
                nn.init.orthogonal_(omega_matrix.weight)

            case "none":
                nn.init.eye_(omega_matrix.weight)


def rotate(x, sin, cos):
    x1, x2 = x[..., ::2], x[..., 1::2] # (..., d//2)
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1) # (..., d//2, 2)
    return x_rotated.flatten(-2) # (..., d)

class GraphRoPE(nn.Module):
    def __init__(self, 
                 k: int, 
                 d: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 enable: bool = True,
                 init_omega: str = "zero",
                 attn_type: str = "Full",
                 shared_omega_q: nn.Module = None,
                 shared_omega_k: nn.Module = None,
                 double_omega: bool = False,
                 freeze_omega: bool = False,
                 return_logits: bool = False,
                 positional_method: str = None,
                 apply_positional: bool = True,
                 aw_cfg=None):
        """
        Multi-head attention with rotational position encoding.
        k: dimension of rotational features
        d: dimension of input features
        num_heads: number of attention heads
        dropout: attention dropout rate
        shared_omega_q: optional shared Omega matrix for Q (or both Q&K if double_omega=False)
        shared_omega_k: optional shared Omega matrix for K (only used if double_omega=True)
        double_omega: if True, use separate Omega matrices for Q and K rotations
        freeze_omega: if True, make Omega matrices non-trainable
        init_omega: initialization strategy for Omega matrix
            - "zero": initialize with zeros
            - "exponential": exponential decay along columns with random frequencies
            - "uniform": standard PyTorch initialization (default)
            - "orthogonal": orthogonal initialization
            - "none": use identity matrix
        attn_type: type of attention
            - "Full": full attention
            - "Linear": linear attention / Performer
        """
        super().__init__()


        self.k = k
        self.d = d
        self.n_head = num_heads
        self.d_head = d // num_heads
        self.init_omega = init_omega
        self.attn_type = attn_type
        self.positional_method = resolve_positional_method(positional_method, enable)
        self.apply_positional = apply_positional
        self.enable = self.positional_method != "none" and apply_positional
        self.double_omega = double_omega
        self.freeze_omega = freeze_omega
        self.return_logits = return_logits
        self.aw_rezero = False
        self.exact_holonomy = None

        if self.return_logits:
            assert self.attn_type == "Full", "return_logits is only supported for Full attention"
        assert self.d % 2 == 0, "d must be divisible by 2"
        assert self.d_head * num_heads == d, "d must be divisible by num_heads"

        self.WQKV = nn.Linear(d, 3 * d)
        self.WO = nn.Linear(d, d)
        
        if self.attn_type == "Linear":
            assert dropout == 0.0, "dropout is not supported for Performer"
            # Keep the optional Performer dependency lazy.  Full-attention
            # WIRE/AW experiments do not need performer-pytorch installed.
            from .performer_layer import FastAttention
            performer_dim = self.d_head
            low_rank_cfg = _cfg_value(aw_cfg, "low_rank", None)
            if self.positional_method == "lr-aw" and self.apply_positional:
                performer_dim *= int(_cfg_value(low_rank_cfg, "rank", 8))
            performer_features = int(
                _cfg_value(low_rank_cfg, "performer_nb_features", 0)
            )
            if performer_features <= 0:
                # Keep the same random-feature count as the corresponding
                # plain Performer; LR-AW expands input columns, not RF rows.
                performer_features = max(
                    1, int(self.d_head * math.log(max(self.d_head, 2)))
                )
            performer_kernel = str(
                _cfg_value(aw_cfg, "performer_kernel", "softmax")
            ).strip().lower()
            if performer_kernel in {"softmax", "favor+", "favor-plus"}:
                generalized_attention = False
                self.performer_kernel = "softmax-favor+"
            elif performer_kernel in {"relu", "generalized-relu"}:
                generalized_attention = True
                self.performer_kernel = "generalized-relu"
            else:
                raise ValueError(
                    "gt.graphrope.aw.performer_kernel must be softmax or relu, "
                    f"got {performer_kernel!r}"
                )
            self.attention = FastAttention(
                dim_heads=performer_dim,
                nb_features=performer_features,
                generalized_attention=generalized_attention,
                kernel_fn=nn.ReLU(),
            )
        else:
            self.dropout = nn.Dropout(dropout)

        if self.positional_method == "wire":
            if shared_omega_q is not None:
                self.OmegaQ = shared_omega_q
            else:
                self.OmegaQ = nn.Linear(k, d//2, bias=False)
                init_omega_matrix(self.OmegaQ, init_omega, d, k)

            if self.double_omega:
                if shared_omega_k is not None:
                    self.OmegaK = shared_omega_k
                else:
                    self.OmegaK = nn.Linear(k, d//2, bias=False)
                    init_omega_matrix(self.OmegaK, init_omega, d, k)
            
            # Freeze omega matrices if requested
            if self.freeze_omega:
                if hasattr(self, 'OmegaQ'):
                    for param in self.OmegaQ.parameters():
                        param.requires_grad = False
                if hasattr(self, 'OmegaK'):
                    for param in self.OmegaK.parameters():
                        param.requires_grad = False

        elif self.positional_method == "exact-holonomy":
            if self.attn_type != "Full":
                raise ValueError("Exact Holonomy-RoPE requires full dense attention")
            exact_cfg = _cfg_value(aw_cfg, "exact", None)
            if bool(_cfg_value(exact_cfg, "precomputed", True)):
                if bool(_cfg_value(exact_cfg, "learnable_frequencies", False)):
                    raise ValueError("precomputed Exact Holonomy-RoPE cannot learn frequencies")
                self.exact_holonomy = ExactHolonomyRoPE(
                    head_dim=self.d_head,
                    field_protocol=str(
                        _cfg_value(exact_cfg, "field_protocol", "topology-rwdiag-skew-v1")
                    ),
                )
                self.exact_holonomy.precomputed = True
            else:
                max_displacement = _cfg_value(aw_cfg, "max_displacement", math.pi)
                if max_displacement is not None and float(max_displacement) <= 0:
                    max_displacement = None
                self.exact_holonomy = LearnableExactHolonomyRoPE(
                    head_dim=self.d_head,
                    field_node_dim=d,
                    field_hidden_dim=int(_cfg_value(aw_cfg, "field_hidden_dim", 32)),
                    max_displacement=max_displacement,
                    frequency_base=float(_cfg_value(aw_cfg, "frequency_base", 10_000.0)),
                    learnable_frequencies=bool(
                        _cfg_value(exact_cfg, "learnable_frequencies", True)
                    ),
                    diffusion_time=float(_cfg_value(exact_cfg, "diffusion_time", 2.0)),
                    eps=float(_cfg_value(exact_cfg, "eps", 1e-12)),
                    soft_phase_normalization=bool(
                        _cfg_value(exact_cfg, "soft_phase_normalization", False)
                    ),
                )

        elif self.positional_method == "lr-aw":
            low_rank_cfg = _cfg_value(aw_cfg, "low_rank", None)
            carriers = tuple(
                _cfg_value(low_rank_cfg, "carrier_frequencies", ())
            )
            max_displacement = _cfg_value(
                low_rank_cfg,
                "max_displacement",
                _cfg_value(aw_cfg, "max_displacement", math.pi),
            )
            if max_displacement is not None and float(max_displacement) <= 0:
                max_displacement = None
            self.low_rank_aw_rope = LowRankComplexRandomFeatureAWRoPE(
                head_dim=self.d_head,
                rank=int(_cfg_value(low_rank_cfg, "rank", 8)),
                num_steps=int(_cfg_value(low_rank_cfg, "num_steps", 6)),
                z=float(_cfg_value(low_rank_cfg, "z", 0.6)),
                carrier_frequencies=carriers or None,
                num_bands=int(_cfg_value(low_rank_cfg, "num_bands", 4)),
                frequency_base=float(
                    _cfg_value(low_rank_cfg, "frequency_base", 10_000.0)
                ),
                anchor_seed=int(_cfg_value(low_rank_cfg, "anchor_seed", 0)),
                denominator_eps=float(
                    _cfg_value(low_rank_cfg, "denominator_eps", 1e-8)
                ),
                confidence_power=float(
                    _cfg_value(low_rank_cfg, "confidence_power", 0.0)
                ),
                field_source=str(
                    _cfg_value(low_rank_cfg, "field_source", "features")
                ),
                max_displacement=max_displacement,
                share_geometry=bool(
                    _cfg_value(low_rank_cfg, "share_geometry", True)
                ),
            )

        elif self.positional_method.startswith("aw"):
            self.aw_rezero = bool(_cfg_value(aw_cfg, "rezero", False))
            # B/ReZero is a paired NoPE comparison.  AW-only module
            # initialization must not advance the RNG seen by FFNs or later
            # Graph Transformer layers, otherwise the shared backbone would
            # start from different weights despite the exact-zero gate.
            paired_backbone_rng = (
                torch.random.get_rng_state() if self.aw_rezero else None
            )
            max_displacement = _cfg_value(aw_cfg, "max_displacement", math.pi)
            if max_displacement is not None and float(max_displacement) <= 0:
                max_displacement = None
            self.aw_rope = AnalyticWalkRoPE(
                dim=d,
                method=self.positional_method,
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
                preserve_input_norm=bool(
                    _cfg_value(aw_cfg, "preserve_input_norm", False)
                ),
                norm_group_size=int(_cfg_value(aw_cfg, "norm_group_size", 0)),
                field_type=str(_cfg_value(aw_cfg, "field_type", "local-antisymmetric")),
                position_dim=int(_cfg_value(aw_cfg, "position_dim", 3)),
            )
            if paired_backbone_rng is not None:
                torch.random.set_rng_state(paired_backbone_rng)
                self.aw_rezero_gate = nn.Parameter(torch.zeros(()))
                    
    def forward(self, batch):
        """
        batch.x: (num_nodes, d) sparse PyG node features
        batch.t: (num_nodes, k) optional WIRE rotational features
        """
        
        # Project in sparse PyG node order.  Densification happens only after
        # the positional method, making the exact same Q/K/V projections
        # available to NoPE, WIRE, and every AW-RoPE variant.
        QKV = self.WQKV(batch.x)  # (num_nodes, 3*d)
        Q, K, V = QKV.chunk(3, dim=-1)  # Each (num_nodes, d)

        # LR-AW constructs a pairwise relative displacement and applies it to
        # the QK score after head reshaping.  Other methods transform Q/K here.
        low_rank_geometry = None
        exact_transport = None
        exact_nodes = None
        if self.apply_positional and self.positional_method == "wire":
            if not hasattr(batch, "t") or batch.t is None:
                raise ValueError("WIRE requires rotational node positions in batch.t")
            phi_q = self.OmegaQ(batch.t)  # (num_nodes, d//2)
            sin_q = torch.sin(phi_q)  # (num_nodes, d//2)
            cos_q = torch.cos(phi_q)  # (num_nodes, d//2)
            
            if self.double_omega:
                phi_k = self.OmegaK(batch.t)  # (num_nodes, d//2)
                sin_k = torch.sin(phi_k)  # (num_nodes, d//2)
                cos_k = torch.cos(phi_k)  # (num_nodes, d//2)
            else:
                sin_k = sin_q
                cos_k = cos_q

            # Apply rotation before dense batching and head reshaping.
            Q = rotate(Q, sin_q, cos_q)  # (num_nodes, d)
            K = rotate(K, sin_k, cos_k)  # (num_nodes, d)

        elif self.apply_positional and self.positional_method == "exact-holonomy":
            if self.exact_holonomy is None:
                raise RuntimeError("Exact Holonomy-RoPE module was not initialized")
            if self.exact_holonomy.precomputed:
                if not hasattr(batch, "exact_holonomy_transport"):
                    raise RuntimeError(
                        "Exact Holonomy-RoPE requires the offline exact_holonomy_transport sidecar"
                    )
                geometry_cache = getattr(batch, "_exact_holonomy_geometry_cache", None)
                cache_key = (self.exact_holonomy.field_protocol, self.d_head)
                if isinstance(geometry_cache, dict) and cache_key in geometry_cache:
                    exact_transport, exact_nodes = geometry_cache[cache_key]
                else:
                    exact_transport, exact_nodes = self.exact_holonomy.pairwise_transport(
                        batch.exact_holonomy_transport,
                        batch.batch,
                    )
                    if isinstance(geometry_cache, dict):
                        geometry_cache[cache_key] = (exact_transport, exact_nodes)
            else:
                exact_transport, exact_nodes = self.exact_holonomy.pairwise_transport(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    edge_weight=getattr(batch, "edge_weight", None),
                )

        elif self.apply_positional and self.positional_method == "lr-aw":
            if not hasattr(batch, "edge_index") or batch.edge_index is None:
                raise ValueError("LR-AW requires sparse graph edges in batch.edge_index")
            geometry_cache = getattr(batch, "_lr_aw_geometry_cache", None)
            linear_adapter = self.attn_type == "Linear"
            cache_key = (
                self.low_rank_aw_rope.performer_cache_key
                if linear_adapter
                else self.low_rank_aw_rope.cache_key
            )
            if (
                self.low_rank_aw_rope.share_geometry
                and isinstance(geometry_cache, dict)
                and cache_key in geometry_cache
            ):
                low_rank_geometry = geometry_cache[cache_key]
            else:
                geometry_builder = (
                    self.low_rank_aw_rope.performer_geometry
                    if linear_adapter
                    else self.low_rank_aw_rope.geometry
                )
                low_rank_geometry = geometry_builder(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    edge_weight=getattr(batch, "edge_weight", None),
                    positions=getattr(batch, "pos", None),
                )
                if (
                    self.low_rank_aw_rope.share_geometry
                    and isinstance(geometry_cache, dict)
                ):
                    geometry_cache[cache_key] = low_rank_geometry

        elif self.apply_positional and self.positional_method.startswith("aw"):
            if not hasattr(batch, "edge_index") or batch.edge_index is None:
                raise ValueError("AW-RoPE requires sparse graph edges in batch.edge_index")
            edge_weight = getattr(batch, "edge_weight", None)
            reverse_edge = getattr(batch, "_aw_reverse_edge", None)
            if reverse_edge is None and self.positional_method in {"aw-nb", "aw-nb-ms"}:
                reverse_edge = build_reverse_edge_index(batch.edge_index, batch.x.shape[0])
                batch._aw_reverse_edge = reverse_edge
            transported_q, transported_k = self.aw_rope(
                Q,
                K,
                batch.x,
                batch.edge_index,
                edge_weight=edge_weight,
                reverse_edge=reverse_edge,
                positions=getattr(batch, "pos", None),
                precomputed_displacement=getattr(
                    batch, "aw_static_edge_displacement", None
                ),
            )
            if self.aw_rezero:
                # One exact-zero scalar gates the paired Q/K transport, which
                # is the attention analogue of the B/ReZero GIN ablation:
                # x + alpha * (AW(x) - x), alpha(0) = 0.
                gate = self.aw_rezero_gate.to(dtype=Q.dtype)
                Q = Q + gate * (transported_q - Q)
                K = K + gate * (transported_k - K)
            else:
                Q, K = transported_q, transported_k

        Q, real_nodes = to_dense_batch(Q, batch.batch)
        K, _ = to_dense_batch(K, batch.batch)
        V, _ = to_dense_batch(V, batch.batch)
        b, n, _ = Q.size()

        # Reshape for multi-head attention after rotation
        Q = Q.view(b, n, self.n_head, self.d_head).transpose(1, 2)  # (b, num_heads, n, head_dim)
        K = K.view(b, n, self.n_head, self.d_head).transpose(1, 2)  # (b, num_heads, n, head_dim)
        V = V.view(b, n, self.n_head, self.d_head).transpose(1, 2)  # (b, num_heads, n, head_dim)

        if self.attn_type == "Full":
            if exact_transport is not None:
                if exact_nodes is None or exact_nodes.shape != real_nodes.shape:
                    raise RuntimeError("Exact Holonomy geometry and dense masks disagree")
                scores = self.exact_holonomy.attention_scores(Q, K, exact_transport)
                scores.masked_fill_(
                    (~real_nodes).unsqueeze(1).unsqueeze(2), float("-inf")
                )
                attn_weights = self.dropout(F.softmax(scores, dim=-1))
                context = torch.matmul(attn_weights, V)
            elif low_rank_geometry is not None:
                displacement, confidence, geometry_nodes = low_rank_geometry
                if geometry_nodes.shape != real_nodes.shape:
                    raise RuntimeError("LR-AW geometry and dense attention masks disagree")
                scores = self.low_rank_aw_rope.attention_scores(
                    Q, K, displacement, confidence
                )
                scores.masked_fill_(
                    (~real_nodes).unsqueeze(1).unsqueeze(2), float("-inf")
                )
                attn_weights = self.dropout(F.softmax(scores, dim=-1))
                context = torch.matmul(attn_weights, V)
            # Prefer fused SDPA for lower memory use unless logits are requested.
            elif not self.return_logits:
                # SDPA boolean masks use True for positions that participate in
                # attention (the inverse convention of key_padding_mask).
                attn_mask = real_nodes.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, n), broadcastable to (b, h, n, n)
                context = F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False,
                )
            else:
                # Manual path that returns logits; use in-place ops to reduce peak memory
                scores = torch.matmul(Q, K.transpose(-2, -1))  # (b, num_heads, n, n)
                scores.mul_(1 / (self.d_head ** 0.5))
                scores.masked_fill_((~real_nodes).unsqueeze(1).unsqueeze(2), float('-inf'))

                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                # Weighted sum of values
                context = torch.matmul(attn_weights, V)

        else:
            if low_rank_geometry is not None:
                anchor_displacement, anchor_valid, geometry_nodes = low_rank_geometry
                if geometry_nodes.shape != real_nodes.shape:
                    raise RuntimeError("LR-AW geometry and dense attention masks disagree")
                Q, K = self.low_rank_aw_rope.performer_lift(
                    Q, K, anchor_displacement, anchor_valid
                )
            # Zero out fake nodes in K and V for linear attention
            mask = real_nodes.unsqueeze(1).unsqueeze(-1)  # (b, 1, n, 1)
            K = K * mask
            V = V * mask
            
            context = self.attention(Q, K, V)

        # Concatenating heads and projecting back
        context = context.transpose(1, 2).contiguous().view(b, n, self.d)
        context = self.WO(context)

        if self.return_logits:
            return context[real_nodes], scores
        else:
            return context[real_nodes]
