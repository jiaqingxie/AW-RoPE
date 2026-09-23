"""Shared-backbone models for controlled AW-RoPE comparisons."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

from aw_rope import AWRoPE, AntisymmetricEdgeField, MultiScaleAWRoPE


Method = Literal[
    "nope", "aw", "nb", "multiscale-nb", "lr-aw", "exact-holonomy"
]
OfficialMethod = Literal[
    "none", "wire", "aw", "aw-nb", "aw-nb-ms", "lr-aw", "exact-holonomy"
]
GINAWVariant = Literal[
    "legacy",
    "rezero",
    "rezero-normalized",
    "single-normalized",
    "structural-rms",
]


def _mlp(dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 2, dim),
    )


class _ASTEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.node_type = nn.Embedding(98, dim)
        self.node_attribute = nn.Embedding(10_030, dim)
        self.depth = nn.Embedding(21, dim)

    def forward(self, batch: object) -> Tensor:
        x = batch.x.long()
        depth = batch.node_depth.view(-1).long().clamp(max=20)
        return self.node_type(x[:, 0]) + self.node_attribute(x[:, 1]) + self.depth(depth)


class _InputEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, encoder: str) -> None:
        super().__init__()
        self.kind = encoder
        if encoder == "continuous":
            self.encoder: nn.Module = nn.Linear(input_dim, hidden_dim)
        elif encoder == "constant":
            self.encoder = nn.Embedding(1, hidden_dim)
        elif encoder == "ogb-atom":
            from ogb.graphproppred.mol_encoder import AtomEncoder

            self.encoder = AtomEncoder(hidden_dim)
        elif encoder == "ast":
            self.encoder = _ASTEncoder(hidden_dim)
        else:
            raise ValueError(f"unknown input encoder: {encoder}")

    def forward(self, batch: object) -> Tensor:
        if self.kind == "constant":
            index = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.edge_index.device)
            return self.encoder(index)
        if self.kind in {"ogb-atom", "ast"}:
            return self.encoder(batch if self.kind == "ast" else batch.x.long())
        x = batch.x.float()
        if x.ndim == 1:
            x = x[:, None]
        return self.encoder(x)


def _fast_reverse_edges(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Vectorized reverse-edge lookup for the simple graphs in benchmarks."""
    if edge_index.shape[1] == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)
    source, target = edge_index
    keys = source * num_nodes + target
    reverse_keys = target * num_nodes + source
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    positions = torch.searchsorted(sorted_keys, reverse_keys)
    in_range = positions < sorted_keys.numel()
    candidate_position = positions.clamp(max=max(sorted_keys.numel() - 1, 0))
    candidate = order[candidate_position]
    matches = in_range & (keys[candidate] == reverse_keys)
    return torch.where(matches, candidate, torch.full_like(candidate, -1))


class _AnalyticBranch(nn.Module):
    def __init__(
        self,
        dim: int,
        method: Method,
        *,
        num_steps: int,
        z: float,
        learnable_frequencies: bool,
        position_dim: int,
        zero_init_gate: bool = False,
        normalize_resolvent: bool = False,
        normalize_displacement_rms: bool = False,
        initial_phase_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.position_dim = position_dim
        self.zero_init_gate = zero_init_gate
        self.normalize_resolvent = normalize_resolvent
        self.normalize_displacement_rms = normalize_displacement_rms
        if position_dim:
            self.coordinate_projection: nn.Module | None = nn.Linear(position_dim, 1, bias=False)
            self.field: nn.Module | None = None
        else:
            self.coordinate_projection = None
            self.field = AntisymmetricEdgeField(
                node_dim=dim,
                hidden_dim=max(32, dim),
                max_displacement=4.0,
            )
        if method == "multiscale-nb":
            self.rope: nn.Module = MultiScaleAWRoPE(
                dim,
                z_values=(0.2, 0.4, 0.6, 0.8),
                num_steps=num_steps,
                learnable_frequencies=learnable_frequencies,
                non_backtracking=True,
            )
        else:
            self.rope = AWRoPE(
                dim,
                num_steps=num_steps,
                initial_z=z,
                learnable_frequencies=learnable_frequencies,
                non_backtracking=method == "nb",
            )
        self.gate = nn.Parameter(torch.tensor(0.0 if zero_init_gate else -2.0))
        if normalize_displacement_rms:
            if initial_phase_temperature <= 0:
                raise ValueError("initial_phase_temperature must be positive")
            self.log_phase_temperature = nn.Parameter(
                torch.tensor(initial_phase_temperature).log()
            )
        else:
            self.register_parameter("log_phase_temperature", None)

    def _normalizer(self) -> Tensor:
        if not self.normalize_resolvent:
            return torch.ones((), device=self.gate.device)
        if not isinstance(self.rope, AWRoPE):
            raise ValueError("normalized GIN AW variants require ordinary AWRoPE")
        z = self.rope.z
        coefficient = torch.ones_like(z)
        total = torch.ones_like(z)
        for _ in range(self.rope.num_steps):
            coefficient = coefficient * z
            total = total + coefficient
        return total

    def _normalize_displacement(
        self,
        displacement: Tensor,
        edge_index: Tensor,
        node_batch: Tensor | None,
    ) -> Tensor:
        if not self.normalize_displacement_rms:
            return displacement
        if node_batch is None:
            edge_batch = torch.zeros_like(edge_index[0])
            graph_count = 1
        else:
            edge_batch = node_batch[edge_index[0]]
            graph_count = int(node_batch.max()) + 1 if node_batch.numel() else 1
        # CUDA autocast can leave the learned displacement in bfloat16 while
        # promoting ``square`` to float32.  Accumulate RMS statistics in
        # explicit float32 so index_add_ always sees matching dtypes, then
        # restore the branch's original activation dtype.
        source_dtype = displacement.dtype
        displacement_float = displacement.float()
        sum_square = torch.zeros(
            graph_count, dtype=displacement_float.dtype, device=displacement.device
        )
        counts = torch.zeros_like(sum_square)
        sum_square.index_add_(0, edge_batch, displacement_float.square())
        counts.index_add_(0, edge_batch, torch.ones_like(displacement_float))
        rms = (sum_square / counts.clamp_min(1)).clamp_min(1e-12).sqrt()
        assert self.log_phase_temperature is not None
        temperature = self.log_phase_temperature.exp().float()
        normalized = (
            temperature
            * displacement_float
            / rms[edge_batch].clamp_min(1e-6)
        )
        return normalized.to(dtype=source_dtype)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        positions: Tensor | None,
        reverse_edge: Tensor | None,
        field_features: Tensor | None = None,
        node_batch: Tensor | None = None,
        precomputed_transport: Tensor | None = None,
    ) -> Tensor:
        if precomputed_transport is not None:
            raise ValueError("AW branches do not consume Exact Holonomy transports")
        if self.coordinate_projection is not None:
            if positions is None:
                raise ValueError("coordinate AW-RoPE requires batch.pos")
            source, target = edge_index
            displacement = self.coordinate_projection(positions[target] - positions[source]).squeeze(-1)
        else:
            assert self.field is not None
            displacement = self.field(
                x if field_features is None else field_features, edge_index
            )
        displacement = self._normalize_displacement(
            displacement, edge_index, node_batch
        )
        self.last_displacement_rms = (
            displacement.detach().square().mean().sqrt()
        )
        transported = self.rope(x, edge_index, displacement, reverse_edge=reverse_edge)
        transported = transported / self._normalizer()
        gate = self.gate if self.zero_init_gate else torch.sigmoid(self.gate)
        return x + gate * (transported - x)


def _canonical_undirected_edges(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Return every non-self-loop undirected edge once in canonical order."""
    source, target = edge_index
    keep = source != target
    lower = torch.minimum(source[keep], target[keep])
    upper = torch.maximum(source[keep], target[keep])
    if lower.numel() == 0:
        return edge_index.new_empty((2, 0))
    keys = lower * num_nodes + upper
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys[order]
    first = torch.ones_like(sorted_keys, dtype=torch.bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    chosen = order[first]
    return torch.stack((lower[chosen], upper[chosen]))


class _ExactHolonomyBranch(nn.Module):
    """Residual GIN adapter for an offline exact/full pair transport."""

    def __init__(
        self,
        dim: int,
        *,
        diffusion_time: float,
        position_dim: int,
        heat_kernel_method: str = "precomputed",
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("exact Holonomy residual dimension must be positive and even")
        if diffusion_time < 0:
            raise ValueError("diffusion_time must be non-negative")
        if heat_kernel_method != "precomputed":
            raise ValueError(
                "canonical exact Holonomy GIN requires heat_kernel_method='precomputed'"
            )
        self.dim = dim
        self.num_frequencies = dim // 2
        self.diffusion_time = float(diffusion_time)
        self.heat_kernel_method = heat_kernel_method
        self.position_dim = position_dim
        # Match the legacy AW residual interpolation at initialization.
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.zero_init_gate = False
        self.register_parameter("log_phase_temperature", None)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        positions: Tensor | None,
        reverse_edge: Tensor | None,
        field_features: Tensor | None = None,
        node_batch: Tensor | None = None,
        precomputed_transport: Tensor | None = None,
    ) -> Tensor:
        del reverse_edge, positions, field_features, edge_index
        if node_batch is None:
            node_batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if precomputed_transport is None:
            raise RuntimeError(
                "exact Holonomy GIN requires batch.exact_holonomy_transport from preprocessing"
            )
        if (
            not torch.is_complex(precomputed_transport)
            or precomputed_transport.ndim != 2
            or precomputed_transport.shape[1] != self.num_frequencies
        ):
            raise ValueError("invalid exact Holonomy sidecar tensor")
        self.last_displacement_rms = x.new_zeros((), dtype=torch.float32)
        graph_count = int(node_batch.max()) + 1 if node_batch.numel() else 1
        counts = torch.bincount(node_batch, minlength=graph_count)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
        if precomputed_transport.shape[0] != int(counts.square().sum()):
            raise ValueError("exact Holonomy sidecar rows do not match batch graph sizes")
        transported_graphs: list[Tensor] = []
        transport_cursor = 0
        for graph_index in range(graph_count):
            count = int(counts[graph_index])
            start = int(offsets[graph_index])
            rows = count * count
            transport = precomputed_transport[
                transport_cursor : transport_cursor + rows
            ].view(count, count, self.num_frequencies).permute(2, 0, 1)
            transport_cursor += rows
            graph_x = x[start : start + count]
            paired = graph_x.reshape(count, self.num_frequencies, 2)
            if graph_x.dtype == torch.float64:
                complex_x = torch.complex(paired[..., 0], paired[..., 1])
            else:
                complex_x = torch.complex(
                    paired[..., 0].float(), paired[..., 1].float()
                )
            complex_output = torch.einsum(
                "fij,jf->if", transport.to(complex_x.dtype), complex_x
            ) / max(count, 1)
            real_output = torch.view_as_real(complex_output).reshape(count, self.dim)
            transported_graphs.append(real_output.to(dtype=x.dtype))
        transported = torch.cat(transported_graphs, dim=0)
        return x + torch.sigmoid(self.gate) * (transported - x)


class GraphPredictionModel(nn.Module):
    """GIN backbone with an optional analytic transport branch per layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 32,
        num_layers: int = 4,
        dropout: float = 0.2,
        method: Method = "nope",
        num_steps: int = 8,
        z: float = 0.8,
        learnable_frequencies: bool = False,
        task: str = "graph-regression",
        encoder: str = "continuous",
        position_dim: int = 0,
        sequence_length: int = 5,
        gin_aw_variant: GINAWVariant = "legacy",
        diffusion_time: float = 2.0,
        exact_heat_kernel_method: str = "precomputed",
    ) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for rotary pairs")
        if method not in {"nope", "aw", "nb", "multiscale-nb", "exact-holonomy"}:
            raise ValueError(f"unknown method: {method}")
        variants = {
            "legacy",
            "rezero",
            "rezero-normalized",
            "single-normalized",
            "structural-rms",
        }
        if gin_aw_variant not in variants:
            raise ValueError(f"unknown GIN AW variant: {gin_aw_variant}")
        if method != "aw" and gin_aw_variant != "legacy":
            raise ValueError("GIN AW variants are only supported for method='aw'")
        self.method = method
        self.gin_aw_variant = gin_aw_variant
        self.task = task
        self.dropout = dropout
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.input_encoder = _InputEncoder(input_dim, hidden_dim, encoder)
        self.convolutions = nn.ModuleList(
            GINConv(_mlp(hidden_dim, dropout), train_eps=True) for _ in range(num_layers)
        )
        self.normalizations = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        single_injection = gin_aw_variant in {"single-normalized", "structural-rms"}
        zero_init_gate = gin_aw_variant != "legacy"
        normalize_resolvent = gin_aw_variant in {
            "rezero-normalized", "single-normalized", "structural-rms"
        }
        # Keep the shared GIN/head initialization paired with the NoPE arm.
        # The AW-only module draws below must not silently change the head seed.
        paired_head_rng = (
            torch.random.get_rng_state()
            if method == "aw" and gin_aw_variant != "legacy"
            else None
        )
        if method == "nope":
            self.analytic_branches = nn.ModuleList()
        elif method == "exact-holonomy":
            self.analytic_branches = nn.ModuleList(
                _ExactHolonomyBranch(
                    hidden_dim,
                    diffusion_time=diffusion_time,
                    position_dim=position_dim,
                    heat_kernel_method=exact_heat_kernel_method,
                )
                for _ in range(num_layers)
            )
        else:
            self.analytic_branches = nn.ModuleList(
                _AnalyticBranch(
                    hidden_dim,
                    method,
                    num_steps=num_steps,
                    z=z,
                    learnable_frequencies=learnable_frequencies,
                    position_dim=position_dim,
                    zero_init_gate=zero_init_gate,
                    normalize_resolvent=normalize_resolvent,
                    normalize_displacement_rms=gin_aw_variant == "structural-rms",
                )
                for _ in range(1 if single_injection else num_layers)
            )
        if gin_aw_variant == "structural-rms":
            self.degree_projection: nn.Module | None = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
            )
        else:
            self.degree_projection = None
        if paired_head_rng is not None:
            torch.random.set_rng_state(paired_head_rng)
        head_outputs = output_dim * sequence_length if task == "sequence-prediction" else output_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, head_outputs),
        )

    def forward(self, batch: object) -> Tensor:
        x = self.input_encoder(batch)
        reverse_edge = None
        if self.method in {"nb", "multiscale-nb"}:
            reverse_edge = _fast_reverse_edges(batch.edge_index, batch.num_nodes)
        positions = getattr(batch, "pos", None)
        node_batch = getattr(batch, "batch", None)
        if self.gin_aw_variant in {"single-normalized", "structural-rms"}:
            field_features = x
            if self.gin_aw_variant == "structural-rms":
                assert self.degree_projection is not None
                degree = torch.bincount(
                    batch.edge_index[0], minlength=batch.num_nodes
                ).to(dtype=x.dtype, device=x.device)
                log_degree = torch.log1p(degree)
                if node_batch is None:
                    mean = log_degree.mean().expand_as(log_degree)
                    variance = log_degree.var(unbiased=False).clamp_min(1e-6).expand_as(log_degree)
                else:
                    graph_count = int(node_batch.max()) + 1 if node_batch.numel() else 1
                    # log1p is promoted to float32 under CUDA autocast even
                    # when the encoder activation is bfloat16. Accumulate in
                    # the actual source dtype so index_add_ remains valid.
                    counts = torch.bincount(node_batch, minlength=graph_count).to(
                        log_degree.dtype
                    )
                    sums = torch.zeros(
                        graph_count, dtype=log_degree.dtype, device=x.device
                    )
                    sums.index_add_(0, node_batch, log_degree)
                    means = sums / counts.clamp_min(1)
                    centered = log_degree - means[node_batch]
                    square_sums = torch.zeros_like(sums)
                    square_sums.index_add_(0, node_batch, centered.square())
                    variances = square_sums / counts.clamp_min(1)
                    mean = means[node_batch]
                    variance = variances[node_batch].clamp_min(1e-6)
                normalized_degree = (log_degree - mean) / variance.sqrt()
                field_features = x + self.degree_projection(normalized_degree[:, None])
            x = self.analytic_branches[0](
                x,
                batch.edge_index,
                positions=positions,
                reverse_edge=reverse_edge,
                field_features=field_features,
                node_batch=node_batch,
            )
        for index, (convolution, normalization) in enumerate(
            zip(self.convolutions, self.normalizations)
        ):
            if self.method != "nope" and self.gin_aw_variant not in {
                "single-normalized", "structural-rms"
            }:
                x = self.analytic_branches[index](
                    x,
                    batch.edge_index,
                    positions=positions,
                    reverse_edge=reverse_edge,
                    node_batch=node_batch,
                    precomputed_transport=getattr(
                        batch, "exact_holonomy_transport", None
                    ),
                )
            update = convolution(x, batch.edge_index)
            x = normalization(
                x + torch.nn.functional.dropout(update, p=self.dropout, training=self.training)
            )
        if self.task == "node-classification":
            return self.head(x)
        graph_features = global_mean_pool(x, batch.batch)
        output = self.head(graph_features)
        if self.task == "sequence-prediction":
            return output.view(-1, self.sequence_length, self.output_dim)
        return output


class _PointAxisLayerNorm(nn.Module):
    """Flax ``LayerNorm(reduction_axes=-2)`` for packed PyG point clouds.

    Scenic's PCT normalizes every feature over the point axis, independently
    for each cloud.  A regular ``torch.nn.LayerNorm`` normalizes the feature
    axis and is therefore not equivalent.  This packed implementation keeps
    the official PCT reduction while allowing several clouds in one PyG batch.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor, node_batch: Tensor) -> Tensor:
        source = x.float()
        graph_count = int(node_batch.max()) + 1 if node_batch.numel() else 1
        sums = source.new_zeros((graph_count, source.shape[-1]))
        sums.index_add_(0, node_batch, source)
        counts = torch.bincount(node_batch, minlength=graph_count).to(source.dtype)
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        centered = source - means[node_batch]
        square_sums = source.new_zeros((graph_count, source.shape[-1]))
        square_sums.index_add_(0, node_batch, centered.square())
        variances = square_sums / counts.clamp_min(1).unsqueeze(-1)
        normalized = centered * torch.rsqrt(variances[node_batch] + self.eps)
        return (normalized * self.weight + self.bias).to(x.dtype)


class _OfficialGraphTransformerBlock(nn.Module):
    """Residual GT block around the official WIRE ``GraphRoPE`` layer."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dropout: float,
        method: OfficialMethod,
        position_dim: int,
        num_steps: int,
        z: float,
        field_type: str,
        learnable_frequencies: bool,
        attention_type: str = "Full",
        pointcloud_style: bool = False,
        performer_nb_features: int = 0,
        diffusion_time: float = 2.0,
        exact_heat_kernel_method: str = "precomputed",
    ) -> None:
        super().__init__()
        from graphgps.layer.graphrope import GraphRoPE

        aw_cfg = SimpleNamespace(
            num_steps=num_steps,
            z=z,
            z_values=[0.2, 0.4, 0.6, 0.8],
            field_hidden_dim=max(32, dim),
            max_displacement=3.141592653589793,
            frequency_base=10_000.0,
            learnable_frequencies=learnable_frequencies,
            learnable_z=True,
            normalize_resolvent=False,
            field_type=field_type,
            position_dim=position_dim,
            performer_kernel="relu" if pointcloud_style else "softmax",
            low_rank=SimpleNamespace(
                rank=8,
                num_steps=num_steps,
                z=z,
                carrier_frequencies=[],
                num_bands=4,
                frequency_base=10_000.0,
                anchor_seed=0,
                denominator_eps=1e-8,
                confidence_power=0.0,
                field_source=(
                    "coordinates" if field_type == "coordinate" else "features"
                ),
                max_displacement=3.141592653589793,
                share_geometry=True,
                performer_nb_features=performer_nb_features,
            ),
            exact=SimpleNamespace(
                diffusion_time=diffusion_time,
                eps=1e-12,
                soft_phase_normalization=False,
                heat_kernel_method=exact_heat_kernel_method,
                learnable_frequencies=False,
                precomputed=True,
                field_protocol="topology-rwdiag-skew-v1",
            ),
        )
        self.method = method
        if attention_type not in {"Full", "Linear"}:
            raise ValueError(f"unknown attention type: {attention_type}")
        self.attention = GraphRoPE(
            k=position_dim if method == "wire" else 0,
            d=dim,
            num_heads=heads,
            # The official Performer/FastAttention implementation does not
            # support attention-weight dropout.  Residual and FFN dropout
            # remain unchanged below.
            dropout=0.0 if attention_type == "Linear" else dropout,
            enable=method != "none",
            init_omega="uniform",
            attn_type=attention_type,
            positional_method=method,
            aw_cfg=aw_cfg,
        )
        self.pointcloud_style = pointcloud_style
        self.dropout = nn.Dropout(dropout)
        if pointcloud_style:
            # Scenic PCT: Conv(QKV attention) -> point-axis LayerNorm -> ReLU
            # -> residual.  It deliberately has no Transformer FFN sublayer.
            self.point_axis_norm = _PointAxisLayerNorm(dim)
        else:
            self.attention_norm = nn.LayerNorm(dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(dim, 2 * dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * dim, dim),
            )
            self.feed_forward_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, batch: object) -> Tensor:
        from torch_geometric.data import Batch

        attention_batch = Batch(
            x=x,
            batch=batch.batch,
            edge_index=batch.edge_index,
        )
        if hasattr(batch, "pos"):
            attention_batch.pos = batch.pos
        if hasattr(batch, "exact_holonomy_transport"):
            attention_batch.exact_holonomy_transport = batch.exact_holonomy_transport
        if self.method == "wire":
            if not hasattr(batch, "pos"):
                raise ValueError("Cartesian WIRE requires batch.pos")
            attention_batch.t = batch.pos
        reverse_edge = getattr(batch, "_aw_reverse_edge", None)
        if reverse_edge is not None:
            attention_batch._aw_reverse_edge = reverse_edge
        geometry_cache = getattr(batch, "_lr_aw_geometry_cache", None)
        if geometry_cache is not None:
            attention_batch._lr_aw_geometry_cache = geometry_cache
        exact_geometry_cache = getattr(batch, "_exact_holonomy_geometry_cache", None)
        if exact_geometry_cache is not None:
            attention_batch._exact_holonomy_geometry_cache = exact_geometry_cache
        update = self.attention(attention_batch)
        if hasattr(attention_batch, "_aw_reverse_edge"):
            batch._aw_reverse_edge = attention_batch._aw_reverse_edge
        if self.pointcloud_style:
            return x + torch.relu(self.point_axis_norm(update, batch.batch))
        x = self.attention_norm(x + self.dropout(update))
        return self.feed_forward_norm(x + self.dropout(self.feed_forward(x)))


class OfficialGraphTransformerPredictionModel(nn.Module):
    """Point-cloud GT using the official WIRE Q/K attention implementation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        method: OfficialMethod = "none",
        num_steps: int = 8,
        z: float = 0.8,
        field_type: str = "coordinate",
        position_dim: int = 3,
        learnable_frequencies: bool = True,
        task: str = "graph-classification",
        attention_type: str = "Full",
        diffusion_time: float = 2.0,
        exact_heat_kernel_method: str = "precomputed",
    ) -> None:
        super().__init__()
        if hidden_dim % heads or hidden_dim % 2:
            raise ValueError("hidden_dim must be even and divisible by heads")
        self.task = task
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            _OfficialGraphTransformerBlock(
                hidden_dim,
                heads=heads,
                dropout=dropout,
                method=method,
                position_dim=position_dim,
                num_steps=num_steps,
                z=z,
                field_type=field_type,
                learnable_frequencies=learnable_frequencies,
                attention_type=attention_type,
                diffusion_time=diffusion_time,
                exact_heat_kernel_method=exact_heat_kernel_method,
            )
            for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: object) -> Tensor:
        x = self.input_projection(batch.x.float())
        batch._lr_aw_geometry_cache = {}
        batch._exact_holonomy_geometry_cache = {}
        for layer in self.layers:
            x = layer(x, batch)
        if self.task == "node-classification":
            return self.head(x)
        return self.head(global_mean_pool(x, batch.batch))


class OfficialPointCloudTransformerPredictionModel(nn.Module):
    """PyG port of the WIRE/Scenic point-cloud prediction topology.

    The attention kernel remains the official GraphRoPE implementation.  The
    point-cloud-specific encoder aggregation and heads mirror PCT: two input
    projections, four attention features concatenated and lifted to 1024
    channels, max pooling for ModelNet40, and local/max/mean fusion for
    ShapeNet segmentation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 1,
        dropout: float = 0.5,
        method: OfficialMethod = "none",
        num_steps: int = 8,
        z: float = 0.8,
        field_type: str = "coordinate",
        position_dim: int = 3,
        learnable_frequencies: bool = True,
        task: str = "graph-classification",
        attention_type: str = "Full",
        diffusion_time: float = 2.0,
        exact_heat_kernel_method: str = "precomputed",
    ) -> None:
        super().__init__()
        if hidden_dim % heads or hidden_dim % 2:
            raise ValueError("hidden_dim must be even and divisible by heads")
        if task not in {"graph-classification", "node-classification"}:
            raise ValueError(f"unsupported point-cloud task: {task}")
        self.task = task
        self.input_projection1 = nn.Linear(input_dim, hidden_dim)
        self.input_norm1 = _PointAxisLayerNorm(hidden_dim)
        self.input_projection2 = nn.Linear(hidden_dim, hidden_dim)
        self.input_norm2 = _PointAxisLayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            _OfficialGraphTransformerBlock(
                hidden_dim,
                heads=heads,
                dropout=dropout,
                method=method,
                position_dim=position_dim,
                num_steps=num_steps,
                z=z,
                field_type=field_type,
                learnable_frequencies=learnable_frequencies,
                attention_type=attention_type,
                pointcloud_style=True,
                # Scenic's official PCT configs fix Performer RFs to 256.
                performer_nb_features=256,
                diffusion_time=diffusion_time,
                exact_heat_kernel_method=exact_heat_kernel_method,
            )
            for _ in range(num_layers)
        )
        encoder_dim = 1024
        self.encoder_projection = nn.Linear(hidden_dim * num_layers, encoder_dim)
        if task == "graph-classification":
            self.classifier1 = nn.Linear(encoder_dim, 4 * hidden_dim)
            self.classifier_norm1 = _PointAxisLayerNorm(4 * hidden_dim)
            self.classifier2 = nn.Linear(4 * hidden_dim, 2 * hidden_dim)
            self.classifier_norm2 = _PointAxisLayerNorm(2 * hidden_dim)
            self.classifier_out = nn.Linear(2 * hidden_dim, output_dim)
            self.head_dropout = nn.Dropout(dropout)
        else:
            category_dim = hidden_dim // 2
            self.category_projection = nn.Linear(16, category_dim)
            self.category_norm = nn.BatchNorm1d(
                category_dim, eps=1e-5, momentum=0.01
            )
            self.segmenter1 = nn.Linear(
                3 * encoder_dim + category_dim, 4 * hidden_dim
            )
            self.segmenter_norm1 = nn.BatchNorm1d(
                4 * hidden_dim, eps=1e-5, momentum=0.01
            )
            self.segmenter2 = nn.Linear(4 * hidden_dim, 2 * hidden_dim)
            self.segmenter_norm2 = nn.BatchNorm1d(
                2 * hidden_dim, eps=1e-5, momentum=0.01
            )
            self.segmenter_out = nn.Linear(2 * hidden_dim, output_dim)
            self.head_dropout = nn.Dropout(dropout)

    def forward(self, batch: object) -> Tensor:
        x = self.input_projection1(batch.x.float())
        x = self.input_norm1(x, batch.batch)
        x = self.input_projection2(x)
        x = self.input_norm2(x, batch.batch)
        batch._lr_aw_geometry_cache = {}
        batch._exact_holonomy_geometry_cache = {}
        layer_outputs: list[Tensor] = []
        for layer in self.layers:
            x = layer(x, batch)
            layer_outputs.append(x)
        pointwise = self.encoder_projection(torch.cat(layer_outputs, dim=-1))
        if self.task == "graph-classification":
            pooled = global_max_pool(pointwise, batch.batch)
            # On a [batch, feature] tensor Scenic's reduction axis -2 is the
            # example axis.  Represent it as a single normalization group.
            normalization_group = torch.zeros(
                pooled.shape[0], dtype=torch.long, device=pooled.device
            )
            output = self.classifier1(pooled)
            output = self.classifier_norm1(output, normalization_group)
            output = self.head_dropout(torch.nn.functional.leaky_relu(output, 0.2))
            output = self.classifier2(output)
            output = self.classifier_norm2(output, normalization_group)
            output = self.head_dropout(torch.nn.functional.leaky_relu(output, 0.2))
            return self.classifier_out(output)
        maximum = global_max_pool(pointwise, batch.batch)[batch.batch]
        mean = global_mean_pool(pointwise, batch.batch)[batch.batch]
        category = getattr(batch, "category", None)
        if category is None or category.numel() != int(batch.num_graphs):
            raise ValueError("ShapeNet PCT requires one category label per cloud")
        category = torch.nn.functional.one_hot(
            category.view(-1).long(), num_classes=16
        ).to(pointwise.dtype)
        category_features = self.category_projection(category)
        category_features = self.category_norm(category_features)
        category_features = torch.nn.functional.leaky_relu(
            category_features, 0.2
        )[batch.batch]
        output = self.segmenter1(torch.cat(
            (pointwise, maximum, mean, category_features), dim=-1
        ))
        output = self.segmenter_norm1(output)
        output = self.head_dropout(torch.nn.functional.leaky_relu(output, 0.2))
        output = self.segmenter2(output)
        output = self.segmenter_norm2(output)
        return self.segmenter_out(torch.nn.functional.leaky_relu(output, 0.2))
