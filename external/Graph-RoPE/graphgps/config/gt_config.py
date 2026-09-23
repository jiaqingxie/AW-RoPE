from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_gt')
def set_cfg_gt(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """

    # Positional encodings argument group
    cfg.gt = CN()

    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = 'SANLayer'

    # Number of Transformer layers in the model
    cfg.gt.layers = 3

    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 8

    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 64

    # Full attention SAN transformer including all possible pairwise edges
    cfg.gt.full_graph = True

    # SAN real vs fake edge attention weighting coefficient
    cfg.gt.gamma = 1e-5

    # Histogram of in-degrees of nodes in the training set used by PNAConv.
    # Used when `gt.layer_type: PNAConv+...`. If empty it is precomputed during
    # the dataset loading process.
    cfg.gt.pna_degrees = []

    # Dropout in feed-forward module.
    cfg.gt.dropout = 0.0

    # Dropout in self-attention.
    cfg.gt.attn_dropout = 0.0

    cfg.gt.layer_norm = False

    cfg.gt.batch_norm = True

    cfg.gt.residual = True

    # BigBird model/GPS-BigBird layer.
    cfg.gt.bigbird = CN()

    cfg.gt.bigbird.attention_type = "block_sparse"

    cfg.gt.bigbird.chunk_size_feed_forward = 0

    cfg.gt.bigbird.is_decoder = False

    cfg.gt.bigbird.add_cross_attention = False

    cfg.gt.bigbird.hidden_act = "relu"

    cfg.gt.bigbird.max_position_embeddings = 128

    cfg.gt.bigbird.use_bias = False

    cfg.gt.bigbird.num_random_blocks = 3

    cfg.gt.bigbird.block_size = 3

    cfg.gt.bigbird.layer_norm_eps = 1e-6

    cfg.gt.graphrope = CN()

    cfg.gt.graphrope.enable = False
    # Positional mechanism inside the official GraphRoPE attention.  Leave
    # empty for backward compatibility (enable=True -> wire, False -> none).
    # Explicit choices: none, wire, exact-holonomy, aw, aw-nb, aw-nb-ms, lr-aw.
    cfg.gt.graphrope.method = ""
    cfg.gt.graphrope.init_omega = "zero"
    cfg.gt.graphrope.attn_type = "Full"
    cfg.gt.graphrope.encoder = ""
    cfg.gt.graphrope.t_dim = 0
    cfg.gt.graphrope.share_omega = False
    cfg.gt.graphrope.double_omega = False
    cfg.gt.graphrope.freeze_omega = False

    # Analytic-Walk RoPE.  These values are ignored by none/WIRE runs, so one
    # common experiment config can switch methods without changing the
    # Graph Transformer backbone or training protocol.
    cfg.gt.graphrope.aw = CN()
    cfg.gt.graphrope.aw.num_steps = 8
    cfg.gt.graphrope.aw.z = 0.8
    cfg.gt.graphrope.aw.z_values = [0.2, 0.4, 0.6, 0.8]
    cfg.gt.graphrope.aw.field_hidden_dim = 32
    cfg.gt.graphrope.aw.max_displacement = 3.141592653589793
    cfg.gt.graphrope.aw.frequency_base = 10000.0
    cfg.gt.graphrope.aw.learnable_frequencies = True
    cfg.gt.graphrope.aw.learnable_z = True
    cfg.gt.graphrope.aw.normalize_resolvent = False
    # B/ReZero AW-RoPE: wrap the transported Q/K branch in an exact-zero
    # residual gate.  At initialization this is functionally identical to
    # the paired NoPE attention while retaining a learnable AW branch.
    cfg.gt.graphrope.aw.rezero = False
    # Scale-safe interpolation for softmax FAVOR+.  Defaults exactly preserve
    # the original AW-RoPE transport used by all existing experiments.
    cfg.gt.graphrope.aw.residual_mix = 1.0
    cfg.gt.graphrope.aw.preserve_input_norm = False
    cfg.gt.graphrope.aw.norm_group_size = 0
    cfg.gt.graphrope.aw.field_type = "local-antisymmetric"
    cfg.gt.graphrope.aw.position_dim = 3
    cfg.gt.graphrope.aw.fixed = CN()
    cfg.gt.graphrope.aw.fixed.cache_path = ""
    cfg.gt.graphrope.aw.fixed.field_protocol = "topology-rwdiag-skew-v1"
    cfg.gt.graphrope.aw.fixed.variant = "nonflat"
    cfg.gt.graphrope.aw.injection = "all"
    # Official GraphGPS/Graph-RoPE graph Performer uses the default FAVOR+
    # softmax feature map.  Point-cloud PCT and explicit compatibility studies
    # may override this with ``relu``.
    cfg.gt.graphrope.aw.performer_kernel = "softmax"

    # Full/Exact Holonomy-RoPE: one dense connection heat kernel for every
    # rotary frequency.  ``precomputed=True`` consumes a fixed sidecar;
    # ``precomputed=False`` learns an antisymmetric edge field end-to-end and
    # differentiates through the exact matrix exponential each forward pass.
    cfg.gt.graphrope.aw.exact = CN()
    cfg.gt.graphrope.aw.exact.diffusion_time = 2.0
    cfg.gt.graphrope.aw.exact.eps = 1e-12
    cfg.gt.graphrope.aw.exact.soft_phase_normalization = False
    cfg.gt.graphrope.aw.exact.heat_kernel_method = "matrix_exp"
    cfg.gt.graphrope.aw.exact.learnable_frequencies = False
    cfg.gt.graphrope.aw.exact.precomputed = True
    cfg.gt.graphrope.aw.exact.cache_path = ""
    cfg.gt.graphrope.aw.exact.field_protocol = "topology-rwdiag-skew-v1"

    # LR-CRF-AW-RoPE: sparse rank-width propagation followed by B-band
    # pairwise rotary scores in full attention, or separable anchor-phase Q/K
    # lifting in Performer.  V1 is parameter-free.
    cfg.gt.graphrope.aw.low_rank = CN()
    cfg.gt.graphrope.aw.low_rank.rank = 8
    cfg.gt.graphrope.aw.low_rank.num_steps = 6
    cfg.gt.graphrope.aw.low_rank.z = 0.6
    # Empty selects the design default nu = pi / (4K).  Two increasing
    # frequencies enable coarse-to-fine carrier unwrapping.
    cfg.gt.graphrope.aw.low_rank.carrier_frequencies = []
    cfg.gt.graphrope.aw.low_rank.num_bands = 4
    cfg.gt.graphrope.aw.low_rank.frequency_base = 10000.0
    cfg.gt.graphrope.aw.low_rank.anchor_seed = 0
    cfg.gt.graphrope.aw.low_rank.denominator_eps = 1e-8
    cfg.gt.graphrope.aw.low_rank.confidence_power = 0.0
    cfg.gt.graphrope.aw.low_rank.field_source = "features"
    cfg.gt.graphrope.aw.low_rank.max_displacement = 3.141592653589793
    cfg.gt.graphrope.aw.low_rank.share_geometry = True
    # 0 keeps the original Performer's RF-row count while lifting its input
    # columns by rank.  Set a positive value to override the RF count.
    cfg.gt.graphrope.aw.low_rank.performer_nb_features = 0
