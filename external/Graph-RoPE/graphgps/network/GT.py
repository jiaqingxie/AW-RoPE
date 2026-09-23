import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch_geometric.utils import to_dense_batch

from graphgps.layer.graphrope import GraphRoPE, init_omega_matrix, resolve_positional_method
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True, use_bias=False, layer_type='Multihead', cfg=None,
                 shared_omega_q=None, shared_omega_k=None,
                 positional_method_override=None,
                 positional_active=True):
        super().__init__()

        self.layer_type = layer_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        if self.layer_type == 'Multihead':
            self.attention = MultiheadAttention(
                embed_dim=in_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                bias=use_bias,
                batch_first=True
            )

        elif self.layer_type == 'GraphRoPE':
            self.attention = GraphRoPE(
                k=cfg.gt.graphrope.t_dim,
                d=self.out_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                enable=cfg.gt.graphrope.enable,
                init_omega=cfg.gt.graphrope.init_omega,
                attn_type=cfg.gt.graphrope.attn_type,
                shared_omega_q=shared_omega_q,
                shared_omega_k=shared_omega_k,
                double_omega=cfg.gt.graphrope.double_omega,
                freeze_omega=cfg.gt.graphrope.freeze_omega,
                positional_method=(positional_method_override
                                   if positional_method_override is not None
                                   else getattr(cfg.gt.graphrope, 'method', None)),
                apply_positional=positional_active,
                aw_cfg=getattr(cfg.gt.graphrope, 'aw', None),
            )

        else:
            raise ValueError(f"Invalid attention type: {layer_type}")

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, batch):

        h_in1 = batch.x  # for first residual connection

        if self.layer_type == 'GraphRoPE':
            h = self.attention(batch)
        elif self.layer_type == 'Multihead':
            h_dense, mask = to_dense_batch(batch.x, batch.batch)
            h = self.attention(h_dense, h_dense, h_dense,
                                  attn_mask=None,
                                  key_padding_mask=~mask)[0][mask]
        else:
            raise ValueError(f"Invalid attention type: {self.layer_type}")


        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch


@register_network('GT')
class GT(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        Encoder = register.node_encoder_dict[
            cfg.dataset.node_encoder_name]
        self.encoder = Encoder(emb_dim=cfg.gt.dim_hidden)

        # Create shared Omega matrix if configured
        shared_omega_q = None
        shared_omega_k = None
        positional_method = resolve_positional_method(
            getattr(cfg.gt.graphrope, 'method', None),
            cfg.gt.graphrope.enable,
        )
        if positional_method == 'wire':
            RoPEncoder = register.node_encoder_dict[cfg.gt.graphrope.encoder]
            self.rotational_encoder = RoPEncoder(emb_dim=cfg.gt.graphrope.t_dim)
            
            if cfg.gt.graphrope.share_omega:
                shared_omega_q = nn.Linear(cfg.gt.graphrope.t_dim, cfg.gt.dim_hidden // 2, bias=False)
                
                # Initialize the shared Omega matrix
                init_omega_matrix(shared_omega_q, cfg.gt.graphrope.init_omega, cfg.gt.dim_hidden, cfg.gt.graphrope.t_dim)
                            
                # Create second shared Omega matrix for K if double_omega is enabled
                if cfg.gt.graphrope.double_omega:
                    shared_omega_k = nn.Linear(cfg.gt.graphrope.t_dim, cfg.gt.dim_hidden // 2, bias=False)
                    
                    init_omega_matrix(shared_omega_k, cfg.gt.graphrope.init_omega, cfg.gt.dim_hidden, cfg.gt.graphrope.t_dim)
                
                # Freeze shared omega matrices if requested
                if cfg.gt.graphrope.freeze_omega:
                    for param in shared_omega_q.parameters():
                        param.requires_grad = False
                    if shared_omega_k is not None:
                        for param in shared_omega_k.parameters():
                            param.requires_grad = False

        layers = []
        injection = str(getattr(cfg.gt.graphrope.aw, 'injection', 'all'))
        if injection not in {
            'first', 'second', 'middle-left', 'middle-right', 'penultimate',
            'last', 'first-half', 'last-half', 'all',
        }:
            raise ValueError(
                "gt.graphrope.aw.injection must select one layer "
                "(first/second/middle-left/middle-right/penultimate/last), "
                "one half (first-half/last-half), or all"
            )
        for layer_index in range(cfg.gt.layers):
            active = (
                injection == 'all'
                or (injection == 'first' and layer_index == 0)
                or (injection == 'second' and layer_index == min(1, cfg.gt.layers - 1))
                or (injection == 'middle-left' and layer_index == (cfg.gt.layers - 1) // 2)
                or (injection == 'middle-right' and layer_index == cfg.gt.layers // 2)
                or (injection == 'penultimate' and layer_index == max(0, cfg.gt.layers - 2))
                or (injection == 'last' and layer_index == cfg.gt.layers - 1)
                or (injection == 'first-half' and layer_index < (cfg.gt.layers + 1) // 2)
                or (injection == 'last-half' and layer_index >= cfg.gt.layers // 2)
            )
            layers.append(AttentionLayer(
                                in_dim=cfg.gt.dim_hidden,
                                out_dim=cfg.gt.dim_hidden,
                                num_heads=cfg.gt.n_heads,
                                dropout=cfg.gt.dropout,
                                attn_dropout=cfg.gt.attn_dropout,
                                layer_norm=cfg.gt.layer_norm,
                                batch_norm=cfg.gt.batch_norm,
                                residual=cfg.gt.residual,
                                layer_type=cfg.gt.layer_type,
                                cfg=cfg,
                                shared_omega_q=shared_omega_q,
                                shared_omega_k=shared_omega_k,
                                positional_method_override=positional_method,
                                positional_active=active))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        
        # Apply initial node encoder
        batch = self.encoder(batch)
        # A fresh cache per model call shares fixed LR-AW geometry across GT
        # layers without retaining stale tensors across optimizer steps.
        batch._lr_aw_geometry_cache = {}
        # Exact Holonomy geometry is already precomputed on disk.  This cache
        # only reshapes/pads it once per mini-batch and shares the tensor across
        # all Transformer layers.
        batch._exact_holonomy_geometry_cache = {}
        
        # Apply shared rotational encoder if configured
        if hasattr(self, 'rotational_encoder'):
            batch = self.rotational_encoder(batch)
        
        # Apply attention layers
        batch = self.layers(batch)
        
        # Apply final head
        batch = self.post_mp(batch)
        return batch
