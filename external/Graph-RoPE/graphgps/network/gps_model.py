import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gps_layer import GPSLayer
from graphgps.layer.graphrope import init_omega_matrix, resolve_positional_method
import torch.nn as nn


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
                
        positional_method = resolve_positional_method(
            getattr(cfg.gt.graphrope, 'method', None),
            cfg.gt.graphrope.enable,
        )
        if positional_method == 'wire' and cfg.gt.graphrope.encoder:
            RoPEncoder = register.node_encoder_dict[
                cfg.gt.graphrope.encoder]
            self.rotational_encoder = RoPEncoder(emb_dim=cfg.gt.graphrope.t_dim)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
            
        # Create shared Omega matrices if GraphRoPE is enabled and sharing is requested
        shared_omega_q = None
        shared_omega_k = None
        positional_method = resolve_positional_method(
            getattr(cfg.gt.graphrope, 'method', None),
            cfg.gt.graphrope.enable,
        )
        if global_model_type == "GraphRoPE" and positional_method == 'wire' and hasattr(cfg.gt.graphrope, 'share_omega') and cfg.gt.graphrope.share_omega:
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
            # Create a copy of the graphrope config and add shared matrices
            graphrope_cfg = cfg.gt.graphrope
            if shared_omega_q is not None:
                # We need to add the shared matrices to the config that gets passed to GPSLayer
                # Since cfg objects might be read-only, we'll pass them as separate parameters
                pass
            
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
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
                graphrope_cfg=graphrope_cfg,
                shared_omega_q=shared_omega_q,
                shared_omega_k=shared_omega_k,
                # Instantiate the same positional module in every layer and
                # bypass it where the injection ablation is inactive.  This
                # keeps parameter counts identical across first/half/all.
                graphrope_method_override=positional_method,
                graphrope_positional_active=active,
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        # A fresh cache per model call shares fixed LR-AW geometry across GPS
        # layers without retaining stale tensors across optimizer steps.
        batch._lr_aw_geometry_cache = {}
        for module in self.children():
            batch = module(batch)
        return batch
