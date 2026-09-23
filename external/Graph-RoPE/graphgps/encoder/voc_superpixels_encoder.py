import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

"""
=== Description of the VOCSuperpixels dataset === 
Each graph is a tuple (x, edge_attr, edge_index, y)
Shape of x : [num_nodes, 14]
Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
Shape of edge_index : [2, num_edges]
Shape of y : [num_nodes]
"""

VOC_node_input_dim = 14
# VOC_edge_input_dim = 1 or 2; defined in class VOCEdgeEncoder

@register_node_encoder('VOCNode')
class VOCNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        node_features = batch.x.to(dtype=self.encoder.weight.dtype)
        batch.x = self.encoder(node_features)

        return batch


@register_edge_encoder('VOCEdge')
class VOCEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        # The original config calls the two-feature variant
        # ``edge_wt_region_boundary``.  AW-Prepared addresses the same offline
        # asset as ``pascalvoc-sp``; keep both identifiers on one contract.
        dataset_name = str(cfg.dataset.name).lower()
        VOC_edge_input_dim = 2 if dataset_name in {
            'edge_wt_region_boundary', 'pascalvoc-sp'
        } else 1
        self.encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        edge_features = batch.edge_attr.to(dtype=self.encoder.weight.dtype)
        batch.edge_attr = self.encoder(edge_features)
        return batch


COCO_NODE_INPUT_DIM = 14
COCO_EDGE_INPUT_DIM = 2


class _COCONodeEncoder(torch.nn.Module):
    """COCO-SP encoder with an explicit raw/LoG-normalized contract."""

    def __init__(self, emb_dim, normalize):
        super().__init__()
        self.normalize = normalize
        self.register_buffer('node_x_mean', torch.tensor([
            4.6977347e-01, 4.4679317e-01, 4.0790915e-01, 7.0808627e-02,
            6.8686441e-02, 6.8498217e-02, 6.7777938e-01, 6.5244222e-01,
            6.2096798e-01, 2.7554795e-01, 2.5910738e-01, 2.2901227e-01,
            2.4261935e+02, 2.8985367e+02,
        ]))
        self.register_buffer('node_x_std', torch.tensor([
            2.6218116e-01, 2.5831082e-01, 2.7416739e-01, 5.7440419e-02,
            5.6832556e-02, 5.7100497e-02, 2.5929087e-01, 2.6201612e-01,
            2.7675411e-01, 2.5456995e-01, 2.5140920e-01, 2.6182330e-01,
            1.5152475e+02, 1.7630779e+02,
        ]))
        self.encoder = torch.nn.Linear(COCO_NODE_INPUT_DIM, emb_dim)

    def forward(self, batch):
        features = batch.x.to(dtype=self.encoder.weight.dtype)
        if self.normalize:
            features = (
                features - self.node_x_mean.view(1, -1)
            ) / self.node_x_std.view(1, -1)
        batch.x = self.encoder(features)
        return batch


class _COCOEdgeEncoder(torch.nn.Module):
    """COCO-SP two-channel edge encoder, optionally train-stat normalized."""

    def __init__(self, emb_dim, normalize):
        super().__init__()
        self.normalize = normalize
        self.register_buffer(
            'edge_x_mean', torch.tensor([0.07848548, 43.68736])
        )
        self.register_buffer(
            'edge_x_std', torch.tensor([0.08902349, 28.473562])
        )
        self.encoder = torch.nn.Linear(COCO_EDGE_INPUT_DIM, emb_dim)

    def forward(self, batch):
        features = batch.edge_attr.to(dtype=self.encoder.weight.dtype)
        if self.normalize:
            features = (
                features - self.edge_x_mean.view(1, -1)
            ) / self.edge_x_std.view(1, -1)
        batch.edge_attr = self.encoder(features)
        return batch


@register_node_encoder('COCONodeNorm')
class COCONodeNormEncoder(_COCONodeEncoder):
    def __init__(self, emb_dim):
        super().__init__(emb_dim, normalize=True)


@register_edge_encoder('COCOEdgeNorm')
class COCOEdgeNormEncoder(_COCOEdgeEncoder):
    def __init__(self, emb_dim):
        super().__init__(emb_dim, normalize=True)


@register_node_encoder('COCONodeRaw')
class COCONodeRawEncoder(_COCONodeEncoder):
    def __init__(self, emb_dim):
        super().__init__(emb_dim, normalize=False)


@register_edge_encoder('COCOEdgeRaw')
class COCOEdgeRawEncoder(_COCOEdgeEncoder):
    def __init__(self, emb_dim):
        super().__init__(emb_dim, normalize=False)
