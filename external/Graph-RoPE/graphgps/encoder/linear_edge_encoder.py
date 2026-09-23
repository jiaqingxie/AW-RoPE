import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # AW-Prepared uses canonical lowercase dataset identifiers while the
        # original GraphGPS configs use uppercase names.  Both datasets have
        # one continuous scalar per edge.
        if str(cfg.dataset.name).upper() in {'MNIST', 'CIFAR10'}:
            self.in_dim = 1
        else:
            raise ValueError("Input edge feature dim is required to be hardset "
                             "or refactored to use a cfg option.")
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        # Mirror the GIN continuous-input contract instead of relying on the
        # serialized tensor dtype.
        edge_attr = batch.edge_attr.to(dtype=self.encoder.weight.dtype)
        batch.edge_attr = self.encoder(edge_attr.view(-1, self.in_dim))
        return batch
