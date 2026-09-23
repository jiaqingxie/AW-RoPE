from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # infer-link parameters (e.g., edge prediction task)
    cfg.dataset.infer_link_label = "None"

    # Grid Rewiring dataset parameters
    cfg.dataset.grid_rewiring = CN()
    cfg.dataset.grid_rewiring.N = 100  # Mean for geometric distribution of node counts
    cfg.dataset.grid_rewiring.p = 0.1  # Fixed rewiring probability
    cfg.dataset.grid_rewiring.N_min = 20  # Minimum number of nodes
    cfg.dataset.grid_rewiring.N_max = 200  # Maximum number of nodes
    cfg.dataset.grid_rewiring.num_graphs = 1000  # Total number of graphs to generate
    cfg.dataset.grid_rewiring.colored_ratio = 0.3  # Fraction of nodes to color
    cfg.dataset.grid_rewiring.dim = 2  # Dimensionality of the base lattice (2 = 2-D grid)

    # GraphCIFAR10 dataset parameters
    cfg.dataset.graph_cifar10 = CN()
    cfg.dataset.graph_cifar10.patch_size = 4  # Size of square patches (must divide 32 evenly: 1,2,4,8,16,32)
    cfg.dataset.graph_cifar10.normalize = True  # Whether to normalize pixel values to [0,1]
    cfg.dataset.graph_cifar10.num_rewire_edges = 0  # Number of random edges to add (0 = no rewiring)
    cfg.dataset.graph_cifar10.store_coords_in_t = False  # Whether to store 2D grid coordinates in batch.t for RoPE use
