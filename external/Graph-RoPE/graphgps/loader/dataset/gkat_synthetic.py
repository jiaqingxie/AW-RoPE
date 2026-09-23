import os
import os.path as osp
import pickle
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import from_networkx, degree


def compute_neighbor_degree_features(data, feature_dim=5):
    """
    Compute node features based on top-k neighbor degrees as described in GKAT paper.
    
    For each node, the feature vector contains the top-k degrees of its neighbors,
    sorted in descending order. If a node has fewer than k neighbors, 
    we pad with zeros.
    
    Args:
        data: PyG Data object
        feature_dim: dimension of feature vector (k=5 in GKAT paper)
    
    Returns:
        torch.Tensor: node features of shape [num_nodes, feature_dim]
    """    
    # Compute degree for each node
    edge_index = data.edge_index
    degrees = degree(edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
    
    # Initialize feature matrix
    num_nodes = data.num_nodes
    features = torch.zeros(num_nodes, feature_dim, dtype=torch.float)
    
    # For each node, collect neighbor degrees
    for node_id in range(num_nodes):
        # Find neighbors of this node
        neighbor_mask = edge_index[0] == node_id
        neighbors = edge_index[1][neighbor_mask]
        
        if len(neighbors) > 0:
            # Get degrees of neighbors
            neighbor_degrees = degrees[neighbors]
            
            # Sort in descending order
            neighbor_degrees_sorted, _ = torch.sort(neighbor_degrees, descending=True)
            
            # Take top-k (or all if fewer than k)
            k = min(len(neighbor_degrees_sorted), feature_dim)
            features[node_id, :k] = neighbor_degrees_sorted[:k]
            # The rest remain as zeros (already initialized)
    
    return features


class GKATSyntheticDataset(InMemoryDataset):
    """
    GKAT Synthetic Motif Detection Datasets.
    
    Five binary graph classification datasets for motif detection:
    - Cycle: Detecting cycle motifs in Erdős-Rényi graphs
    - Grid: Detecting grid motifs in Erdős-Rényi graphs  
    - Ladder: Detecting ladder motifs in Erdős-Rényi graphs
    - CircularLadder: Detecting circular ladder motifs in Erdős-Rényi graphs
    - Caveman: Detecting caveman motifs in Erdős-Rényi graphs
    
    Each dataset contains graphs where the task is to classify whether
    a specific motif pattern is present (label=1) or absent (label=0).
    
    Node features: Each node has a 5-dimensional feature vector containing
    the top-5 degrees of its neighbors, sorted in descending order.
    If a node has fewer than 5 neighbors, the vector is padded with zeros.
    
    From the GKAT paper: "From block-Toeplitz matrices to differential equations 
    on graphs: towards a general theory for scalable masked Transformers"
    """

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root (str): Root directory where the dataset should be saved.
            name (str): One of 'Cycle', 'Grid', 'Ladder', 'CircularLadder', 'Caveman'
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk.
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset.
        """
        self.name = name
        assert name in ['Cycle', 'Grid', 'Ladder', 'CircularLadder', 'Caveman'], \
            f"Unknown dataset name: {name}. Must be one of: Cycle, Grid, Ladder, CircularLadder, Caveman"
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['train_graphs.pkl', 'train_labels.npy', 'val_graphs.pkl', 'val_labels.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Note: The GKAT datasets should be manually placed in the raw directory
        # as they are not available for automatic download
        for filename in self.raw_file_names:
            filepath = osp.join(self.raw_dir, filename)
            if not osp.exists(filepath):
                raise FileNotFoundError(
                    f"GKAT dataset file {filename} not found in {self.raw_dir}. "
                    f"Please manually place the GKAT dataset files in the raw directory. "
                    f"Expected files: {self.raw_file_names}"
                )

    def process(self):
        # Load train data
        with open(osp.join(self.raw_dir, 'train_graphs.pkl'), 'rb') as f:
            train_graphs = pickle.load(f)
        train_labels = np.load(osp.join(self.raw_dir, 'train_labels.npy'))
        
        # Load validation data
        with open(osp.join(self.raw_dir, 'val_graphs.pkl'), 'rb') as f:
            val_graphs = pickle.load(f)
        val_labels = np.load(osp.join(self.raw_dir, 'val_labels.npy'))
        
        # Combine train and val data
        all_graphs = train_graphs + val_graphs
        all_labels = np.concatenate([train_labels, val_labels])
        
        # Convert NetworkX graphs to PyTorch Geometric Data objects
        data_list = []
        for i, (graph, label) in enumerate(zip(all_graphs, all_labels)):
            # Convert NetworkX graph to PyG Data
            data = from_networkx(graph)
            
            # Add proper GKAT node features: top-5 neighbor degrees
            data.x = compute_neighbor_degree_features(data, feature_dim=5)
            
            # Add graph-level label
            data.y = torch.tensor([int(label)], dtype=torch.long)
            
            data_list.append(data)
        
        # Apply pre-filter if provided
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # Store split indices for later use
        n_train = len(train_graphs)
        n_val = len(val_graphs)
        self.split_idxs = [
            list(range(n_train)),                    # train indices
            list(range(n_train, n_train + n_val)),   # val indices  
            list(range(n_train, n_train + n_val))    # use val data as test split
        ]

    def get_idx_split(self):
        """Return train/val split indices."""
        if not hasattr(self, 'split_idxs'):
            # Reconstruct split indices based on original data sizes
            # Train: 1536, Val: 512 for each dataset
            n_train = 1536
            n_val = 512
            self.split_idxs = [
                list(range(n_train)),
                list(range(n_train, n_train + n_val)),
                list(range(n_train, n_train + n_val))
            ]
        
        return {
            'train': torch.tensor(self.split_idxs[0], dtype=torch.long),
            'valid': torch.tensor(self.split_idxs[1], dtype=torch.long),
            'test': torch.tensor(self.split_idxs[2], dtype=torch.long) if self.split_idxs[2] else torch.tensor([], dtype=torch.long)
        }

    @property
    def num_classes(self):
        return 2  # Binary classification

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name}, {len(self)})' 