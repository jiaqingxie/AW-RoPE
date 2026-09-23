import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym.config import cfg
from torchvision import datasets


class GraphMNIST(InMemoryDataset):
    """
    Graph version of the MNIST dataset.
    
    Converts MNIST images into grid graphs where:
    - Each square patch of pixels becomes a node
    - Neighboring patches are connected in a grid structure
    - Node features are the flattened pixel values from each patch
    - Graph labels are the original MNIST digit labels (0-9)
    - Optional random edge rewiring can be applied
    
    Configuration parameters:
    - patch_size: Size of square patches (e.g., 4 means 4x4 pixel patches)
    - normalize: Whether to normalize pixel values to [0,1]
    - num_rewire_edges: Number of random edges to add (0 = no rewiring)
    
    The 28x28 MNIST images are divided into patches. For example:
    - patch_size=4: Creates 7x7=49 nodes per image (28/4=7)
    - patch_size=7: Creates 4x4=16 nodes per image (28/7=4)
    - patch_size=14: Creates 2x2=4 nodes per image (28/14=2)
    - patch_size=28: Creates 1x1=1 node per image (full image)
    
    Each node is connected to its 4-neighbors in the grid (no diagonal connections).
    If num_rewire_edges > 0, additional random edges are added between randomly
    chosen node pairs that are not already connected.
    """

    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root (str): Root directory where the dataset should be saved
            split (str): Which split to load ('train', 'val', 'test')
            transform (callable, optional): A function/transform that takes in a Data object
            pre_transform (callable, optional): A function/transform that takes in a Data object  
            pre_filter (callable, optional): A function that takes in a Data object
        """
        self.name = 'GraphMNIST'
        assert split in ['train', 'val', 'test']
        self.split = split
        
        # Get configuration parameters
        self.patch_size = int(getattr(cfg.dataset.graph_mnist, 'patch_size', 4))
        self.normalize = bool(getattr(cfg.dataset.graph_mnist, 'normalize', True))
        self.num_rewire_edges = int(getattr(cfg.dataset.graph_mnist, 'num_rewire_edges', 0))
        
        # Validate patch size
        if 28 % self.patch_size != 0:
            raise ValueError(f"patch_size {self.patch_size} must divide 28 evenly. "
                           f"Valid options: {[i for i in [1,2,4,7,14,28] if 28 % i == 0]}")
        
        # Validate rewire edges
        if self.num_rewire_edges < 0:
            raise ValueError(f"num_rewire_edges must be non-negative, got {self.num_rewire_edges}")
        
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    def _param_str(self):
        """Helper to format key params for filenames/dirs."""
        norm_str = "norm" if self.normalize else "raw"
        return f"patch{self.patch_size}_{norm_str}_rewire{self.num_rewire_edges}"

    @property
    def raw_dir(self):
        return osp.join(self.root, 'GraphMNIST', 'raw')

    @property  
    def processed_dir(self):
        # Make processed dir unique for each param combo
        return osp.join(self.root, f'GraphMNIST_{self._param_str()}', 'processed')

    @property
    def raw_file_names(self):
        return []  # MNIST will be downloaded by torchvision

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        """Download MNIST data using torchvision."""
        # Download train and test sets
        datasets.MNIST(self.raw_dir, train=True, download=True)
        datasets.MNIST(self.raw_dir, train=False, download=True)

    def _create_grid_edges(self, grid_height, grid_width):
        """
        Create edge indices for a grid graph.
        
        Args:
            grid_height (int): Number of rows in the grid
            grid_width (int): Number of columns in the grid
            
        Returns:
            torch.Tensor: Edge indices of shape [2, num_edges]
        """
        edges = []
        
        for i in range(grid_height):
            for j in range(grid_width):
                node_idx = i * grid_width + j
                
                # Connect to right neighbor
                if j < grid_width - 1:
                    right_neighbor = i * grid_width + (j + 1)
                    edges.append([node_idx, right_neighbor])
                    edges.append([right_neighbor, node_idx])  # Undirected
                
                # Connect to bottom neighbor  
                if i < grid_height - 1:
                    bottom_neighbor = (i + 1) * grid_width + j
                    edges.append([node_idx, bottom_neighbor])
                    edges.append([bottom_neighbor, node_idx])  # Undirected
        
        if not edges:
            # Single node case
            return torch.empty((2, 0), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _add_rewire_edges(self, edge_index, num_nodes):
        """
        Add random rewiring edges to the existing edge index.
        
        Args:
            edge_index (torch.Tensor): Existing edge indices of shape [2, num_edges]
            num_nodes (int): Total number of nodes in the graph
            
        Returns:
            torch.Tensor: Edge indices with rewired edges added, shape [2, num_edges + 2*num_rewire_edges]
        """
        if self.num_rewire_edges == 0:
            return edge_index
        
        # Convert edge_index to set of tuples for efficient lookup
        existing_edges = set()
        if edge_index.numel() > 0:
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                existing_edges.add((min(u, v), max(u, v)))  # Store as undirected edges
        
        # Generate random edges
        rewire_edges = []
        undirected_edges_added = 0  # Track actual undirected edges added
        attempts = 0
        max_attempts = self.num_rewire_edges * 10  # Prevent infinite loops
        
        while undirected_edges_added < self.num_rewire_edges and attempts < max_attempts:
            # Randomly sample two different nodes
            u = torch.randint(0, num_nodes, (1,)).item()
            v = torch.randint(0, num_nodes, (1,)).item()
            
            if u != v:  # No self-loops
                edge = (min(u, v), max(u, v))
                if edge not in existing_edges:
                    # Add both directions for undirected graph
                    rewire_edges.extend([[u, v], [v, u]])
                    existing_edges.add(edge)
                    undirected_edges_added += 1  # Increment undirected edge count
            
            attempts += 1
        
        if len(rewire_edges) == 0:
            return edge_index
        
        # Convert to tensor and concatenate with existing edges
        rewire_tensor = torch.tensor(rewire_edges, dtype=torch.long).t()
        
        if edge_index.numel() > 0:
            return torch.cat([edge_index, rewire_tensor], dim=1)
        else:
            return rewire_tensor

    def _image_to_patches(self, image):
        """
        Convert a 28x28 image into patches.
        
        Args:
            image (torch.Tensor): Image of shape [28, 28]
            
        Returns:
            torch.Tensor: Patches of shape [num_patches, patch_size^2]
        """
        # Ensure image is the right shape
        assert image.shape == (28, 28), f"Expected (28, 28), got {image.shape}"
        
        # Calculate grid dimensions
        grid_size = 28 // self.patch_size
        
        # Reshape into patches using unfold
        patches = image.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
        # patches shape: [grid_size, grid_size, patch_size, patch_size]
        
        # Flatten each patch and arrange in row-major order
        patches = patches.contiguous().view(grid_size * grid_size, self.patch_size * self.patch_size)
        # patches shape: [num_patches, patch_size^2]
        
        return patches

    def _create_node_features(self, patches):
        """
        Create node features from patches.
        
        Args:
            patches (torch.Tensor): Patches of shape [num_patches, patch_size^2]
            
        Returns:
            torch.Tensor: Node features of shape [num_patches, patch_size^2]
        """
        features = patches.float()
        
        if self.normalize:
            # Normalize to [0, 1]
            features = features / 255.0
        
        return features

    def process(self):
        """Process MNIST data into graph format."""
        # Load MNIST data
        if self.split in ['train', 'val']:
            mnist_dataset = datasets.MNIST(self.raw_dir, train=True, download=False)
            data, targets = mnist_dataset.data, mnist_dataset.targets
            
            # Create train/val split (first 50k for train, last 10k for val)
            if self.split == 'train':
                data, targets = data[:50000], targets[:50000]
            else:  # val
                data, targets = data[50000:], targets[50000:]
        else:  # test
            mnist_dataset = datasets.MNIST(self.raw_dir, train=False, download=False)
            data, targets = mnist_dataset.data, mnist_dataset.targets

        grid_size = 28 // self.patch_size
        
        data_list = []
        print(f"Processing {len(data)} {self.split} images into graphs...")
        
        for idx in range(len(data)):
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{len(data)} images")
            
            # Get image and label
            image = data[idx]  # Shape: [28, 28]
            label = targets[idx].item()
            
            # Convert image to patches
            patches = self._image_to_patches(image)  # Shape: [num_patches, patch_size^2]
            
            # Create node features
            node_features = self._create_node_features(patches)  # Shape: [num_patches, patch_size^2]
            
            # Create edge indices for grid connectivity
            edge_index = self._create_grid_edges(grid_size, grid_size)  # Shape: [2, num_edges]
            
            # Add rewiring edges
            edge_index = self._add_rewire_edges(edge_index, grid_size * grid_size)  # Shape: [2, num_edges + 2*num_rewire_edges]
            
            # Create PyG Data object
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                y=torch.tensor(label, dtype=torch.long),
                num_nodes=node_features.shape[0]
            )
            
            if self.pre_filter is not None and not self.pre_filter(graph_data):
                continue

            if self.pre_transform is not None:
                graph_data = self.pre_transform(graph_data)

            data_list.append(graph_data)

        print(f"Finished processing {len(data_list)} graphs for {self.split} split")
        
        # Save processed data
        torch.save(self.collate(data_list), 
                   osp.join(self.processed_dir, f'{self.split}.pt'))

    def get_idx_split(self):
        """Return split indices for train/val/test."""
        # For GraphMNIST, splits are handled by creating separate dataset instances
        # This method is called by the master loader for compatibility
        # We'll create splits based on standard MNIST divisions
        
        # Standard MNIST: 60k train (we split to 50k train + 10k val), 10k test
        train_size = 50000
        val_size = 10000
        test_size = 10000
        
        return {
            'train': torch.arange(train_size),
            'valid': torch.arange(val_size), 
            'test': torch.arange(test_size)
        }

    @property
    def num_classes(self):
        """Number of classes in MNIST (digits 0-9)."""
        return 10

    @property
    def num_node_features(self):
        """Number of node features."""
        return self.patch_size * self.patch_size

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'patch_size={self.patch_size}, '
                f'normalize={self.normalize}, '
                f'num_rewire_edges={self.num_rewire_edges})')