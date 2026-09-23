import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym.config import cfg
from torchvision import datasets


class GraphCIFAR10(InMemoryDataset):
    """
    Graph version of the CIFAR-10 dataset.
    
    Converts CIFAR-10 images into grid graphs where:
    - Each square patch of pixels becomes a node
    - Neighboring patches are connected in a grid structure
    - Node features are the flattened pixel values from each patch (RGB channels)
    - Graph labels are the original CIFAR-10 class labels (0-9)
    - Optional random edge rewiring can be applied
    
    Configuration parameters:
    - patch_size: Size of square patches (e.g., 4 means 4x4 pixel patches)
    - normalize: Whether to normalize pixel values to [0,1]
    - num_rewire_edges: Number of random edges to add (0 = no rewiring)
    - store_coords_in_t: Whether to store 2D grid coordinates in batch.t
    
    The 32x32 CIFAR-10 images are divided into patches. For example:
    - patch_size=4: Creates 8x8=64 nodes per image (32/4=8)
    - patch_size=8: Creates 4x4=16 nodes per image (32/8=4)
    - patch_size=16: Creates 2x2=4 nodes per image (32/16=2)
    - patch_size=32: Creates 1x1=1 node per image (full image)
    
    Each node is connected to its 4-neighbors in the grid (no diagonal connections).
    Node features include all 3 RGB channels, so each node has patch_size^2 * 3 features.
    If num_rewire_edges > 0, additional random edges are added between randomly
    chosen node pairs that are not already connected.
    
    If store_coords_in_t is True, normalized 2D grid coordinates are stored in batch.t
    for use with rotational positional encoding (RoPE). Coordinates range from [0,1]
    and represent the (x,y) position of each patch in the grid layout.
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
        self.name = 'GraphCIFAR10'
        assert split in ['train', 'val', 'test']
        self.split = split
        
        # Get configuration parameters
        self.patch_size = int(getattr(cfg.dataset.graph_cifar10, 'patch_size', 4))
        self.normalize = bool(getattr(cfg.dataset.graph_cifar10, 'normalize', True))
        self.num_rewire_edges = int(getattr(cfg.dataset.graph_cifar10, 'num_rewire_edges', 0))
        self.store_coords_in_t = bool(getattr(cfg.dataset.graph_cifar10, 'store_coords_in_t', False))
        
        # Validate patch size
        if 32 % self.patch_size != 0:
            raise ValueError(f"patch_size {self.patch_size} must divide 32 evenly. "
                           f"Valid options: {[i for i in [1,2,4,8,16,32] if 32 % i == 0]}")
        
        # Validate rewire edges
        if self.num_rewire_edges < 0:
            raise ValueError(f"num_rewire_edges must be non-negative, got {self.num_rewire_edges}")
        
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    def _param_str(self):
        """Helper to format key params for filenames/dirs."""
        norm_str = "norm" if self.normalize else "raw"
        coords_str = "_coords" if self.store_coords_in_t else ""
        return f"patch{self.patch_size}_{norm_str}_rewire{self.num_rewire_edges}{coords_str}"

    @property
    def raw_dir(self):
        return osp.join(self.root, 'GraphCIFAR10', 'raw')

    @property  
    def processed_dir(self):
        # Make processed dir unique for each param combo
        return osp.join(self.root, f'GraphCIFAR10_{self._param_str()}', 'processed')

    @property
    def raw_file_names(self):
        return []  # CIFAR-10 will be downloaded by torchvision

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        """Download CIFAR-10 data using torchvision."""
        # Download train and test sets
        datasets.CIFAR10(self.raw_dir, train=True, download=True)
        datasets.CIFAR10(self.raw_dir, train=False, download=True)

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

    def _create_grid_coordinates(self, grid_height, grid_width):
        """
        Create normalized 2D grid coordinates for each patch/node.
        
        Args:
            grid_height (int): Number of rows in the grid
            grid_width (int): Number of columns in the grid
            
        Returns:
            torch.Tensor: Grid coordinates of shape [num_nodes, 2] with values in [0, 1]
        """
        coordinates = []
        
        for i in range(grid_height):
            for j in range(grid_width):
                # Normalize coordinates to [0, 1] range
                # For a single node (1x1 grid), coordinates will be (0.5, 0.5)
                if grid_height == 1:
                    norm_y = 0.5
                else:
                    norm_y = i / (grid_height - 1)
                    
                if grid_width == 1:
                    norm_x = 0.5
                else:
                    norm_x = j / (grid_width - 1)
                
                coordinates.append([norm_x, norm_y])
        
        return torch.tensor(coordinates, dtype=torch.float32)

    def _image_to_patches(self, image):
        """
        Convert a 32x32x3 image into patches.
        
        Args:
            image (torch.Tensor): Image of shape [3, 32, 32] or [32, 32, 3]
            
        Returns:
            torch.Tensor: Patches of shape [num_patches, patch_size^2 * 3]
        """
        # Ensure image is the right shape - torchvision CIFAR10 returns [3, 32, 32]
        if image.shape == (3, 32, 32):
            # Convert from [3, 32, 32] to [32, 32, 3] for easier patch extraction
            image = image.permute(1, 2, 0)
        
        assert image.shape == (32, 32, 3), f"Expected (32, 32, 3), got {image.shape}"
        
        # Calculate grid dimensions
        grid_size = 32 // self.patch_size
        
        # Extract patches for each channel separately, then concatenate
        patches_per_channel = []
        for c in range(3):  # RGB channels
            channel = image[:, :, c]  # [32, 32]
            # Use unfold to extract patches
            channel_patches = channel.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
            # channel_patches shape: [grid_size, grid_size, patch_size, patch_size]
            channel_patches = channel_patches.contiguous().view(grid_size * grid_size, self.patch_size * self.patch_size)
            patches_per_channel.append(channel_patches)
        
        # Concatenate all channels for each patch
        # patches shape: [num_patches, patch_size^2 * 3]
        patches = torch.cat(patches_per_channel, dim=1)
        
        return patches

    def _create_node_features(self, patches):
        """
        Create node features from patches.
        
        Args:
            patches (torch.Tensor): Patches of shape [num_patches, patch_size^2 * 3]
            
        Returns:
            torch.Tensor: Node features of shape [num_patches, patch_size^2 * 3]
        """
        features = patches.float()
        
        if self.normalize:
            # Normalize to [0, 1]
            features = features / 255.0
        
        return features

    def process(self):
        """Process CIFAR-10 data into graph format."""
        # Load CIFAR-10 data
        if self.split in ['train', 'val']:
            cifar_dataset = datasets.CIFAR10(self.raw_dir, train=True, download=False)
            data, targets = cifar_dataset.data, cifar_dataset.targets
            
            # Convert targets to tensor if it's a list
            if isinstance(targets, list):
                targets = torch.tensor(targets)
            
            # Create train/val split (first 40k for train, last 10k for val)
            if self.split == 'train':
                data, targets = data[:40000], targets[:40000]
            else:  # val
                data, targets = data[40000:], targets[40000:]
        else:  # test
            cifar_dataset = datasets.CIFAR10(self.raw_dir, train=False, download=False)
            data, targets = cifar_dataset.data, cifar_dataset.targets
            
            # Convert targets to tensor if it's a list
            if isinstance(targets, list):
                targets = torch.tensor(targets)

        grid_size = 32 // self.patch_size
        
        data_list = []
        print(f"Processing {len(data)} {self.split} images into graphs...")
        
        for idx in range(len(data)):
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{len(data)} images")
            
            # Get image and label
            image = torch.from_numpy(data[idx])  # Shape: [32, 32, 3]
            label = targets[idx].item() if hasattr(targets[idx], 'item') else targets[idx]
            
            # Convert image to patches
            patches = self._image_to_patches(image)  # Shape: [num_patches, patch_size^2 * 3]
            
            # Create node features
            node_features = self._create_node_features(patches)  # Shape: [num_patches, patch_size^2 * 3]
            
            # Create edge indices for grid connectivity
            edge_index = self._create_grid_edges(grid_size, grid_size)  # Shape: [2, num_edges]
            
            # Add rewiring
            edge_index = self._add_rewire_edges(edge_index, node_features.shape[0])
            
            # Create PyG Data object
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                y=torch.tensor(label, dtype=torch.long),
                num_nodes=node_features.shape[0]
            )
            
            # Add grid coordinates to batch.t if enabled
            if self.store_coords_in_t:
                grid_coords = self._create_grid_coordinates(grid_size, grid_size)
                graph_data.t = grid_coords
            
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
        # For GraphCIFAR10, splits are handled by creating separate dataset instances
        # This method is called by the master loader for compatibility
        # We'll create splits based on standard CIFAR-10 divisions
        
        # Standard CIFAR-10: 50k train (we split to 40k train + 10k val), 10k test
        train_size = 40000
        val_size = 10000
        test_size = 10000
        
        return {
            'train': torch.arange(train_size),
            'valid': torch.arange(val_size), 
            'test': torch.arange(test_size)
        }

    @property
    def num_classes(self):
        """Number of classes in CIFAR-10."""
        return 10

    @property
    def num_node_features(self):
        """Number of node features."""
        return self.patch_size * self.patch_size * 3  # 3 for RGB channels

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'patch_size={self.patch_size}, '
                f'normalize={self.normalize}, '
                f'num_rewire_edges={self.num_rewire_edges}, '
                f'store_coords_in_t={self.store_coords_in_t})') 