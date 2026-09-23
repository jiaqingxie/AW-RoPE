import os
import os.path as osp
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx, degree
from torch_geometric.graphgym.config import cfg


class GridRewiringDataset(InMemoryDataset):
    """
    Grid Rewiring Colored Subgraph Diameter Regression Dataset.
    
    Generates synthetic graphs that start from a 2D grid lattice and apply 
    Watts-Strogatz style rewiring. The task is to predict the diameter of
    the largest connected component formed by colored nodes.
    
    Configuration parameters:
    - p: Probability of rewiring each edge (controls small-world vs grid structure)
    - N: Mean for geometric distribution of node counts
    - N_min: Minimum number of nodes
    - N_max: Maximum number of nodes
    - colored_ratio: Fraction of nodes to color (default 0.3)
    
    The dataset creates a graph regression task where:
    - Node features are binary integers (0 or 1, indicating uncolored/colored)
    - Graph labels are the diameter of the largest connected component
      formed by colored nodes
    
    Parameters are read from cfg.dataset.grid_rewiring_* configuration options.
    """

    def __init__(self, root, name='grid_rewiring', transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root (str): Root directory where the dataset should be saved.
            name (str): Dataset name identifier
            transform (callable, optional): A function/transform that takes in a Data object
            pre_transform (callable, optional): A function/transform that takes in a Data object
            pre_filter (callable, optional): A function that takes in a Data object
        """
        self.name = name
        # Store key config params for unique file naming
        self.p = float(getattr(cfg.dataset.grid_rewiring, 'p', 0.0))
        self.N = int(getattr(cfg.dataset.grid_rewiring, 'N', 50))
        # Dimensionality of the lattice (default 2 = 2-D grid)
        self.dim = int(getattr(cfg.dataset.grid_rewiring, 'dim', 2))
        self.colored_ratio = float(getattr(cfg.dataset.grid_rewiring, 'colored_ratio', 0.3))
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _param_str(self):
        # Helper to format key params for filenames/dirs
        p_str = f"p{self.p:.3f}".replace('.', 'p')
        N_str = f"N{self.N}"
        c_str = f"c{self.colored_ratio:.2f}".replace('.', 'p')
        d_str = f"d{self.dim}"
        return f"{p_str}_{N_str}_{c_str}_{d_str}"

    @property
    def raw_dir(self):
        return osp.join(self.root, 'GridRewiring', 'raw')

    @property
    def processed_dir(self):
        # Make processed dir unique for each key param combo
        return osp.join(self.root, f'GridRewiring_{self._param_str()}', 'processed')

    @property
    def raw_file_names(self):
        return []  # No raw files needed as we generate synthetically

    @property
    def processed_file_names(self):
        # Make processed file unique for each key param combo
        return [f'data_{self._param_str()}.pt']

    def download(self):
        pass  # No download needed for synthetic data

    def _create_grid_graph(self, n):
        """
        Create an approximately n-node *d*-dimensional grid graph, where *d* is
        given by ``self.dim``.
        
        For *d* = 2 this reduces to a rectangular lattice as implemented
        previously.  For *d* > 2 we create an axis-aligned hyper-rectangular
        lattice using :pyfunc:`networkx.grid_graph`.
        
        Args:
            n (int): Desired (approximate) number of nodes.
        Returns:
            NetworkX graph: d-dimensional grid graph with ``≈ n`` nodes.
        """
        if self.dim <= 0:
            raise ValueError(f"Grid dimension must be positive, got {self.dim}.")

        if self.dim == 2:
            # Preserve the original 2-D logic so that node counts stay close
            sqrt_n = int(np.sqrt(n))
            best_diff = float('inf')
            best_dims = (sqrt_n, sqrt_n)
            for m in range(max(1, sqrt_n - 2), sqrt_n + 3):
                for k in range(max(1, sqrt_n - 2), sqrt_n + 3):
                    diff = abs(m * k - n)
                    if diff < best_diff:
                        best_diff = diff
                        best_dims = (m, k)
            dims = list(best_dims)
        else:
            # Start with each dimension ≈ n^(1/d)
            base = int(round(n ** (1.0 / self.dim)))
            dims = [max(1, base) for _ in range(self.dim)]
            product = int(np.prod(dims))
            idx = 0
            # Increment dimensions cyclically until we reach at least n nodes
            while product < n:
                dims[idx % self.dim] += 1
                idx += 1
                product = int(np.prod(dims))

        # Build the grid graph. For 2-D we can still use grid_2d_graph for speed.
        if self.dim == 2:
            G = nx.grid_2d_graph(dims[0], dims[1])
        else:
            G = nx.grid_graph(dim=dims)

        # Relabel nodes with consecutive integers for consistency
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

        return G

    def _rewire_graph(self, G, p, seed=None):
        """
        Apply Watts-Strogatz style rewiring to a graph.
        
        Args:
            G: NetworkX graph (starting from grid structure)
            p: Probability of rewiring each edge
            seed: Random seed for reproducibility
            
        Returns:
            G_rewired: Rewired graph
        """
        if seed is not None:
            np.random.seed(seed)
            
        G_rewired = G.copy()
        nodes = list(G_rewired.nodes())
        edges = list(G_rewired.edges())
        
        # Rewire each edge with probability p
        for u, v in edges:
            if np.random.random() < p:
                # Remove the original edge
                G_rewired.remove_edge(u, v)
                
                # Choose a new target node (different from u and not already connected to u)
                possible_targets = [node for node in nodes 
                                  if node != u and not G_rewired.has_edge(u, node)]
                
                if possible_targets:
                    new_target = np.random.choice(possible_targets)
                    G_rewired.add_edge(u, new_target)
                else:
                    # If no valid targets, keep the original edge
                    G_rewired.add_edge(u, v)
        
        return G_rewired

    def _compute_colored_subgraph_diameter(self, G, colored_nodes):
        """
        Compute the diameter of the connected subgraph induced by colored nodes.
        
        Args:
            G: NetworkX graph
            colored_nodes: List of node indices that are colored (feature=1)
            
        Returns:
            max_diameter: Maximum diameter among all connected components of colored nodes
        """
        if len(colored_nodes) == 0:
            return 0
            
        # Create subgraph with only colored nodes
        colored_subgraph = G.subgraph(colored_nodes)
        
        max_diameter = 0
        
        # Check each connected component
        for component in nx.connected_components(colored_subgraph):
            if len(component) == 1:
                component_diameter = 0
            else:
                component_subgraph = colored_subgraph.subgraph(component)
                try:
                    # Compute diameter of this component
                    component_diameter = nx.diameter(component_subgraph)
                except nx.NetworkXError:
                    # Disconnected component (shouldn't happen, but just in case)
                    component_diameter = 0
                    
            max_diameter = max(max_diameter, component_diameter)
            
        return max_diameter

    def _generate_colored_nodes(self, G, colored_ratio):
        """
        Generate coloring for nodes and compute the diameter of the largest colored component.
        
        Args:
            G: NetworkX graph
            colored_ratio: Fraction of nodes to color
            
        Returns:
            colored_nodes: List of colored node indices
            diameter: Actual diameter of the largest colored component
        """
        n = G.number_of_nodes()
        num_colored = max(1, int(n * colored_ratio))
        
        colored_nodes = np.random.choice(n, size=num_colored, replace=False).tolist()
        diameter = self._compute_colored_subgraph_diameter(G, colored_nodes)
        
        return colored_nodes, diameter

    def process(self):
        # Get configuration parameters
        p = cfg.dataset.grid_rewiring.p  # Fixed p value for rewiring
        N = cfg.dataset.grid_rewiring.N  # Mean for geometric distribution
        N_min = cfg.dataset.grid_rewiring.N_min  # Minimum number of nodes
        N_max = cfg.dataset.grid_rewiring.N_max  # Maximum number of nodes
        num_graphs = cfg.dataset.grid_rewiring.num_graphs  # Total number of graphs
        colored_ratio = getattr(cfg.dataset.grid_rewiring, 'colored_ratio', 0.3)  # Fraction of nodes to color
        
        # Calculate geometric distribution probability parameter
        geom_p = 1.0 / N
        
        data_list = []
        graph_idx = 0
        max_attempts = num_graphs * 3  # Safety limit
        
        print(f"Generating {num_graphs} Grid Rewiring graphs for colored subgraph diameter regression...")
        print(f"Colored ratio: {colored_ratio}, Rewiring probability: {p}")
        
        while len(data_list) < num_graphs and graph_idx < max_attempts:
            try:
                # Sample number of nodes from geometric distribution
                np.random.seed(42 + graph_idx)
                n = np.random.geometric(geom_p)
                n = max(N_min, min(N_max, n))
                
                # Generate d-dimensional grid graph and apply rewiring
                G = self._create_grid_graph(n)
                G = self._rewire_graph(G, p, seed=42 + graph_idx)
                
                # Ensure the graph is connected after rewiring
                if not nx.is_connected(G):
                    components = list(nx.connected_components(G))
                    for i in range(len(components) - 1):
                        node1 = np.random.choice(list(components[i]))
                        node2 = np.random.choice(list(components[i + 1]))
                        G.add_edge(node1, node2)
                
                # Update n to actual number of nodes
                n = G.number_of_nodes()
                
                # Generate colored nodes and compute diameter
                colored_nodes, actual_diameter = self._generate_colored_nodes(G, colored_ratio)
                
                # Create binary node features (0 = uncolored, 1 = colored)
                node_features = [[0] for _ in range(n)]
                for node_idx in colored_nodes:
                    node_features[node_idx] = [1]
                    
                data = from_networkx(G)
                data.x = torch.tensor(node_features, dtype=torch.long)
                # Store the target diameter directly (already strictly positive)
                data.y = torch.tensor([float(actual_diameter)], dtype=torch.float)
                data.num_nodes = n
                data_list.append(data)
                
                if (len(data_list)) % 100 == 0:
                    print(f"Generated {len(data_list)}/{num_graphs} graphs")
                    if len(data_list) > 0:
                        diameters = [d.y.item() for d in data_list[-100:]]
                        print(f"Last 100 diameters: min={min(diameters):.1f}, max={max(diameters):.1f}, mean={np.mean(diameters):.1f}")
            
            except Exception as e:
                print(f"Error generating graph {graph_idx}: {e}")
            
            graph_idx += 1

        if len(data_list) == 0:
            raise RuntimeError("No graphs were successfully generated!")

        print(f"Successfully generated {len(data_list)} Grid Rewiring graphs")
        print(f"Colored ratio: {colored_ratio}, Rewiring probability: {p}")
        
        if len(data_list) > 0:
            node_counts = [d.num_nodes for d in data_list]
            print(f"Node count distribution: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}")
            
            diameters = [d.y.item() for d in data_list]
            print(f"Diameter distribution: min={min(diameters):.1f}, max={max(diameters):.1f}, mean={np.mean(diameters):.1f}")

        # Apply pre-filter if provided
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # Create train/val/test splits by graph
        n_total = len(data_list)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        # Shuffle graphs for random split with fixed seed
        torch.manual_seed(42)
        indices = torch.randperm(n_total)
        
        self.split_idxs = [
            indices[:n_train].tolist(),
            indices[n_train:n_train + n_val].tolist(),
            indices[n_train + n_val:].tolist()
        ]
        
        print(f"Created splits: train={len(self.split_idxs[0])}, val={len(self.split_idxs[1])}, test={len(self.split_idxs[2])}")

    def get_idx_split(self):
        """Return train/val/test split indices by graph."""
        if not hasattr(self, 'split_idxs') or self.split_idxs is None:
            # Generate default splits if not available
            n_total = len(self)
            n_train = int(0.7 * n_total)
            n_val = int(0.15 * n_total)
            
            torch.manual_seed(42)
            indices = torch.randperm(n_total)
            
            self.split_idxs = [
                indices[:n_train].tolist(),
                indices[n_train:n_train + n_val].tolist(),
                indices[n_train + n_val:].tolist()
            ]
        
        return {
            'train': torch.tensor(self.split_idxs[0], dtype=torch.long),
            'valid': torch.tensor(self.split_idxs[1], dtype=torch.long),
            'test': torch.tensor(self.split_idxs[2], dtype=torch.long)
        }

    @property
    def num_classes(self):
        """Return number of classes for classification tasks."""
        return 1
    
    @property 
    def num_node_features(self):
        """Return number of node features."""
        return 1
    
    @property
    def num_node_types(self):
        """Return number of node types for TypeDictNodeEncoder."""
        return 2  # 0 = uncolored, 1 = colored

    def __repr__(self):
        try:
            return f'GridRewiringDataset(N={cfg.dataset.grid_rewiring.N}, p={cfg.dataset.grid_rewiring.p}, {cfg.dataset.grid_rewiring.num_graphs})'
        except:
            return f'GridRewiringDataset({len(self)} graphs)' 
