from copy import deepcopy
from pathlib import Path

from tqdm import tqdm
from functools import partial
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from graphgps.encoder.graphormer_encoder import graphormer_pre_processing
import scipy.sparse.linalg # For Lanczos
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence
import os # For potential disk caching
import h5py # For HDF5 storage

def add_spectral_stats(dataset, pe_types, is_undirected, cfg):

    spectral_request_configs = {}  # Key: (lap_norm_str_or_None, use_lanczos_bool), Value: max_freqs_needed
    structural_request_configs = {}  # Key: structural_type, Value: config dict

    spectral_pe_types_lap = ['LapPE', 'LapRoPE', 'EquivStableLapPE', 'SignNet']
    for pe_type in pe_types:
        if pe_type in spectral_pe_types_lap:
            pecfg = getattr(cfg, f"posenc_{pe_type}")
            lap_norm = pecfg.eigen.laplacian_norm.lower()
            if lap_norm == 'none':
                lap_norm = None
            use_lanczos = pecfg.eigen.use_lanczos
            max_f = pecfg.eigen.max_freqs
            key = (lap_norm, use_lanczos)
            current_max_f = spectral_request_configs.get(key, 0)
            spectral_request_configs[key] = max(current_max_f, max_f)

    

    if any(pt in pe_types for pt in ['HKdiagSE', 'HKfullPE']):
        key_hk = (None, False)  # Heat kernels require unnormalized Laplacian, full spectrum.
        spectral_request_configs[key_hk] = 1e9 # max number of nodes across all graphs (hopefully never more nodes than this)
    
    cache_file = Path(dataset.processed_dir) / 'spectral_cache.h5'
    structural_cache_file = Path(dataset.processed_dir) / 'structural_cache.h5'

    # Initialize cache structures
    spectral_cache = {'eigenvals': {}, 'eigenvecs': {}, 'node_counts': {}}
    structural_cache = {}
    
    if cache_file.exists():
        print(f"Loading spectral cache from {cache_file}")
        try:
            with h5py.File(cache_file, 'r') as f:
                # Load from HDF5
                if 'eigenvals' in f:
                    for key_str in f['eigenvals'].keys():
                        key = eval(key_str)  # Convert string back to tuple
                        eigenvals_list = []
                        eigenvecs_list = []
                        node_counts_list = []
                        
                        eigenvals_group = f['eigenvals'][key_str]
                        eigenvecs_group = f['eigenvecs'][key_str]
                        
                        # Check if node_counts group exists (for backward compatibility)
                        if 'node_counts' in f and key_str in f['node_counts']:
                            node_counts_group = f['node_counts'][key_str]
                            has_node_counts = True
                        else:
                            has_node_counts = False
                        
                        for i in range(len(eigenvals_group.keys())):
                            eigenvals_flat = eigenvals_group[str(i)][:]
                            eigenvecs_flat = eigenvecs_group[str(i)][:]
                            
                            if has_node_counts:
                                # New format: reshape using stored node counts and eigenvals length
                                N = int(node_counts_group[str(i)][()])
                                node_counts_list.append(N)
                                eigenvals_list.append(eigenvals_flat)  # eigenvals are already (k,)
                                # Determine k from eigenvals length and reshape eigenvecs accordingly
                                k = len(eigenvals_flat)
                                eigenvecs_list.append(eigenvecs_flat.reshape(N, k))
                            else:
                                # Old format: assume original shapes (for backward compatibility)
                                eigenvals_list.append(eigenvals_flat)
                                eigenvecs_list.append(eigenvecs_flat)
                                # Infer node count from eigenvals length
                                node_counts_list.append(len(eigenvals_flat))
                        
                        spectral_cache['eigenvals'][key] = eigenvals_list
                        spectral_cache['eigenvecs'][key] = eigenvecs_list
                        spectral_cache['node_counts'][key] = node_counts_list
        except (OSError, KeyError) as e:
            print(f"Warning: Could not load cache file {cache_file}: {e}")
            spectral_cache = {'eigenvals': {}, 'eigenvecs': {}, 'node_counts': {}}
 

    # Only save the cache if we add to it
    dirty = False

    def is_cache_sufficient(cache_key, required_max_freqs):
        """Check if the cache has sufficient eigenvectors for the current request."""
        if cache_key not in spectral_cache['eigenvals']:
            return False
        
        # Check if we have any graphs in the cache for this key
        if not spectral_cache['eigenvals'][cache_key]:
            return False
            
        # For each graph, check if we have enough eigenvectors
        for i, eigenvals in enumerate(spectral_cache['eigenvals'][cache_key]):
            num_cached_eigs = len(eigenvals)
            
            # Get the number of nodes for this graph to determine max possible eigenvectors
            if cache_key in spectral_cache['node_counts'] and i < len(spectral_cache['node_counts'][cache_key]):
                num_nodes = spectral_cache['node_counts'][cache_key][i]
                max_possible_eigs = num_nodes
            else:
                # Fallback: assume we need to check against required_max_freqs
                max_possible_eigs = required_max_freqs
            
            # Check if we have sufficient eigenvectors:
            # - If the graph has fewer nodes than requested freqs, we need all N eigenvectors
            # - If the graph has more nodes than requested freqs, we need at least required_max_freqs
            min_required_eigs = min(required_max_freqs, max_possible_eigs)
            
            if num_cached_eigs < min_required_eigs:
                # For Lanczos, we might have requested fewer than N, so we need to check
                # if we have enough. For dense methods, we should have all available.
                return False
        
        return True


    # Check which caches need to be computed or recomputed
    keys_to_compute = []
    for key, max_freqs in spectral_request_configs.items():
        if not is_cache_sufficient(key, max_freqs):
            keys_to_compute.append(key)
            dirty = True
        else:
            lap_norm_val, use_lanczos_val = key
            print(f"Spectral cache for {'Lanczos' if use_lanczos_val else 'full'} {lap_norm_val or 'unnormalized'} Laplacian already exists with sufficient eigenvectors!")

    for key in keys_to_compute:
        max_freqs = spectral_request_configs[key]
        lap_norm_val, use_lanczos_val = key
        
        if key in spectral_cache['eigenvals']:
            print(f"Recomputing spectral cache for {'Lanczos' if use_lanczos_val else 'full'} {lap_norm_val or 'unnormalized'} Laplacian (insufficient eigenvectors in cache)")
        else:
            print(f"Computing spectral cache for {'Lanczos' if use_lanczos_val else 'full'} {lap_norm_val or 'unnormalized'} Laplacian (no cache found)")

        print(dataset)
        
        # Handle different dataset types - some use _data_list, others (like InMemoryDataset) use indexing
        if hasattr(dataset, '_data_list') and dataset._data_list is not None and len(dataset._data_list) > 0:
            items_to_process = dataset._data_list
        else:
            # For InMemoryDataset and similar, access through indexing
            items_to_process = [dataset[i] for i in range(len(dataset))]

        # Extract serializable data for parallel processing
        serializable_data = []
        for data in items_to_process:
            if hasattr(data, 'num_nodes'):
                num_nodes = data.num_nodes
            else:
                num_nodes = data.x.shape[0]
            
            # Convert tensors to numpy for serialization
            edge_index_np = data.edge_index.cpu().numpy()
            
            serializable_data.append({
                'edge_index': edge_index_np,
                'num_nodes': num_nodes
            })

        eigendecompose_parallel_partial = partial(eigendecompose_parallel,
                                                 is_undirected=is_undirected,
                                                 lap_norm=lap_norm_val,
                                                 use_lanczos=use_lanczos_val,
                                                 max_f_needed=max_freqs)

        # Description for progress tracking
        desc = f"Eigendecomposing {key} ("
        if use_lanczos_val:
            desc += f"Lanczos k<min({max_freqs},N)"
        else:
            desc += "NumPy eigh"
        desc += ")"
        
        # Check if parallel processing is enabled
        # Older GraphGPS configs expose this switch as ``parallel_eigen``;
        # newer WIRE preprocessing code renamed it without registering the
        # replacement in every checkout.  Accept both names so the official
        # configuration remains loadable across the supported PyG versions.
        use_parallel = getattr(
            cfg,
            'posenc_parallel_eigen',
            getattr(cfg, 'parallel_eigen', False),
        )
        
        if use_parallel:
            # Use multiprocessing for parallel eigendecomposition
            # Use standard multiprocessing to avoid deadlocks
            num_workers = min(mp.cpu_count(), 8)  # Limit workers to avoid resource conflicts
            chunksize = max(1, len(serializable_data) // (num_workers * 4))
            
            # Set multiprocessing start method to spawn for better isolation
            ctx = mp.get_context('spawn')
            
            with ctx.Pool(processes=num_workers) as pool:
                # Use imap with tqdm for progress tracking
                print(f"Using {num_workers} workers with chunksize {chunksize}")
                results = list(tqdm(
                    pool.imap(eigendecompose_parallel_partial, serializable_data, chunksize=chunksize),
                    total=len(serializable_data),
                    desc=desc
                ))
        else:
            # Use sequential processing
            print("Using sequential processing (single-threaded)")
            results = []
            for data_item in tqdm(serializable_data, desc=desc):
                result = eigendecompose_parallel_partial(data_item)
                results.append(result)

        # Split eigenvalues, eigenvectors, and node counts into separate lists
        spectral_cache['eigenvals'][key] = [result[0] for result in results]
        spectral_cache['eigenvecs'][key] = [result[1] for result in results]
        # Extract node counts from eigenvector shapes
        spectral_cache['node_counts'][key] = [result[1].shape[0] for result in results]

    
    # Handle different dataset types for copying caches back to data objects
    if hasattr(dataset, '_data_list') and dataset._data_list is not None and len(dataset._data_list) > 0:
        # Case 1: Dataset has _data_list (most datasets)
        for i in tqdm(range(len(dataset)), desc="Copying caches to data objects"):
            data_obj = dataset._data_list[i]
            
            # Initialize spectral caches
            if not hasattr(data_obj, 'eigenvals_cache') or not isinstance(data_obj.eigenvals_cache, dict):
                data_obj.eigenvals_cache = {}
            if not hasattr(data_obj, 'eigenvecs_cache') or not isinstance(data_obj.eigenvecs_cache, dict):
                data_obj.eigenvecs_cache = {}
                
            # Copy spectral cache entries
            for key_loop in spectral_request_configs.keys():
                if key_loop in spectral_cache['eigenvals'] and i < len(spectral_cache['eigenvals'][key_loop]):
                    data_obj.eigenvals_cache[key_loop] = spectral_cache['eigenvals'][key_loop][i]
                if key_loop in spectral_cache['eigenvecs'] and i < len(spectral_cache['eigenvecs'][key_loop]):
                    data_obj.eigenvecs_cache[key_loop] = spectral_cache['eigenvecs'][key_loop][i]
            
    else:
        # Case 2: InMemoryDataset and similar - need to create _data_list to make modifications persistent
        print("Converting InMemoryDataset to use _data_list for persistent cache storage")
        data_list = []
        
        for i in tqdm(range(len(dataset)), desc="Copying caches to data objects"):
            data_obj = dataset.get(i)  # Use .get() to avoid any transforms
            
            # Initialize spectral caches
            if not hasattr(data_obj, 'eigenvals_cache') or not isinstance(data_obj.eigenvals_cache, dict):
                data_obj.eigenvals_cache = {}
            if not hasattr(data_obj, 'eigenvecs_cache') or not isinstance(data_obj.eigenvecs_cache, dict):
                data_obj.eigenvecs_cache = {}
                
            # Copy spectral cache entries
            for key_loop in spectral_request_configs.keys():
                if key_loop in spectral_cache['eigenvals'] and i < len(spectral_cache['eigenvals'][key_loop]):
                    data_obj.eigenvals_cache[key_loop] = spectral_cache['eigenvals'][key_loop][i]
                if key_loop in spectral_cache['eigenvecs'] and i < len(spectral_cache['eigenvecs'][key_loop]):
                    data_obj.eigenvecs_cache[key_loop] = spectral_cache['eigenvecs'][key_loop][i]
                        
            data_list.append(data_obj)
        
        # Update the dataset to use _data_list structure
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)

    if dirty:
        if not cfg.save_cache:
            print("Skipping cache save (cfg.save_cache=False)")
            return
        
        print(f"Saving spectral cache to {cache_file}")
        try:
            with h5py.File(cache_file, 'w') as f:
                eigenvals_group = f.create_group('eigenvals')
                eigenvecs_group = f.create_group('eigenvecs')
                node_counts_group = f.create_group('node_counts')
                
                for key, eigenvals_list in spectral_cache['eigenvals'].items():
                    key_str = str(key)  # Convert tuple key to string
                    
                    # Create subgroups for this key
                    eigenvals_subgroup = eigenvals_group.create_group(key_str)
                    eigenvecs_subgroup = eigenvecs_group.create_group(key_str)
                    node_counts_subgroup = node_counts_group.create_group(key_str)
                    
                    # Store each array in the list as a separate dataset
                    for i, (eigenvals_array, eigenvecs_array) in enumerate(zip(eigenvals_list, spectral_cache['eigenvecs'][key])):
                        eigenvals_subgroup.create_dataset(str(i), data=eigenvals_array, compression='gzip')
                        eigenvecs_subgroup.create_dataset(str(i), data=eigenvecs_array.flatten(), compression='gzip')
                        node_counts_subgroup.create_dataset(str(i), data=spectral_cache['node_counts'][key][i])
        except Exception as e:
            print(f"Warning: Could not save cache file {cache_file}: {e}")


def remove_spectral_cache(data):
    if hasattr(data, 'eigenvals_cache'):
        delattr(data, 'eigenvals_cache')
    if hasattr(data, 'eigenvecs_cache'):
        delattr(data, 'eigenvecs_cache')
    if hasattr(data, 'node_counts_cache'):
        delattr(data, 'node_counts_cache')


def eigendecompose_parallel(serializable_data, is_undirected, lap_norm, use_lanczos, max_f_needed):
    """
    Computes an eigen-decomposition for a single graph using serializable data.
    Returns the computed eigenvalues and eigenvectors.
    """
    # Set single threading for numerical libraries to avoid conflicts
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # Set PyTorch to single thread
    torch.set_num_threads(1)
    
    N = serializable_data['num_nodes']
    edge_index_np = serializable_data['edge_index']

    if N == 0: # Handle empty graphs
        return (np.array([]), np.array([]).reshape(0, 0))

    # Convert numpy back to tensor for processing
    edge_index = torch.from_numpy(edge_index_np)

    if is_undirected:
        undir_edge_index = edge_index
    else:
        undir_edge_index = to_undirected(edge_index)

    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization=lap_norm, num_nodes=N)
    )
    
    k = min(max_f_needed, N)
    
    computed_spectrum = None
    # scipy.sparse.linalg.eigsh requires 0 < k < N (N=L.shape[0])
    if use_lanczos and k > 0 and k < N:
        try:
            computed_spectrum = scipy.sparse.linalg.eigsh(L, k=k, which='SM')
        except:
            # Fallback to dense if Lanczos fails to converge or encounters ARPACK errors
            # This can happen for small or difficult matrices
            L_dense = L.toarray()
            evals, evecs = np.linalg.eigh(L_dense)
            # Store all N eigenvectors/eigenvalues since we computed the full decomposition
            indices = np.argsort(evals)
            computed_spectrum = (evals[indices], evecs[:, indices])
            # Note: Consider logging this fallback event.
    else:
        # Fallback to dense eigendecomposition if lanczos is not used,
        # or if k=0, k=N.
        L_dense = L.toarray()
        # Always compute and store full decomposition when using dense method
        evals, evecs = np.linalg.eigh(L_dense)
        indices = np.argsort(evals)
        computed_spectrum = (evals[indices], evecs[:, indices])

    return computed_spectrum[0], computed_spectrum[1]


def eigendecompose(data, is_undirected, lap_norm, use_lanczos, max_f_needed):
    """
    Computes an eigen-decomposition for a single graph.
    Returns the computed eigenvalues and eigenvectors.
    """
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes
    else:
        N = data.x.shape[0]

    if N == 0: # Handle empty graphs
        return (np.array([]), np.array([]).reshape(0, 0))

    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization=lap_norm, num_nodes=N)
    )
    
    k = min(max_f_needed, N)
    
    computed_spectrum = None
    # scipy.sparse.linalg.eigsh requires 0 < k < N (N=L.shape[0])
    if use_lanczos and k > 0 and k < N:
        try:
            computed_spectrum = scipy.sparse.linalg.eigsh(L, k=k, which='SM')
        except:
            # Fallback to dense if Lanczos fails to converge or encounters ARPACK errors
            # This can happen for small or difficult matrices
            L_dense = L.toarray()
            evals, evecs = np.linalg.eigh(L_dense)
            # Store all N eigenvectors/eigenvalues since we computed the full decomposition
            indices = np.argsort(evals)
            computed_spectrum = (evals[indices], evecs[:, indices])
            # Note: Consider logging this fallback event.
    else:
        # Fallback to dense eigendecomposition if lanczos is not used,
        # or if k=0, k=N.
        L_dense = L.toarray()
        # Always compute and store full decomposition when using dense method
        evals, evecs = np.linalg.eigh(L_dense)
        indices = np.argsort(evals)
        computed_spectrum = (evals[indices], evecs[:, indices])

    return computed_spectrum[0], computed_spectrum[1]


def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """Precompute positional encodings for the given graph,
    assuming spectral information is already in data.computed_spectra_cache.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.
    'Graphormer': Computes spatial types and optionally edges along shortest paths.

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    for t in pe_types:
        if t not in ['LapPE', 'LapRoPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE', 'GraphormerBias']:
            raise ValueError(f"Unexpected PE stats selection {t} not in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes.
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    
    if 'LapPE' in pe_types:
        laplacian_norm_type = cfg.posenc_LapPE.eigen.laplacian_norm.lower()
        if laplacian_norm_type == 'none':
            laplacian_norm_type = None
        use_lanczos = cfg.posenc_LapPE.eigen.use_lanczos
    elif 'LapRoPE' in pe_types:
        laplacian_norm_type = cfg.posenc_LapRoPE.eigen.laplacian_norm.lower()
        if laplacian_norm_type == 'none':
            laplacian_norm_type = None
        use_lanczos = cfg.posenc_LapRoPE.eigen.use_lanczos
    elif 'SignNet' in pe_types:
        laplacian_norm_type = cfg.posenc_SignNet.eigen.laplacian_norm.lower()
        if laplacian_norm_type == 'none':
            laplacian_norm_type = None
        use_lanczos = cfg.posenc_SignNet.eigen.use_lanczos
    else:
        raise ValueError(f"Unexpected PE stats selection {pe_types} not supported")
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types or 'LapRoPE' in pe_types or 'SignNet' in pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        evals = data.eigenvals_cache[(laplacian_norm_type, use_lanczos)]
        evects = data.eigenvecs_cache[(laplacian_norm_type, use_lanczos)]

        assert evals.shape[0] == evects.shape[1], f"Number of eigenvalues ({evals.shape[0]}) must match number of eigenvector columns ({evects.shape[1]})"
        assert evects.shape[0] == N, f"Eigenvectors must have {N} rows (one per node), got {evects.shape[0]}"

        if 'LapPE' in pe_types:
            max_freqs=cfg.posenc_LapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_LapPE.eigen.eigvec_norm
        elif 'EquivStableLapPE' in pe_types:  
            max_freqs=cfg.posenc_EquivStableLapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_EquivStableLapPE.eigen.eigvec_norm
        elif 'LapRoPE' in pe_types:
            max_freqs=cfg.posenc_LapRoPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_LapRoPE.eigen.eigvec_norm
        elif 'SignNet' in pe_types:
            max_freqs=cfg.posenc_SignNet.eigen.max_freqs
            eigvec_norm=cfg.posenc_SignNet.eigen.eigvec_norm
        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)

    if 'SignNet' in pe_types:
        signnet_cfg = cfg.posenc_SignNet
            
        norm_type = signnet_cfg.eigen.laplacian_norm.lower()
        if norm_type == 'none':
            norm_type = None
        use_lanczos = signnet_cfg.eigen.use_lanczos
        
        evals_sn = data.eigenvals_cache[(norm_type, use_lanczos)]
        evects_sn = data.eigenvecs_cache[(norm_type, use_lanczos)]
        
        data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
            evals=evals_sn, evects=evects_sn,
            max_freqs=signnet_cfg.eigen.max_freqs,
            eigvec_norm=signnet_cfg.eigen.eigvec_norm)

    # Random Walks.
    if 'RWSE' in pe_types:
        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param.times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing

    # Heat Kernels.
    if 'HKdiagSE' in pe_types or 'HKfullPE' in pe_types:
        # Get the eigenvalues and eigenvectors of the regular Laplacian,
        # if they have not yet been computed for 'eigen'.
        evals_heat = data.eigenvals_cache[(None, False)]
        evects_heat = data.eigenvecs_cache[(None, False)]
        evals_heat = torch.from_numpy(evals_heat)
        evects_heat = torch.from_numpy(evects_heat)

        # Get the full heat kernels.
        if 'HKfullPE' in pe_types:
            # The heat kernels can't be stored in the Data object without
            # additional padding because in PyG's collation of the graphs the
            # sizes of tensors must match except in dimension 0. Do this when
            # the full heat kernels are actually used downstream by an Encoder.
            raise NotImplementedError()
            # heat_kernels, hk_diag = get_heat_kernels(evects_heat, evals_heat,
            #                                   kernel_times=kernel_param.times)
            # data.pestat_HKdiagSE = hk_diag
        # Get heat kernel diagonals in more efficient way.
        if 'HKdiagSE' in pe_types:
            kernel_param = cfg.posenc_HKdiagSE.kernel
            if len(kernel_param.times) == 0:
                raise ValueError("Diffusion times are required for heat kernel")
            hk_diag = get_heat_kernels_diag(evects_heat, evals_heat,
                                            kernel_times=kernel_param.times,
                                            space_dim=0)
            data.pestat_HKdiagSE = hk_diag


    # Electrostatic interaction inspired kernel.
    if 'ElstaticSE' in pe_types:
        elstatic = get_electrostatic_function_encoding(undir_edge_index, N)
        data.pestat_ElstaticSE = elstatic

    if 'GraphormerBias' in pe_types:
        data = graphormer_pre_processing(
            data,
            cfg.posenc_GraphormerBias.num_spatial_types
        )

 

    # Remove caches so we can collate
    remove_spectral_cache(data)
    
    return data


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = evects.shape[0]  # Number of nodes (rows in eigenvector matrix)
    k = len(evals)  # Number of computed eigenvalues/eigenvectors

    # # Validate that we have enough eigenvectors for the request
    # if k < max_freqs:
    #     import warnings
    #     warnings.warn(f"Requested {max_freqs} eigenvectors but only {k} are available. "
    #                  f"Output will be padded with NaN values. Consider recomputing the cache "
    #                  f"with sufficient eigenvectors.", UserWarning)

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if len(evals) < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - len(evals)), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if len(evals) < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - len(evals)), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul,
                          dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(heat_kernels, dim=0)  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes.
    """
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],  # Min of Vi -> j
        electrostatic.max(dim=0)[0],  # Max of Vi -> j
        electrostatic.mean(dim=0),  # Mean of Vi -> j
        electrostatic.std(dim=0),  # Std of Vi -> j
        electrostatic.min(dim=1)[0],  # Min of Vj -> i
        electrostatic.max(dim=0)[0],  # Max of Vj -> i
        electrostatic.mean(dim=1),  # Mean of Vj -> i
        electrostatic.std(dim=1),  # Std of Vj -> i
        (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
        (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
    ], dim=1)

    return green_encoding


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs
