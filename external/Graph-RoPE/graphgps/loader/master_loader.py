import logging
import os.path as osp
import time
from functools import partial

import numpy as np
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import (Actor, GNNBenchmarkDataset, HeterophilousGraphDataset,
                                      LRGBDataset, MalNetTiny as PyGMalNetTiny,
                                      Planetoid, TUDataset, WebKB, WikipediaNetwork, ZINC, QM9)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader

from graphgps.loader.dataset.aqsol_molecules import AQSOL
from graphgps.loader.dataset.coco_superpixels import COCOSuperpixels
from graphgps.loader.dataset.gkat_synthetic import GKATSyntheticDataset
from graphgps.loader.dataset.graph_cifar10 import GraphCIFAR10
from graphgps.loader.dataset.malnet_tiny import MalNetTiny
from graphgps.loader.dataset.voc_superpixels import VOCSuperpixels
from graphgps.loader.dataset.grid_rewiring_synthetic import GridRewiringDataset
from graphgps.loader.dataset.wire_synthetic import WIRESyntheticDataset
from graphgps.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from graphgps.transform.posenc_stats import add_spectral_stats, compute_posenc_stats
from graphgps.transform.task_preprocessing import task_specific_preprocessing
from graphgps.transform.transforms import (pre_transform_in_memory,
                                           typecast_x, concat_x_and_pos,
                                           clip_graphs_to_size)


def _attach_precomputed_exact_holonomy(dataset, sidecar_path):
    """Attach a complete static Holonomy transport to an in-memory dataset."""
    sidecar_path = osp.abspath(sidecar_path)
    marker_path = sidecar_path + '.done.json'
    if not osp.isfile(sidecar_path) or not osp.isfile(marker_path):
        raise FileNotFoundError(
            f"static exact Holonomy sidecar is missing: {sidecar_path}; "
            "run scripts/precompute_static_exact_holonomy.py before training"
        )
    payload = torch.load(
        sidecar_path, map_location='cpu', mmap=True, weights_only=True
    )
    if payload.get('protocol') != 'precomputed-exact-holonomy-transport-v1':
        raise RuntimeError(f"unexpected exact Holonomy sidecar protocol: {sidecar_path}")
    exact_cfg = cfg.gt.graphrope.aw.exact
    expected_field = str(exact_cfg.field_protocol)
    if payload.get('field_protocol') != expected_field:
        raise RuntimeError(
            f"exact Holonomy field {payload.get('field_protocol')!r} != {expected_field!r}"
        )
    if int(payload.get('num_frequencies', -1)) != cfg.gt.dim_hidden // cfg.gt.n_heads // 2:
        raise RuntimeError("exact Holonomy sidecar frequency count does not match head width")
    if abs(float(payload.get('diffusion_time', -1)) - float(exact_cfg.diffusion_time)) > 1e-12:
        raise RuntimeError("exact Holonomy sidecar diffusion time does not match config")
    node_slices = dataset.slices.get('x')
    if node_slices is None:
        raise RuntimeError('exact Holonomy sidecar requires dataset.x graph boundaries')
    node_counts = (node_slices[1:] - node_slices[:-1]).cpu().to(torch.int64)
    stored_counts = payload.get('node_counts')
    if not isinstance(stored_counts, torch.Tensor) or not torch.equal(node_counts, stored_counts):
        raise RuntimeError("exact Holonomy sidecar graph/node boundaries do not match dataset")
    transport = payload.get('transport')
    transport_slices = payload.get('transport_slices')
    expected_rows = int(node_counts.square().sum())
    expected_frequencies = int(payload['num_frequencies'])
    if not torch.is_complex(transport) or tuple(transport.shape) != (
        expected_rows, expected_frequencies
    ):
        raise RuntimeError("exact Holonomy sidecar transport has an invalid shape or dtype")
    if (
        not isinstance(transport_slices, torch.Tensor)
        or transport_slices.shape != node_slices.shape
        or int(transport_slices[-1]) != expected_rows
    ):
        raise RuntimeError("exact Holonomy sidecar slices are invalid")
    dataset._data.exact_holonomy_transport = transport
    dataset.slices['exact_holonomy_transport'] = transport_slices
    dataset._data_list = None
    dataset._exact_holonomy_sidecar = sidecar_path
    logging.info(
        "Loaded static exact Holonomy transport (%s, t=%s) from %s",
        payload['field_protocol'], payload['diffusion_time'], sidecar_path,
    )
    return dataset


def _attach_precomputed_static_aw_field(dataset, sidecar_path):
    """Attach the fixed ``O(E)`` connection field used by AW-static."""
    sidecar_path = osp.abspath(sidecar_path)
    marker_path = sidecar_path + '.done.json'
    if not osp.isfile(sidecar_path) or not osp.isfile(marker_path):
        raise FileNotFoundError(
            f"static AW field sidecar is missing: {sidecar_path}; "
            "run scripts/precompute_static_aw_field.py before training"
        )
    payload = torch.load(
        sidecar_path, map_location='cpu', mmap=True, weights_only=True
    )
    if payload.get('protocol') != 'precomputed-static-holonomy-edge-field-v1':
        raise RuntimeError(f"unexpected static AW sidecar protocol: {sidecar_path}")
    fixed_cfg = cfg.gt.graphrope.aw.fixed
    if payload.get('field_protocol') != str(fixed_cfg.field_protocol):
        raise RuntimeError("static AW field protocol does not match config")
    if payload.get('variant') != str(fixed_cfg.variant):
        raise RuntimeError("static AW field variant does not match config")
    dataset_edge_slices = dataset.slices.get('edge_index')
    stored_edge_slices = payload.get('edge_slices')
    if (
        dataset_edge_slices is None
        or not isinstance(stored_edge_slices, torch.Tensor)
        or not torch.equal(
            dataset_edge_slices.cpu().to(torch.int64),
            stored_edge_slices.cpu().to(torch.int64),
        )
    ):
        raise RuntimeError("static AW sidecar edge boundaries do not match dataset")
    displacement = payload.get('edge_displacement')
    if (
        not isinstance(displacement, torch.Tensor)
        or displacement.ndim != 1
        or displacement.numel() != int(stored_edge_slices[-1])
        or not torch.is_floating_point(displacement)
    ):
        raise RuntimeError("static AW sidecar displacement is invalid")
    dataset._data.aw_static_edge_displacement = displacement
    dataset.slices['aw_static_edge_displacement'] = stored_edge_slices
    dataset._data_list = None
    dataset._aw_static_field_sidecar = sidecar_path
    logging.info(
        "Loaded static AW field (%s, %s) from %s",
        payload['field_protocol'], payload['variant'], sidecar_path,
    )
    return dataset


def _attach_precomputed_laprope(dataset, sidecar_path):
    """Attach a validated flat LapRoPE sidecar without recomputing spectra."""
    sidecar_path = osp.abspath(sidecar_path)
    marker_path = sidecar_path + '.done.json'
    if not osp.isfile(sidecar_path) or not osp.isfile(marker_path):
        raise FileNotFoundError(
            f"prepared GraphRoPE spectra are missing: {sidecar_path}"
        )
    payload = torch.load(
        sidecar_path, map_location='cpu', mmap=True, weights_only=True
    )
    if payload.get('protocol') != 'lrgb-graphrope-laprope-sidecar-v1':
        raise RuntimeError(f"unexpected LapRoPE sidecar protocol: {sidecar_path}")
    node_slices = dataset.slices.get('x')
    if node_slices is None:
        raise RuntimeError('LapRoPE sidecar requires node slices from dataset.x')
    stored_slices = payload.get('node_slices')
    if (
        not isinstance(stored_slices, torch.Tensor)
        or not torch.equal(node_slices.cpu().to(torch.int64), stored_slices)
    ):
        raise RuntimeError(
            f"LapRoPE sidecar graph/node boundaries mismatch: {sidecar_path}"
        )
    eigvecs = payload.get('eigvecs')
    graph_eigvals = payload.get('eigvals')
    max_freqs = int(payload.get('max_freqs', -1))
    graph_count = len(node_slices) - 1
    node_count = int(node_slices[-1])
    if tuple(eigvecs.shape) != (node_count, max_freqs):
        raise RuntimeError(f"LapRoPE eigenvector shape mismatch: {sidecar_path}")
    if tuple(graph_eigvals.shape) != (graph_count, max_freqs):
        raise RuntimeError(f"LapRoPE eigenvalue shape mismatch: {sidecar_path}")
    if max_freqs != int(cfg.posenc_LapRoPE.eigen.max_freqs):
        raise RuntimeError(
            f"LapRoPE sidecar has {max_freqs} frequencies but config requests "
            f"{cfg.posenc_LapRoPE.eigen.max_freqs}"
        )
    node_counts = node_slices[1:] - node_slices[:-1]
    eigvals = torch.repeat_interleave(
        graph_eigvals, node_counts.cpu().to(torch.int64), dim=0
    ).unsqueeze(2)
    dataset._data.EigVecs = eigvecs
    dataset._data.EigVals = eigvals
    dataset.slices['EigVecs'] = node_slices
    dataset.slices['EigVals'] = node_slices
    # join_dataset_splits keeps a graph-level cache.  Clear it after updating
    # the collated storage so subsequent get(i) calls slice the sidecar-backed
    # tensors instead of returning stale Data objects without EigVals/EigVecs.
    dataset._data_list = None
    dataset._aw_precomputed_posenc = {'LapRoPE'}
    logging.info(f"Loaded prepared LapRoPE spectra from {sidecar_path}")
    return dataset


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    # Show distribution of graph sizes.
    # In-memory LRGB datasets already store exact graph boundaries.  Reading
    # those boundaries avoids materializing every graph merely to print a
    # histogram (over half a million Python objects for PCQM-Contact).
    node_slices = getattr(dataset, 'slices', {}).get('x')
    if isinstance(node_slices, torch.Tensor):
        graph_sizes = (node_slices[1:] - node_slices[:-1]).cpu().numpy()
    else:
        graph_sizes = np.asarray([
            d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
            for d in dataset
        ])
    hist, bin_edges = np.histogram(graph_sizes, bins=10)
    logging.info(f'   Graph size distribution:')
    logging.info(f'     mean: {np.mean(graph_sizes)}')
    for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        logging.info(
            f'     bin {i}: [{start:.2f}, {end:.2f}]: '
            f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
        )


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format == 'WIRE-Synthetic':
        dataset = WIRESyntheticDataset(dataset_dir, name)

    elif format == 'AW-Prepared':
        dataset = preformat_AW_Prepared(dataset_dir, name)

    elif format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'Actor':
            if name != 'none':
                raise ValueError(f"Actor class provides only one dataset.")
            dataset = Actor(dataset_dir)

        elif pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'HeterophilousGraphDataset':
            dataset = preformat_HeterophilousGraphDataset(dataset_dir, name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'WebKB':
            dataset = WebKB(dataset_dir, name)

        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError(f"crocodile not implemented")
            dataset = WikipediaNetwork(dataset_dir, name,
                                       geom_gcn_preprocess=True)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)
            
        elif pyg_dataset_id == 'AQSOL':
            dataset = preformat_AQSOL(dataset_dir)

        elif pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'GKATSynthetic':
            dataset = preformat_GKATSynthetic(dataset_dir, name)

        elif pyg_dataset_id == 'GridRewiring':
            dataset = preformat_GridRewiring(dataset_dir, name)

        elif pyg_dataset_id == 'GraphCIFAR10':
            dataset = preformat_GraphCIFAR10(dataset_dir, name)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg'):
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))

        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)

        elif name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")
    else:
        raise ValueError(f"Unknown data format: {format}")

    if cfg.posenc_LapRoPE.enable:
        if format == 'AW-Prepared' and str(name).lower() == 'coco-sp':
            dataset = _attach_precomputed_laprope(
                dataset,
                osp.join(dataset_dir, 'graphrope_spectra',
                         'coco-sp-laprope-k10.pt'),
            )
        elif format == 'OGB' and name == 'PCQM4Mv2Contact-shuffle':
            dataset = _attach_precomputed_laprope(
                dataset,
                osp.join(osp.dirname(dataset_dir), 'graphrope_spectra',
                         'pcqm-contact-laprope-k10.pt'),
            )

    pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))

    log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    precomputed_pe = set(getattr(dataset, '_aw_precomputed_posenc', set()))
    if pe_enabled_list and set(pe_enabled_list).issubset(precomputed_pe):
        logging.info(
            f"Using prepared positional encodings: {sorted(precomputed_pe)}"
        )
    elif pe_enabled_list:
        # Estimate directedness based on 10 graphs to save time.
        # Note: dataset may have been modified by task_specific_preprocessing, re-evaluate directedness if critical.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  PE processing: estimated to be undirected: {is_undirected}")

        # Stage 1: Precompute and cache spectral statistics (eigen-decompositions)
        # This will use disk caching if dataset.processed_dir is available and passed.
        s_time_spec = time.perf_counter()
        logging.info(f"Precomputing and caching spectral statistics (e.g. Eigendecompositions) for PEs: {pe_enabled_list}...")
        
        add_spectral_stats(dataset, pe_enabled_list, is_undirected, cfg)

        elapsed_spec = time.perf_counter() - s_time_spec
        logging.info(f"Spectral statistics computation/caching done! Took {time.strftime('%H:%M:%S', time.gmtime(elapsed_spec)) + f'{elapsed_spec:.2f}'[-3:]}")

        # for i in range(10):
        #     print(dataset[i])

        # Stage 2: Compute actual PE values (e.g., LapPE EigVals, RWSE stats) using cached spectra.
        s_time_pe = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding values (e.g., EigVals, RWSE) for PEs: {pe_enabled_list}...")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - s_time_pe # This elapsed is for the second stage only
        overall_elapsed = time.perf_counter() - s_time_spec # Overall time for both stages
        timestr_pe = time.strftime('%H:%M:%S', time.gmtime(elapsed)) + f'{elapsed:.2f}'[-3:]
        timestr_overall = time.strftime('%H:%M:%S', time.gmtime(overall_elapsed)) + f'{overall_elapsed:.2f}'[-3:]
        logging.info(f"Positional Encoding value computation done! Took {timestr_pe}")
        logging.info(f"Total PE preprocessing time (spectral + values): {timestr_overall}")

    graphrope_method = str(getattr(cfg.gt.graphrope, 'method', '')).strip().lower()
    if (
        graphrope_method.startswith('aw')
        and str(cfg.gt.graphrope.aw.field_type) == 'precomputed-static'
    ):
        fixed_cfg = cfg.gt.graphrope.aw.fixed
        if not str(fixed_cfg.cache_path).strip():
            raise RuntimeError(
                'gt.graphrope.aw.fixed.cache_path is required for AW-static'
            )
        dataset = _attach_precomputed_static_aw_field(
            dataset, str(fixed_cfg.cache_path)
        )
    # Limit dataset size for testing/overfitting if NUM_SAMPLES is set
    # dataset = limit_dataset_size(dataset, NUM_SAMPLES)

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        # Convert split indices to tensors if they are lists
        split_idxs = dataset.split_idxs
        if isinstance(split_idxs[0], list):
            split_idxs = [torch.tensor(split_idx, dtype=torch.long) for split_idx in split_idxs]
        set_dataset_splits(dataset, split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Attach the dense exact transport only after every transform and split
    # mutation has finished.  GraphGym's split helpers access
    # ``InMemoryDataset.data`` and may invalidate PyG's materialized item
    # cache.  Attaching earlier therefore allowed the sidecar to be logged as
    # loaded while the first training mini-batch no longer contained it.
    if graphrope_method in {'exact', 'full-holonomy', 'exact-holonomy', 'holonomy-rope'}:
        exact_cfg = cfg.gt.graphrope.aw.exact
        if bool(exact_cfg.precomputed):
            if not str(exact_cfg.cache_path).strip():
                raise RuntimeError(
                    'gt.graphrope.aw.exact.cache_path is required for precomputed exact Holonomy-RoPE'
                )
            dataset = _attach_precomputed_exact_holonomy(
                dataset, str(exact_cfg.cache_path)
            )
            sample = dataset.get(0)
            if not hasattr(sample, 'exact_holonomy_transport'):
                raise RuntimeError(
                    'exact Holonomy sidecar did not survive GraphGym dataset finalization'
                )
    
    # Special handling for WattsStrogatz and GridRewiring datasets to fix PyTorch Geometric slicing issues
    # The set_dataset_splits function adds graph index attributes that are incompatible
    # with PyG's slicing mechanism for InMemoryDataset, so we remove their SLICES but keep the attributes
    if format == 'PyG-GridRewiring':
        problematic_attrs = ['train_graph_index', 'val_graph_index', 'test_graph_index']
        for attr in problematic_attrs:
            # Only remove the slices, keep the data attributes as they're needed for training
            if attr in dataset.slices:
                del dataset.slices[attr]
    
    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])
        # print(f"Indegrees: {cfg.gt.pna_degrees}")
        # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")


    return dataset


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    elif name == "PATTERN":
        # Preserve the official Graph-RoPE/WIRE preprocessing contract.  The
        # shared LapPE input encoder converts these one-hot values to its
        # floating-point weight dtype immediately before the linear layer.
        tf_list = [partial(typecast_x, type_str='long')]
    elif name in ['CLUSTER', 'CSL', 'DD']:
        tf_list = []
    else:
        raise ValueError(f"Loading dataset '{name}' from "
                         f"GNNBenchmarkDataset is not supported.")

    if name in ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']:
        dataset = join_dataset_splits(
            [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
            for split in ['train', 'val', 'test']]
        )
        pre_transform_in_memory(dataset, T.Compose(tf_list))
    elif name == 'CSL':
        dataset = GNNBenchmarkDataset(root=dataset_dir, name=name)

    return dataset


class AWPreparedLRGBDataset(LRGBDataset):
    """LRGBDataset that is forbidden from downloading in GPU workloads."""

    def download(self):
        required = [
            osp.join(self.processed_dir, f'{split}.pt')
            for split in ('train', 'val', 'test')
        ]
        missing = [path for path in required if not osp.isfile(path)]
        if missing:
            raise FileNotFoundError(
                'prepared LRGB dataset is incomplete; downloads are disabled: '
                + ', '.join(missing)
            )


def preformat_AW_Prepared(dataset_dir, name):
    """Load the project's already prepared benchmark assets without download.

    The AW-ROPE project deliberately stores each public source under a stable
    source-specific directory instead of GraphGPS's historical directory
    names.  This adapter keeps the official GraphGPS model/training stack but
    points it at those local processed files.  A missing processed asset fails
    before the GPU trial starts; no dataset class is allowed to download.
    """
    root = osp.abspath(dataset_dir)
    key = str(name).lower()
    gnn_names = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR10',
        'pattern': 'PATTERN',
        'cluster': 'CLUSTER',
    }
    if key in gnn_names:
        local_root = osp.join(root, 'gnn_benchmark')
        required = osp.join(local_root, gnn_names[key], 'processed', 'train_data.pt')
        if not osp.isfile(required):
            raise FileNotFoundError(f"prepared dataset is missing: {required}")
        return preformat_GNNBenchmarkDataset(local_root, gnn_names[key])

    lrgb_names = {
        'peptides-func': 'Peptides-func',
        'peptides-struct': 'Peptides-struct',
        'pascalvoc-sp': 'PascalVOC-SP',
        'coco-sp': 'COCO-SP',
    }
    if key in lrgb_names:
        local_root = osp.join(root, 'lrgb')
        local_dir = {
            'peptides-func': 'peptides-func',
            'peptides-struct': 'peptides-struct',
            'pascalvoc-sp': 'pascalvoc-sp',
            'coco-sp': 'coco-sp',
        }[key]
        required = osp.join(local_root, local_dir, 'processed', 'train.pt')
        if not osp.isfile(required):
            raise FileNotFoundError(f"prepared dataset is missing: {required}")
        return join_dataset_splits([
            AWPreparedLRGBDataset(
                root=local_root, name=lrgb_names[key], split=split
            )
            for split in ('train', 'val', 'test')
        ])

    if key == 'malnet-tiny':
        local_root = osp.join(root, 'malnet_tiny')
        required = osp.join(local_root, 'processed', 'data.pt')
        if not osp.isfile(required):
            raise FileNotFoundError(f"prepared dataset is missing: {required}")
        dataset = join_dataset_splits([
            PyGMalNetTiny(root=local_root, split=split)
            for split in ('train', 'val', 'test')
        ])
        pre_transform_in_memory(dataset, T.LocalDegreeProfile())
        return dataset

    raise ValueError(f"unknown AW-Prepared dataset: {name}")


def preformat_HeterophilousGraphDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's HeterophilousGraphDataset.
    
    The heterophilous graphs from "A Critical Look at the Evaluation of GNNs 
    under Heterophily: Are We Really Making Progress?" paper.
    
    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset ('Roman-empire', 'Amazon-ratings', 
              'Minesweeper', 'Tolokers', 'Questions')
    
    Returns:
        PyG dataset object
    """
    valid_names = ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']
    if name not in valid_names:
        raise ValueError(f"Unsupported HeterophilousGraphDataset name: {name}. "
                         f"Must be one of: {valid_names}")
    
    dataset = HeterophilousGraphDataset(root=dataset_dir, name=name)
    return dataset


def preformat_MalNetTiny(dataset_dir, feature_set):
    """Load and preformat Tiny version (5k graphs) of MalNet

    Args:
        dataset_dir: path where to store the cached dataset
        feature_set: select what node features to precompute as MalNet
            originally doesn't have any node nor edge features

    Returns:
        PyG dataset object
    """
    if feature_set in ['none', 'Constant']:
        tf = T.Constant()
    elif feature_set == 'OneHotDegree':
        tf = T.OneHotDegree()
    elif feature_set == 'LocalDegreeProfile':
        tf = T.LocalDegreeProfile()
    else:
        raise ValueError(f"Unexpected transform function: {feature_set}")

    dataset = MalNetTiny(dataset_dir)
    dataset.name = 'MalNetTiny'
    logging.info(f'Computing "{feature_set}" node features for MalNetTiny.')
    pre_transform_in_memory(dataset, tf)

    split_dict = dataset.get_idx_split()
    dataset.split_idxs = [split_dict['train'],
                          split_dict['valid'],
                          split_dict['test']]

    return dataset


def preformat_OGB_Graph(dataset_dir, name):
    """Load and preformat OGB Graph Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Graph dataset

    Returns:
        PyG dataset object
    """
    dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]

    if name == 'ogbg-code2':
        from graphgps.loader.ogbg_code2_utils import idx2vocab, \
            get_vocab_mapping, augment_edge, encode_y_to_arr
        num_vocab = 5000  # The number of vocabulary used for sequence prediction
        max_seq_len = 5  # The maximum sequence length to predict

        seq_len_list = np.array([len(seq) for seq in dataset.data.y])
        logging.info(f"Target sequences less or equal to {max_seq_len} is "
            f"{np.sum(seq_len_list <= max_seq_len) / len(seq_len_list)}")

        # Building vocabulary for sequence prediction. Only use training data.
        vocab2idx, idx2vocab_local = get_vocab_mapping(
            [dataset.data.y[i] for i in s_dict['train']], num_vocab)
        logging.info(f"Final size of vocabulary is {len(vocab2idx)}")
        idx2vocab.extend(idx2vocab_local)  # Set to global variable to later access in CustomLogger

        # Set the transform function:
        # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
        # encode_y_to_arr: add y_arr to PyG data object, indicating the array repres
        dataset.transform = T.Compose(
            [augment_edge,
             lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])

        # Subset graphs to a maximum size (number of nodes) limit.
        pre_transform_in_memory(dataset, partial(clip_graphs_to_size,
                                                 size_limit=1000))

    return dataset


def preformat_OGB_PCQM4Mv2(dataset_dir, name):
    """Load and preformat PCQM4Mv2 from OGB LSC.

    OGB-LSC provides 4 data index splits:
    2 with labeled molecules: 'train', 'valid' meant for training and dev
    2 unlabeled: 'test-dev', 'test-challenge' for the LSC challenge submission

    We will take random 150k from 'train' and make it a validation set and
    use the original 'valid' as our testing set.

    Note: PygPCQM4Mv2Dataset requires rdkit

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of the training set

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from ogb.lsc import PygPCQM4Mv2Dataset
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2Dataset, '
                      'make sure RDKit is installed.')
        raise e


    dataset = PygPCQM4Mv2Dataset(root=dataset_dir)
    split_idx = dataset.get_idx_split()

    rng = default_rng(seed=42)
    train_idx = rng.permutation(split_idx['train'].numpy())
    train_idx = torch.from_numpy(train_idx)

    # Leave out 150k graphs for a new validation set.
    valid_idx, train_idx = train_idx[:150000], train_idx[150000:]
    if name == 'full':
        split_idxs = [train_idx,  # Subset of original 'train'.
                      valid_idx,  # Subset of original 'train' as validation set.
                      split_idx['valid']  # The original 'valid' as testing set.
                      ]

    elif name == 'subset':
        # Further subset the training set for faster debugging.
        subset_ratio = 0.1
        subtrain_idx = train_idx[:int(subset_ratio * len(train_idx))]
        subvalid_idx = valid_idx[:50000]
        subtest_idx = split_idx['valid']  # The original 'valid' as testing set.

        dataset = dataset[torch.cat([subtrain_idx, subvalid_idx, subtest_idx])]
        data_list = [data for data in dataset]
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        n1, n2, n3 = len(subtrain_idx), len(subvalid_idx), len(subtest_idx)
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]

    elif name == 'inference':
        split_idxs = [split_idx['valid'],  # The original labeled 'valid' set.
                      split_idx['test-dev'],  # Held-out unlabeled test dev.
                      split_idx['test-challenge']  # Held-out challenge test set.
                      ]

        dataset = dataset[torch.cat(split_idxs)]
        data_list = [data for data in dataset]
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        n1, n2, n3 = len(split_idxs[0]), len(split_idxs[1]), len(split_idxs[2])
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]
        # Check prediction targets.
        assert(all([not torch.isnan(dataset[i].y)[0] for i in split_idxs[0]]))
        assert(all([torch.isnan(dataset[i].y)[0] for i in split_idxs[1]]))
        assert(all([torch.isnan(dataset[i].y)[0] for i in split_idxs[2]]))

    else:
        raise ValueError(f'Unexpected OGB PCQM4Mv2 subset choice: {name}')
    dataset.split_idxs = split_idxs
    return dataset


def preformat_PCQM4Mv2Contact(dataset_dir, name):
    """Load PCQM4Mv2-derived molecular contact link prediction dataset.

    Note: This dataset requires RDKit dependency!

    Args:
       dataset_dir: path where to store the cached dataset
       name: the type of dataset split: 'shuffle', 'num-atoms'

    Returns:
       PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary
        from graphgps.loader.dataset.pcqm4mv2_contact import \
            PygPCQM4Mv2ContactDataset, \
            structured_neg_sampling_transform
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2ContactDataset, '
                      'make sure RDKit is installed.')
        raise e

    split_name = name.split('-', 1)[1]
    dataset = PygPCQM4Mv2ContactDataset(dataset_dir, subset='530k')
    # Inductive graph-level split (there is no train/test edge split).
    s_dict = dataset.get_idx_split(split_name)
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    if cfg.dataset.resample_negative:
        dataset.transform = structured_neg_sampling_transform
    return dataset


def preformat_Peptides(dataset_dir, name):
    """Load Peptides dataset, functional or structural.

    Note: This dataset requires RDKit dependency!

    Args:
        dataset_dir: path where to store the cached dataset
        name: the type of dataset split:
            - 'peptides-functional' (10-task classification)
            - 'peptides-structural' (11-task regression)

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from graphgps.loader.dataset.peptides_functional import \
            PeptidesFunctionalDataset
        from graphgps.loader.dataset.peptides_structural import \
            PeptidesStructuralDataset
    except Exception as e:
        logging.error('ERROR: Failed to import Peptides dataset class, '
                      'make sure RDKit is installed.')
        raise e

    dataset_type = name.split('-', 1)[1]
    if dataset_type == 'functional':
        dataset = PeptidesFunctionalDataset(dataset_dir)
    elif dataset_type == 'structural':
        dataset = PeptidesStructuralDataset(dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    return dataset


def preformat_TUDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's TUDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS', 'TRIANGLES']:
        func = None
    elif name.startswith('IMDB-') or name == "COLLAB" or name == "REDDIT-MULTI-5K":
        func = T.Constant()
    else:
        raise ValueError(f"Loading dataset '{name}' from "
                         f"TUDataset is not supported.")
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    return dataset


def preformat_ZINC(dataset_dir, name):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """
    if name not in ['subset', 'full']:
        raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    dataset = join_dataset_splits(
        [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_AQSOL(dataset_dir):
    """Load and preformat AQSOL datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [AQSOL(root=dataset_dir, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_VOCSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat VOCSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [VOCSuperpixels(root=dataset_dir, name=name,
                        slic_compactness=slic_compactness,
                        split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_COCOSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat COCOSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [COCOSuperpixels(root=dataset_dir, name=name,
                         slic_compactness=slic_compactness,
                         split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_GKATSynthetic(dataset_dir, name):
    """Load and preformat GKAT Synthetic motif detection datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific motif type: 'Cycle', 'Grid', 'Ladder', 'CircularLadder', 'Caveman'

    Returns:
        PyG dataset object
    """
    if name not in ['Cycle', 'Grid', 'Ladder', 'CircularLadder', 'Caveman']:
        raise ValueError(f"Unknown GKAT Synthetic dataset: {name}. "
                         f"Must be one of: Cycle, Grid, Ladder, CircularLadder, Caveman")
    
    # Use the parent directory of dataset_dir to find the actual dataset files
    # This accounts for the master loader creating a subdirectory
    parent_dir = osp.dirname(dataset_dir) if dataset_dir.endswith('GKATSynthetic') else dataset_dir
    dataset = GKATSyntheticDataset(parent_dir, name)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]
    
    return dataset


def preformat_GridRewiring(dataset_dir, name):
    """Load and preformat Grid Rewiring synthetic datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: dataset name identifier (typically 'grid_rewiring')

    Returns:
        PyG dataset object
    """
    # Use the parent directory of dataset_dir to store the actual dataset files
    parent_dir = osp.dirname(dataset_dir) if dataset_dir.endswith('GridRewiring') else dataset_dir
    dataset = GridRewiringDataset(parent_dir, name)
    
    # Set node encoder configuration for TypeDictNodeEncoder
    # GridRewiring has 2 node types: 0 = uncolored, 1 = colored
    cfg.dataset.node_encoder_num_types = 2
    
    # Get split indices and store them as split_idxs attribute
    # Do NOT call set_dataset_splits here as it creates incompatible graph index attributes
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s].tolist() for s in ['train', 'valid', 'test']]
    
    return dataset



def preformat_GraphCIFAR10(dataset_dir, name):
    """Load and preformat GraphCIFAR10 dataset.

    Creates a graph version of CIFAR-10 where images are divided into patches
    that become nodes in a grid graph.

    Args:
        dataset_dir: path where to store the cached dataset
        name: split name ('train', 'val', 'test') or dataset name

    Returns:
        PyG dataset object
    """
    # GraphCIFAR10 uses join_dataset_splits pattern like ZINC and AQSOL
    dataset = join_dataset_splits(
        [GraphCIFAR10(root=dataset_dir, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset



def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    # get() takes an underlying storage index, not a subset-local index.
    # MalNetTiny(split=...) uses _indices over a shared 5,000-graph store.
    # Resolve those indices explicitly, without applying runtime transforms.
    data_list = [dataset.get(int(index))
                 for dataset in datasets for index in dataset.indices()]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]
