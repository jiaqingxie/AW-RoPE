from __future__ import annotations
from collections import Counter
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import route_phase_prediction as r

torch.set_num_threads(1)


def unpermute(data, pair, label):
    inverse = torch.argsort(torch.tensor(data["metadata"][pair]["permutation"]))
    return data["x"][2*pair+label, inverse]


@pytest.mark.parametrize("motifs,m", [(2, 6), (2, 8), (4, 12), (4, 14)])
def test_balanced_paired_shell_and_edge_type_marginals(motifs, m):
    data = r.generate(12, 43, motifs=motifs, route_length=m)
    assert data["y"].tolist() == [0., 1.]*12
    for pair in range(12):
        a, b = (unpermute(data, pair, label) for label in (0, 1))
        assert torch.equal(a[[0, m]], b[[0, m]])
        for shell in range(1, m):
            assert sorted(map(tuple, a[[shell, 2*m-shell]].tolist())) == sorted(map(tuple, b[[shell, 2*m-shell]].tolist()))
        def edge_types(x):
            return Counter((tuple(x[u, :4].tolist()), tuple(x[v, :4].tolist()))
                           for u in range(2*m) for v in ((u+1)%(2*m), (u-1)%(2*m)))
        assert edge_types(a) == edge_types(b)
        for label, lanes in enumerate(data["metadata"][pair]["lanes"]):
            assert int((motifs//2-sum(lanes)) % 2 == 0) == label
        assert torch.equal(data["edges"][2*pair], data["edges"][2*pair+1])
        assert torch.equal(data["t"][2*pair], data["t"][2*pair+1])


def test_raw_split_identity_and_determinism():
    a, b, c = r.generate(8, 90), r.generate(8, 90), r.generate(8, 91)
    assert a["tensor_sha256"] == b["tensor_sha256"] != c["tensor_sha256"]
    hashes = [{r.state_hash({"x": item}) for item in d["x"]} for d in (a, c)]
    assert hashes[0].isdisjoint(hashes[1])


def test_random_gauges_remain_orthonormal_laplacian_eigenvectors():
    data = r.generate(4, 72)
    n = data["x"].shape[1]
    eigenvalues = 2-2*torch.cos(torch.tensor([1, 1, 2, 2])*(2*torch.pi/n))
    for t, edge in zip(data["t"], data["edges"]):
        lap = 2*torch.eye(n)
        lap[edge[0], edge[1]] = -1
        torch.testing.assert_close(t.T@t, torch.eye(4), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(lap@t, t*eigenvalues, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("readout", ["endpoint", "attention"])
def test_trainable_budget_backbone_initialization_and_gradients(readout):
    data = r.generate(3, 52)
    counts, hashes = [], []
    for arm in r.ARMS+r.CONTROLS:
        torch.manual_seed(29)
        model = r.RoutePredictor(arm, readout=readout, seed=29, L=4)
        counts.append(model.parameter_audit["total"])
        hashes.append(model.shared_initial_hash)
        loss = F.binary_cross_entropy_with_logits(model(data), data["y"])
        loss.backward()
        for name, value in model.named_parameters():
            if value.requires_grad:
                assert value.grad is not None, (arm, name)
                assert torch.isfinite(value.grad).all(), (arm, name)
    assert len(set(counts)) == len(set(hashes)) == 1


@pytest.mark.parametrize("arm", ["wire", "mixing", "wire-mixing", "gradient"])
def test_endpoint_collision_for_arbitrary_parameters(arm):
    data = r.generate(5, 21)
    data = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in data.items()}
    torch.manual_seed(81)
    model = r.RoutePredictor(arm, readout="endpoint", seed=81).double()
    # Test after changing every trainable parameter, including nonzero WIRE
    # frequencies, not just the initial zero-rotation special case.
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p)*.3)
    logits = model(data)
    torch.testing.assert_close(logits[::2], logits[1::2], atol=2e-12, rtol=2e-12)


def test_phase_free_matches_production_zero_field():
    data = r.generate(3, 52)
    b, n, _ = data["x"].shape
    edge = r.flatten_edges(data["edges"], n)
    x = torch.randn(b*n, 16, dtype=torch.float64)
    actual = r.phase_free(x, edge, 12, .9)
    expected = r.AW.truncated_walk_resolvent(x, edge, x.new_zeros(edge.shape[1]),
                     x.new_ones(8), z=x.new_tensor(.9), num_steps=12)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("motifs,m", [(2, 6), (2, 8), (4, 12), (4, 14)])
def test_constructive_field_solves_parity_with_margin(motifs, m):
    data = r.generate(10, 212, route_length=m, motifs=motifs)
    b, n, _ = data["x"].shape
    x = data["x"].double().flatten(0, 1)
    edge = r.flatten_edges(data["edges"], n)
    source, target = edge
    field = (torch.pi/2)*(x[source, 2]*x[target, 3]-x[source, 3]*x[target, 2])
    signal = x.new_zeros(b*n, 2)
    signal[:, 0] = x[:, 1]
    z = .8
    output = r.AW.truncated_walk_resolvent(signal, edge, field, x.new_ones(1),
                    z=x.new_tensor(z), num_steps=m)
    endpoint = (output.reshape(b, n, 2)*data["x"][:, :, :1]).sum(1)
    power = endpoint.square().sum(1)/(4*(z/2)**(2*m))
    closed = []
    for pair, meta in enumerate(data["metadata"]):
        strengths = torch.tensor(meta["strengths"], dtype=torch.float64)
        motif_phases = torch.pi/2*strengths.prod(1)
        for label in (0, 1):
            directions = 1-2*torch.tensor(meta["lanes"][label], dtype=torch.float64)
            closed.append(torch.cos((motif_phases*directions).sum()/2)**2)
    torch.testing.assert_close(power, torch.stack(closed), atol=2e-12, rtol=2e-12)
    assert (power[::2] < .2).all()
    assert (power[1::2] > .8).all()


@pytest.mark.parametrize("arm", r.ARMS+r.CONTROLS)
def test_arbitrary_permutation_equivariance_and_no_batch_leakage(arm):
    data = r.generate(2, 91, motifs=2, route_length=8)
    torch.manual_seed(19)
    model = r.RoutePredictor(arm, L=4, seed=19).double().eval()
    data = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in data.items()}
    original = model(data)
    permuted = {k: v.clone() for k, v in data.items() if isinstance(v, torch.Tensor)}
    g = torch.Generator().manual_seed(231)
    for i in range(len(data["x"])):
        p = torch.randperm(16, generator=g)
        inverse = torch.argsort(p)
        permuted["x"][i], permuted["t"][i] = data["x"][i, p], data["t"][i, p]
        permuted["edges"][i] = inverse[data["edges"][i]]
    torch.testing.assert_close(model(permuted), original, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(model(r.batch(data, slice(0, 1))), original[:1], atol=2e-12, rtol=2e-12)


def test_zero_phase_intervention_removes_endpoint_pair_information():
    data = r.generate(4, 13, route_length=8, motifs=2)
    data = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in data.items()}
    torch.manual_seed(14)
    model = r.RoutePredictor("aw", readout="endpoint", L=8, seed=14).double()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p)*.6)
    logits = model(data, intervention="zero-phase")
    torch.testing.assert_close(logits[::2], logits[1::2], atol=2e-12, rtol=2e-12)
