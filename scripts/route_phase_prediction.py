#!/usr/bin/env python3
"""Paired route-parity prediction with matched transport and neural budgets.

Two equal-length routes contain an even number of separated A->B motifs.
Opposite-label examples move the last complete motif to the other route.
Every distance shell has the same multiset of raw node features, and the
directed edge-type histogram is unchanged. Labels are combinatorial parity,
not a teacher network's predictions. All four arms receive the same raw
graph, features and graph-only spectral coordinates.

The endpoint readout has a polynomial-mixing collision guarantee. The
attention readout deliberately removes this restriction and is an essential
control: the collision theorem does NOT apply to a full WIRE network.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LAYER = ROOT / "external/Graph-RoPE/graphgps/layer/aw_rope.py"
spec = importlib.util.spec_from_file_location("route_phase_production_aw", LAYER)
AW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AW)
ARMS = ("wire", "mixing", "wire-mixing", "aw")
CONTROLS = ("gradient",)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def generate(pairs, seed, *, route_length=14, motifs=4, permute=True,
             spectral_gauge=True, strength_range=(.95, 1.05), nuisance=.1):
    """Each pair is kept in one split; no labels enter feature generation.

    The first motifs-1 lane choices are random; setting the last lane gives
    the two parity classes. Whole node features are exchanged between lanes
    when moving the last motif. Laplacian eigenspace bases and permutations
    are generated independently of the class and shared within each pair.
    """
    if motifs < 2 or motifs % 2 or route_length < 3*motifs:
        raise ValueError("need an even number of motifs and route_length >= 3*motifs")
    g = torch.Generator().manual_seed(seed)
    m, n = route_length, 2*route_length
    x = torch.zeros(pairs, 2, n, 6)
    x[:, :, 0, 0] = 1
    x[:, :, m, 1] = 1
    x[..., 4:] = nuisance * torch.randn(pairs, 1, n, 2, generator=g)
    x[:, :, [0, m], 4:] = 0
    cyclic = torch.arange(n)
    base_edge = torch.stack([cyclic.repeat_interleave(2),
                            torch.stack([(cyclic+1)%n, (cyclic-1)%n], -1).flatten()])
    theta = cyclic * (2*math.pi/n)
    base_t = torch.stack([theta.cos(), theta.sin(), (2*theta).cos(), (2*theta).sin()], -1)
    base_t *= math.sqrt(2/n)  # orthonormal real Laplacian eigenvectors
    ts, edges, metadata = [], [], []
    for i in range(pairs):
        # Slack is distributed over zero-only gaps, preserving disjoint motifs.
        gaps = torch.bincount(torch.randint(motifs+1, (m-3*motifs,), generator=g), minlength=motifs+1)
        positions, cursor = [], 1+int(gaps[0])
        for event in range(motifs):
            positions.append(cursor)
            cursor += 3 + int(gaps[event+1])
        strengths = strength_range[0] + (strength_range[1]-strength_range[0])*torch.rand(motifs, 2, generator=g)
        prefix = torch.randint(2, (motifs-1,), generator=g).tolist()
        class_lanes = []
        for label in (0, 1):
            # high ideal power iff (M/2 - number_on_lower_route) is even.
            last = (motifs//2 - sum(prefix) - (1-label)) % 2
            lanes = prefix + [last]
            class_lanes.append(lanes)
            for event, (p, lane) in enumerate(zip(positions, lanes)):
                for offset in (0, 1):
                    v = p+offset if lane == 0 else n-p-offset
                    x[i, label, v, 2+offset] = strengths[event, offset]
        for p in (positions[-1], positions[-1]+1):
            x[i, 1, p, 4:] = x[i, 0, n-p, 4:]
            x[i, 1, n-p, 4:] = x[i, 0, p, 4:]
        t = base_t.clone()
        if spectral_gauge:
            for block in (0, 2):
                phi = float(torch.rand((), generator=g))*2*math.pi
                reflection = 2*int(torch.randint(2, (), generator=g))-1
                orthogonal = torch.tensor([[math.cos(phi), -reflection*math.sin(phi)],
                                           [math.sin(phi), reflection*math.cos(phi)]])
                t[:, block:block+2] = t[:, block:block+2] @ orthogonal
        permutation = torch.randperm(n, generator=g) if permute else torch.arange(n)
        inverse = torch.argsort(permutation)
        x[i] = x[i, :, permutation].clone()
        ts.append(t[permutation].expand(2, n, 4).clone())
        edges.append(inverse[base_edge].expand(2, 2, 2*n).clone())
        metadata.append({"positions": positions, "lanes": class_lanes,
                         "strengths": strengths.tolist(), "permutation": permutation.tolist()})
    result = {"x": x.flatten(0, 1), "t": torch.stack(ts).flatten(0, 1),
              "edges": torch.stack(edges).flatten(0, 1),
              "y": torch.tensor([0., 1.]).repeat(pairs)}
    result["metadata"] = metadata
    result["tensor_sha256"] = state_hash({k:v for k,v in result.items() if isinstance(v, torch.Tensor)})
    return result


def batch(data, indices):
    return {k: data[k][indices] for k in ("x", "t", "edges", "y")}


def flatten_edges(edges, n):
    return (edges + torch.arange(len(edges), device=edges.device)[:, None, None]*n).permute(1, 0, 2).reshape(2, -1)


def phase_free(x, edges, L, z):
    source, target = edges
    degree = x.new_zeros(len(x)).index_add_(0, source, x.new_ones(len(source)))
    weights = degree[source].reciprocal()
    state, result, coefficient = x, x, 1.
    for _ in range(L):
        state = torch.zeros_like(x).index_add_(0, source, state[target]*weights[:, None])
        coefficient *= z
        result = result + coefficient*state
    return result


class BudgetedNodeAdapter(nn.Module):
    """Active pointwise residual capacity with an exact parameter budget.

    A two-layer SiLU MLP uses the largest whole hidden width that fits.
    Fewer than 2*d+1 remaining coefficients weight fixed random nonlinear
    features. No dummy, masked-out or unused trainable parameters are added.
    Baselines receive MORE pointwise capacity to offset AW's edge scorer.
    This adapter never consumes spectral coordinates or neighbor features.
    """
    def __init__(self, d, budget):
        super().__init__()
        width, remainder = divmod(budget, 2*d+1)
        if width < 1:
            raise ValueError("adapter budget must support a hidden unit")
        self.mlp = nn.Sequential(nn.Linear(d, width), nn.SiLU(), nn.Linear(width, d, bias=False))
        if remainder:
            self.coefficients = nn.Parameter(torch.randn(remainder)*.01)
            self.register_buffer("rf_input", torch.randn(d, remainder)/math.sqrt(d))
            self.register_buffer("rf_output", torch.randn(remainder, d)/math.sqrt(max(remainder, 1)))
        else:
            self.register_parameter("coefficients", None)
        assert sum(p.numel() for p in self.parameters()) == budget

    def forward(self, h):
        residual = self.mlp(h)
        if self.coefficients is not None:
            residual = residual + (F.silu(h @ self.rf_input)*self.coefficients) @ self.rf_output
        return h + .1*residual


class RoutePredictor(nn.Module):
    def __init__(self, arm, *, readout="attention", d=16, L=18, z=.9,
                 layers=1, seed=0, matched_budget=True):
        super().__init__()
        if arm not in ARMS + CONTROLS:
            raise ValueError(arm)
        if readout not in ("endpoint", "attention"):
            raise ValueError(readout)
        self.arm, self.readout, self.d, self.L, self.z, self.layers = arm, readout, d, L, z, layers
        self.encoder = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 1))
        if readout == "attention":
            self.qkv = nn.ModuleList([nn.Linear(d, 3*d) for _ in range(layers)])
            self.output = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, 2*d), nn.SiLU(), nn.Linear(2*d, d)) for _ in range(layers)])
        elif layers != 1:
            raise ValueError("the endpoint theorem concerns one linear-transport stage")
        self.shared_initial_hash = state_hash(self.state_dict())
        self.pos, self.adapters = nn.ModuleList(), nn.ModuleList()
        # d/2 frequencies and the production 2d->32->1 edge scorer.
        positional_budget = ((2*d+1)*32 + 33 + d//2) + d*(2*d+1)
        with torch.random.fork_rng():
            torch.manual_seed(918273+seed)
            for _ in range(layers):
                if arm in ("aw", "gradient"):
                    pos = AW.AnalyticWalkRoPE(
                        d, num_steps=L, initial_z=z, learnable_z=False,
                        field_type="matched-potential" if arm == "gradient" else "local-antisymmetric")
                elif arm in ("wire", "wire-mixing"):
                    pos = nn.Linear(4, d//2, bias=False)
                    nn.init.zeros_(pos.weight)
                else:
                    pos = nn.Identity()
                count = sum(p.numel() for p in pos.parameters())
                self.pos.append(pos)
                self.adapters.append(BudgetedNodeAdapter(d, positional_budget-count) if matched_budget else nn.Identity())
        self.parameter_audit = {"total": sum(p.numel() for p in self.parameters()),
                                "positional": sum(p.numel() for p in self.pos.parameters()),
                                "pointwise_adapter": sum(p.numel() for p in self.adapters.parameters()),
                                "matched_budget": matched_budget,
                                "shared_initial_hash": self.shared_initial_hash}

    def transform(self, q, k, h, edges, t, layer, intervention=None):
        if intervention not in (None, "zero-phase"):
            raise ValueError(intervention)
        if self.arm in ("aw", "gradient") and intervention == "zero-phase":
            return phase_free(torch.cat((q, k), -1), edges, self.L, self.z).chunk(2, -1)
        if self.arm in ("aw", "gradient"):
            return self.pos[layer](q, k, h, edges)
        if self.arm in ("mixing", "wire-mixing"):
            q, k = phase_free(torch.cat((q, k), -1), edges, self.L, self.z).chunk(2, -1)
        if self.arm in ("wire", "wire-mixing"):
            angle = self.pos[layer](t)
            q, k = AW.rotate_pairs(q, angle), AW.rotate_pairs(k, angle)
        return q, k

    def forward(self, data, *, intervention=None):
        x, t = data["x"], data["t"]
        b, n, _ = x.shape
        edge = flatten_edges(data["edges"], n)
        h = self.encoder(x).flatten(0, 1)
        for layer in range(self.layers):
            h = self.adapters[layer](h)
            if self.readout == "endpoint":
                h, _ = self.transform(h, h, h, edge, t.flatten(0, 1), layer, intervention)
            else:
                q, k, v = self.qkv[layer](h).chunk(3, -1)
                q, k = self.transform(q, k, h, edge, t.flatten(0, 1), layer, intervention)
                q, k, v = (value.reshape(b, n, self.d) for value in (q, k, v))
                attention = torch.softmax(q @ k.transpose(-1, -2)/math.sqrt(self.d), -1)
                h = self.norm[layer](h + self.output[layer]((attention @ v).flatten(0, 1)))
                h = h + self.ff[layer](h)
        h = h.reshape(b, n, self.d)
        root = (h*x[:, :, :1]).sum(1)
        return self.head(self.final_norm(root)).squeeze(-1)


@torch.no_grad()
def evaluate(model, data, batch_size=128, return_logits=False, intervention=None):
    model.eval()
    logits = torch.cat([model(batch(data, slice(i, i+batch_size)), intervention=intervention) for i in range(0, len(data["y"]), batch_size)])
    y = data["y"]
    metrics = {"loss": F.binary_cross_entropy_with_logits(logits, y).item(),
               "accuracy": ((logits >= 0) == y.bool()).float().mean().item(),
               "pair_correct": (((logits[::2] < 0) & (logits[1::2] >= 0)).float().mean().item()),
               "mean_pair_logit_gap": (logits[1::2]-logits[::2]).abs().mean().item(),
               "max_pair_logit_gap": (logits[1::2]-logits[::2]).abs().max().item()}
    return (metrics, logits) if return_logits else metrics


def fit(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.out.exists():
        raise FileExistsError("fresh output required: " + str(args.out))
    args.out.mkdir(parents=True)
    config = vars(args).copy()
    config["out"] = str(args.out)
    atomic_json(args.out / "config.json", config)
    generator_args = {"route_length": args.route_length, "motifs": args.motifs}
    train = generate(args.train_pairs, args.data_seed, **generator_args)
    val = generate(args.val_pairs, args.data_seed+1, **generator_args)
    model = RoutePredictor(args.arm, readout=args.readout, d=args.width, L=args.depth,
                           z=args.z, layers=args.layers, seed=args.seed)
    source_hashes = {"experiment": sha(__file__), "production_aw": sha(LAYER)}
    atomic_json(args.out / "audit.json", {"parameters": model.parameter_audit,
                "source_hashes": source_hashes,
                "train_tensor_sha256": train["tensor_sha256"], "validation_tensor_sha256": val["tensor_sha256"],
                "test_access_during_training": False, "selection": "minimum validation BCE; earliest tie"})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    generator = torch.Generator().manual_seed(500000+args.seed)
    best = None
    start = time.monotonic()
    trajectory = args.out / "trajectory.jsonl"
    for step in range(args.steps+1):
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val)
            row = {"step": step, "validation": metrics, "seconds": time.monotonic()-start}
            with trajectory.open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            if best is None or metrics["loss"] < best["validation"]["loss"]:
                best = row
                torch.save({"model": model.state_dict(), "step": step, "config": config,
                            "source_hashes": source_hashes}, args.out / "best.pt")
            atomic_json(args.out / "progress.json", {"current": row, "best": best, "complete": False})
            print(json.dumps({"arm": args.arm, "seed": args.seed, **row}), flush=True)
        if step == args.steps:
            break
        model.train()
        pairs = torch.randint(args.train_pairs, (args.batch_pairs,), generator=generator)
        indices = (2*pairs[:, None] + torch.arange(2)).flatten()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(batch(train, indices)), train["y"][indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        if step == 0:
            gradient_audit = {name: {"count": parameter.numel(), "grad_present": parameter.grad is not None,
                                     "grad_finite": parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                                     "nonzero_elements": int(torch.count_nonzero(parameter.grad)) if parameter.grad is not None else 0}
                              for name, parameter in model.named_parameters()}
            atomic_json(args.out / "gradient-audit.json", gradient_audit)
            if not all(v["grad_present"] and v["grad_finite"] for v in gradient_audit.values()):
                raise RuntimeError("unused or non-finite trainable parameters")
        optimizer.step()
    state = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    summary = {"config": config, "best": best, "train": evaluate(model, train),
               "parameter_audit": model.parameter_audit, "checkpoint_sha256": sha(args.out/"best.pt"),
               "trajectory_sha256": sha(trajectory), "source_hashes": source_hashes}
    if args.phase == "formal":
        # Only instantiated after optimization and checkpoint selection end.
        test = generate(args.test_pairs, args.data_seed+2, **generator_args)
        summary["test"], predictions = evaluate(model, test, return_logits=True)
        summary["test_tensor_sha256"] = test["tensor_sha256"]
        np.savez_compressed(args.out/"test-predictions.npz", logits=predictions.numpy(), labels=test["y"].numpy())
        if args.arm in ("aw", "gradient"):
            summary["test_zero_phase"] = evaluate(model, test, intervention="zero-phase")
        if args.ood_length:
            ood = generate(args.test_pairs, args.data_seed+3, route_length=args.ood_length, motifs=args.motifs)
            summary["ood"], predictions = evaluate(model, ood, return_logits=True)
            summary["ood_tensor_sha256"] = ood["tensor_sha256"]
            np.savez_compressed(args.out/"ood-predictions.npz", logits=predictions.numpy(), labels=ood["y"].numpy())
    atomic_json(args.out/"result.json", summary)
    atomic_json(args.out/"progress.json", {"current": row, "best": best, "complete": True})
    print(json.dumps({"complete": True, "arm": args.arm, "seed": args.seed, "best": best,
                      "test": summary.get("test"), "ood": summary.get("ood")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS+CONTROLS, required=True)
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--readout", choices=("endpoint", "attention"), default="attention")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-seed", type=int, default=120917)
    parser.add_argument("--route-length", type=int, default=14)
    parser.add_argument("--motifs", type=int, default=4)
    parser.add_argument("--ood-length", type=int, default=0)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--z", type=float, default=.9)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--train-pairs", type=int, default=1024)
    parser.add_argument("--val-pairs", type=int, default=256)
    parser.add_argument("--test-pairs", type=int, default=1024)
    parser.add_argument("--batch-pairs", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
