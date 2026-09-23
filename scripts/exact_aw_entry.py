"""Training instrumentation for the isolated Exact-AW comparison."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import time


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    import torch
    import graphgps  # noqa: F401
    from graphgps.layer.aw_rope import AnalyticWalkRoPE
    from graphgps.layer.graphrope import GraphRoPE
    from torch_geometric.graphgym.config import cfg
    from torch_geometric.graphgym.register import train_dict
    from exact_aw_operator import RUNTIME_REVISION, install_exact_backend
    arm, phase = os.environ["AW_EXACT_ARM"], os.environ["AW_EXACT_PHASE"]
    if arm not in ("exact-aw", "sparse-aw"):
        raise ValueError(arm)
    if arm == "exact-aw":
        install_exact_backend(AnalyticWalkRoPE, GraphRoPE)
    path = Path(os.environ["AW_EXACT_REPORT"])
    report = {"arm": arm, "phase": phase, "complete": False, "train_epochs": [],
              "runtime_revision": RUNTIME_REVISION,
              "operator": "complete-resolvent" if arm == "exact-aw" else "finite-resolvent",
              "cache_condition": "prepared dataset caches; warm/mixed, not cold start"}
    training = importlib.import_module("graphgps.train.custom_train")
    original_epoch = training.train_epoch
    if phase == "efficiency":
        def epoch(*args, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = original_epoch(*args, **kwargs)
            torch.cuda.synchronize()
            report["train_epochs"].append({"seconds": time.perf_counter()-start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            _atomic_json(path, report)
            return result
        training.train_epoch = epoch
    original_train = train_dict["custom"]

    def train(loggers, loaders, model, optimizer, scheduler):
        modules = [(name, m) for name, m in model.named_modules() if isinstance(m, AnalyticWalkRoPE)]
        if not modules or any(m.method != "aw" or m.normalize_resolvent or m.residual_mix != 1
                              or m.preserve_input_norm for _, m in modules):
            raise ValueError("unsupported positional action for matched Exact-AW")
        report.update({"seed": cfg.seed, "initial_state_sha256": state_hash(model),
            "solve_backend": ("real-block" if next(model.parameters()).is_cuda else "native")
                             if arm == "exact-aw" else "finite-recurrence",
            "cuda_linalg_preference": str(torch.backends.cuda.preferred_linalg_library())
                                      if arm == "exact-aw" and next(model.parameters()).is_cuda else "not-used",
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "split_sizes": [len(loader.dataset) for loader in loaders],
            "layers": [{"name": name, "K": m.num_steps, "z_initial": float(m.z),
                        "learnable_z": m.z_logit.requires_grad,
                        "learnable_frequencies": m.frequencies.requires_grad} for name, m in modules]})
        if phase == "efficiency":
            original_write = loggers[0].write_epoch
            def write_epoch(i):
                start = time.perf_counter()
                result = original_write(i)
                torch.cuda.synchronize()
                report["train_epochs"][-1]["seconds"] += time.perf_counter()-start
                report["train_epochs"][-1]["epoch"] = i
                _atomic_json(path, report)
                return result
            loggers[0].write_epoch = write_epoch
        _atomic_json(path, report)
        start = time.perf_counter()
        original_train(loggers, loaders, model, optimizer, scheduler)
        report.update({"complete": True, "training_with_eval_ckpt_seconds": time.perf_counter()-start,
                       "z_final": {name: float(m.z) for name, m in modules}})
        _atomic_json(path, report)
    train_dict["custom"] = train
    project = Path(__file__).resolve().parents[1]
    runpy.run_path(str(project / "external/Graph-RoPE/main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
