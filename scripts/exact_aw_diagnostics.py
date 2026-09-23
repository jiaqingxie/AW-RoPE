"""Reproducible forward/gradient truncation curves; no predictive claims."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from exact_aw_operator import RUNTIME_REVISION, exact_walk_resolvent


def production():
    path = Path(__file__).resolve().parents[1]/"external/Graph-RoPE/graphgps/layer/aw_rope.py"
    spec = importlib.util.spec_from_file_location("exact_aw_diagnostic_production", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(device="cpu"):
    aw = production()
    torch.manual_seed(606)
    dtype = torch.float64
    edges = torch.tensor([[0,1,1,2,2,3,3,0,0,2],[1,0,2,1,3,2,0,3,2,0]],device=device)
    x = torch.randn(5,8,dtype=dtype,device=device,requires_grad=True)
    half = torch.tensor([.3,.7,-.4,.5,.8],dtype=dtype,device=device)
    a = torch.stack((half,-half),-1).flatten().requires_grad_()
    frequencies = torch.tensor([.2,.7,1.3,2.],dtype=dtype,device=device,requires_grad=True)
    weight = torch.linspace(.5,1.5,10,dtype=dtype,device=device).requires_grad_()
    readout = torch.randn_like(x)
    rows = []
    for decay in (.4,.6,.8,.95):
        z = torch.tensor(decay,dtype=dtype,device=device,requires_grad=True)
        inputs = (x,a,frequencies,weight,z)
        exact = exact_walk_resolvent(x,edges,a,frequencies,z=z,edge_weight=weight)
        exact_grad = torch.autograd.grad((exact*readout).sum(),inputs,retain_graph=True)
        full_norm = torch.view_as_complex(exact.reshape(5,4,2).contiguous()).abs().max()
        input_norm = torch.view_as_complex(x.reshape(5,4,2).contiguous()).abs().max()
        exact_vector = torch.cat([g.flatten() for g in exact_grad])
        for steps in (0,1,2,4,8,16,32,64):
            approx = aw.truncated_walk_resolvent(x,edges,a,frequencies,z=z,num_steps=steps,edge_weight=weight)
            grads = torch.autograd.grad((approx*readout).sum(),inputs,allow_unused=True,retain_graph=True)
            grads = [torch.zeros_like(v) if g is None else g for g,v in zip(grads,inputs)]
            delta = torch.view_as_complex((exact-approx).reshape(5,4,2).contiguous()).abs().max()
            grad_delta = torch.cat([g.flatten() for g in grads])-exact_vector
            bound = decay**(steps+1)/(1-decay)*float(input_norm)
            if float(delta) > bound + 1e-10:
                raise ValueError("forward tail bound violated")
            if not all(torch.isfinite(v).all() for v in (exact,approx,grad_delta)):
                raise ValueError("nonfinite diagnostic")
            rows.append({"z":decay,"K":steps,"forward_absolute":float(delta),
                "forward_relative":float(delta/full_norm),"forward_bound":bound,
                "gradient_absolute_l2":float(torch.linalg.vector_norm(grad_delta)),
                "gradient_relative_l2":float(torch.linalg.vector_norm(grad_delta)/torch.linalg.vector_norm(exact_vector))})
    return {"protocol":"exact-aw-forward-gradient-v1","valid":True,"dtype":"float64/complex128",
            "runtime_revision": RUNTIME_REVISION,
            "solve_backend": "real-block" if torch.device(device).type == "cuda" else "native",
            "cuda_linalg_preference": str(torch.backends.cuda.preferred_linalg_library())
                                      if torch.device(device).type == "cuda" else "not-used",
            "device":str(device),"rows":rows,"scope":"fixed graph/input/readout; derivatives with respect to input, edge displacements, frequencies, weights, z; not independent-training convergence or predictive accuracy"}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cpu");args=parser.parse_args()
    result=evaluate(args.device);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.4,2.8),layout="constrained")
    for decay in (.4,.6,.8,.95):
        rows=[r for r in result["rows"] if r["z"]==decay and r["K"]>0]
        for ax,key in zip(axes,("forward_relative","gradient_relative_l2")):
            ax.loglog([r["K"] for r in rows],[max(r[key],1e-16) for r in rows],marker="o",markersize=3,label=f"z={decay:g}")
    for ax,title in zip(axes,("Forward error","Gradient error")):
        ax.set(title=title,xlabel="Truncation depth K",ylabel="Relative error")
        ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.15)
    axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(args.output/"forward_gradient.pdf");fig.savefig(args.output/"forward_gradient.png",dpi=180)
    print(json.dumps({"valid":True,"points":len(result["rows"]),"output":str(args.output)}))


if __name__ == "__main__": main()
