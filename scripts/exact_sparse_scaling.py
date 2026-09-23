"""Resumable, isolated Exact/Sparse production-layer benchmark."""
import argparse, concurrent.futures, fcntl, hashlib, json, math, os
from pathlib import Path
import queue, statistics, subprocess, sys, threading, time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/"experiments/exact_sparse_scaling_20260907.json").read_text())
PROTOCOL=MANIFEST["protocol"]

def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    tmp.replace(path)

def specs():
    rows=[]
    for n in MANIFEST["table_nodes"]:
        for kernel in MANIFEST["kernels"]:
            for seed in MANIFEST["repeats"]:
                for method in ["nope","wire","exact","sparse"]:
                    rows.append(dict(suite="table4",n=n,degree=6,graphs=8,kernel=kernel,seed=seed,method=method,K=16))
    for n in MANIFEST["nodes"]:
        for degree in MANIFEST["degrees"]:
            for seed in MANIFEST["repeats"]:
                for method,k in [("exact",16)]+[("sparse",k) for k in MANIFEST["depths"]]:
                    rows.append(dict(suite="scaling",n=n,degree=degree,graphs=1,kernel="relu",seed=seed,method=method,K=k))
    for c in rows:
        c["id"]=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]
    return rows

def state_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def worker(c,root):
    import torch
    for p in [ROOT/"src",ROOT/"external/Graph-RoPE",ROOT/"scripts"]:
        sys.path.insert(0,str(p))
    from graphgps.layer.graphrope import GraphRoPE
    from graphgps.layer.aw_rope import AnalyticWalkRoPE,truncated_walk_resolvent
    from exact_aw_operator import install_exact_backend,exact_walk_resolvent,RUNTIME_REVISION
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(0)
    path=root/"cells"/c["id"]/"result.json"
    r=dict(protocol=PROTOCOL,config=c,status="running",runtime_revision=RUNTIME_REVISION,
           torch_version=torch.__version__,cuda_version=torch.version.cuda,
           device=torch.cuda.get_device_name(0),scope=MANIFEST["scope"])
    save(path,r)
    try:
        n,g,d=c["n"],c["graphs"],MANIFEST["width"]
        u=torch.arange(n); edges=[]
        for off in range(1,c["degree"]//2+1):
            v=(u+off)%n
            edges.extend([torch.stack((u,v)),torch.stack((v,u))])
        base=torch.cat(edges,1)
        edge=torch.cat([base+j*n for j in range(g)],1).cuda()
        gen=torch.Generator().manual_seed(20260907+c["seed"])
        x=torch.randn(g*n,d,generator=gen).cuda().requires_grad_(True)
        batch=torch.arange(g).repeat_interleave(n).cuda()
        coords=torch.zeros(g*n,10)
        if c["method"]=="wire":
            adj=torch.zeros(n,n,dtype=torch.float64)
            adj[base[0],base[1]]=1
            deg=adj.sum(1).clamp_min(1).rsqrt()
            _,vec=torch.linalg.eigh(torch.eye(n)-deg[:,None]*adj*deg[None,:])
            coords=torch.nn.functional.normalize(vec[:,:10].float(),dim=0).repeat(g,1)
        payload=SimpleNamespace(x=x,edge_index=edge,batch=batch,t=coords.cuda())
        cfg=SimpleNamespace(performer_kernel=c["kernel"],low_rank=None,num_steps=c["K"],
            z=MANIFEST["z"],z_values=(.2,.4,.6,.8),field_hidden_dim=1,max_displacement=math.pi,
            frequency_base=10000.,learnable_frequencies=True,learnable_z=True,
            normalize_resolvent=False,residual_mix=1.,preserve_input_norm=False,
            norm_group_size=0,field_type="local-antisymmetric",position_dim=3,rezero=False)
        torch.manual_seed(4400+c["seed"])
        method="aw" if c["method"] in ("exact","sparse") else c["method"]
        model=GraphRoPE(k=10 if method=="wire" else 0,d=d,num_heads=MANIFEST["heads"],
            dropout=0.,enable=method!="nope",init_omega="uniform",attn_type="Linear",
            positional_method=method,aw_cfg=cfg).cuda().train()
        r.update(initial_state_sha256=state_hash(model),parameters=sum(p.numel() for p in model.parameters()),
            input_sha256=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
            graph_sha256=hashlib.sha256(edge.cpu().numpy().tobytes()).hexdigest(),
            edges=edge.shape[1])
        if c["method"]=="exact":
            install_exact_backend(AnalyticWalkRoPE,GraphRoPE)
        save(path,r)
        def step():
            model.zero_grad(set_to_none=True); x.grad=None
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            loss=model(payload).square().mean()
            torch.cuda.synchronize(); middle=time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(); end=time.perf_counter()
            peak=torch.cuda.max_memory_allocated()
            if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
                raise FloatingPointError("nonfinite output/input gradient")
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("nonfinite parameter gradient")
            return dict(forward_ms=1000*(middle-start),backward_ms=1000*(end-middle),
                        total_ms=1000*(end-start),peak_bytes=peak)
        for _ in range(MANIFEST["warmup_steps"]): step()
        r["steps"]=[step() for _ in range(MANIFEST["measured_steps"])]
        r["summary"]={k:statistics.fmean(t[k] for t in r["steps"]) for k in r["steps"][0]}
        if c["method"]=="sparse" and n<=1024:
            with torch.no_grad():
                field=model.aw_rope.edge_field(x,edge)
                freq,z=model.aw_rope.frequencies,model.aw_rope.z
                a=exact_walk_resolvent(x,edge,field,freq,z=z,batch=batch)
                b=truncated_walk_resolvent(x,edge,field,freq,z=z,num_steps=c["K"])
                r["relative_transport_l2_error"]=float((a-b).norm()/a.norm().clamp_min(1e-12))
        r["status"]="complete"
    except torch.OutOfMemoryError as exc:
        r.update(status="cuda-oom",error=str(exc))
    except Exception as exc:
        r.update(status="error",error=repr(exc))
        save(path,r)
        raise
    r["completed_at"]=time.time(); save(path,r)

def aggregate(root):
    rows=[]; failures=[]; pairs={}
    for c in specs():
        p=root/"cells"/c["id"]/"result.json"
        if not p.exists(): continue
        r=json.loads(p.read_text())
        if r["config"]!=c or r["protocol"]!=PROTOCOL: raise ValueError("identity mismatch")
        rows.append(r)
        if r["status"]=="error": failures.append(c["id"])
        if r["status"]=="complete" and c["method"] in ("exact","sparse"):
            key=tuple(c[k] for k in ["suite","n","degree","graphs","kernel","seed"])
            fp=tuple(r[k] for k in ["initial_state_sha256","input_sha256","graph_sha256"])
            if key in pairs and pairs[key]!=fp: failures.append("pair mismatch "+str(key))
            pairs[key]=fp
    total=len(specs()); terminal=sum(r["status"] in ("complete","cuda-oom","timeout") for r in rows)
    result=dict(protocol=PROTOCOL,manifest=MANIFEST,rows=rows,failures=failures,total=total,
        terminal=terminal,complete=sum(r["status"]=="complete" for r in rows),
        valid=terminal==total and not failures)
    save(root/"aggregate.json",result)
    return result

def run(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/"writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=queue.Queue(); guard=threading.Lock(); active={}; aborted=threading.Event()
        for c in specs():
            p=root/"cells"/c["id"]/"result.json"
            if not p.exists() or json.loads(p.read_text())["status"] not in ("complete","cuda-oom","timeout"):
                tasks.put(c)
        def lane(gpu):
            while True:
                if aborted.is_set(): return
                try: c=tasks.get_nowait()
                except queue.Empty: return
                dest=root/"cells"/c["id"]; dest.mkdir(parents=True,exist_ok=True)
                save(dest/"config.json",c)
                with guard: active[gpu]=c
                env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
                cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"worker","--root",str(root),"--cell",json.dumps(c)]
                with (dest/"worker.log").open("a") as log:
                    try:
                        code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                                            timeout=MANIFEST["timeout_seconds"]).returncode
                        if code:
                            old=json.loads((dest/"result.json").read_text()) if (dest/"result.json").exists() else {}
                            if old.get("status")!="error":
                                save(dest/"result.json",dict(protocol=PROTOCOL,config=c,status="error",error="exit "+str(code)))
                    except subprocess.TimeoutExpired:
                        save(dest/"result.json",dict(protocol=PROTOCOL,config=c,status="timeout",
                            error="declared 900-second cell budget exceeded; not an OOM"))
                with guard:
                    active.pop(gpu,None); a=aggregate(root)
                    if a["failures"]: aborted.set()
                    progress={k:a[k] for k in ["complete","terminal","total","failures"]}
                    progress.update(time=time.time(),active=list(active.values()),pending=tasks.qsize())
                    save(root/"progress.json",progress)
                    print(json.dumps(progress),flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lane,range(4)))
        if not aggregate(root)["valid"]: raise RuntimeError("benchmark audit failed")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["plan","worker","run","aggregate"])
    p.add_argument("--root",type=Path,required=True); p.add_argument("--cell")
    a=p.parse_args()
    if a.action=="plan": print(json.dumps(dict(total=len(specs()),manifest=MANIFEST),indent=2))
    elif a.action=="worker":
        c=json.loads(a.cell)
        if c not in specs(): raise ValueError("undeclared cell")
        worker(c,a.root)
    elif a.action=="run": run(a.root)
    else: print(json.dumps({k:v for k,v in aggregate(a.root).items() if k not in ("rows","manifest")}))
