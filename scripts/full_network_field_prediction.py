"""Full attention tests for context-dependent fields, including bypass controls.

The cycle task has a one-block zero/local-gradient collision proof. That
proof does not cover more blocks, arbitrary fixed edge templates, spectral
coordinates, or nonlocal potential estimators. These are explicit controls.
The diamond task restores unrestricted feature access to the earlier toy.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import route_phase_prediction as r

ARMS = ('zero', 'static', 'static-template', 'gradient', 'unrestricted', 'projected-flat', 'wire', 'wire-mixing')
PATTERNS = ((0,1,2,0,0,1,1,0,0,1,0,0), (0,1,1,0,0,1,2,0,0,1,0,0))


def generate(pairs, seed, task='cycle'):
    g = torch.Generator().manual_seed(seed)
    n = 12 if task == 'cycle' else 4
    if task == 'cycle':
        v = torch.arange(n)
        edge = torch.stack([v.repeat_interleave(2), torch.stack([(v+1)%n, (v-1)%n], -1).flatten()])
        # Index each undirected edge by its clockwise source, with signed reverse.
        template_id = torch.stack([v, (v-1)%n], -1).flatten()
        template_sign = torch.tensor([1., -1.]).repeat(n)
        angle = 2*math.pi*v/n
        coordinates = math.sqrt(2/n)*torch.stack([angle.cos(), angle.sin(), (2*angle).cos(), (2*angle).sin()], -1)
    else:
        edge = torch.tensor([[0,1,1,3,0,2,2,3], [1,0,3,1,2,0,3,2]])
        template_id = torch.arange(4).repeat_interleave(2)
        template_sign = torch.tensor([1., -1.]).repeat(4)
        adjacency = torch.zeros(n,n).index_put(tuple(edge), torch.ones(edge.shape[1]))
        _, coordinates = torch.linalg.eigh(torch.diag(adjacency.sum(-1))-adjacency)
    xs, es, ts, ids, signs = [], [], [], [], []
    for _ in range(pairs):
        if task == 'cycle':
            colors = torch.tensor(PATTERNS)
            # Per-color continuous context is identical within every pair.
            nuisance = .1*torch.randn(3,2,generator=g)
            x = torch.cat([torch.ones(2,n,1), F.one_hot(colors,3).float(), nuisance[colors]], -1)
        else:
            # Opposite-label cases have identical role identities and nuisance.
            a = float(2*torch.randint(2,(),generator=g)-1)
            x = torch.zeros(2,n,6);x[:,:,:4] = torch.eye(n)
            x[:,1,4] = a;x[0,2,4] = -a;x[1,2,4] = a
            x[:,:,5] = .1*torch.randn(1,n,generator=g)
        permutation = torch.randperm(n,generator=g)
        inverse = torch.argsort(permutation)
        t = coordinates.clone()
        if task == 'cycle':
            for block in (0,2):
                theta = 2*math.pi*float(torch.rand((),generator=g))
                t[:,block:block+2] = t[:,block:block+2] @ torch.tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        xs.append(x[:,permutation]); es.append(inverse[edge].expand(2,-1,-1))
        ts.append(t[permutation].expand(2,-1,-1));ids.append(template_id.expand(2,-1));signs.append(template_sign.expand(2,-1))
    result = dict(x=torch.cat(xs).double(), edges=torch.cat(es), t=torch.cat(ts).double(),
                  template_id=torch.cat(ids), template_sign=torch.cat(signs).double(),
                  y=torch.tensor([0.,1.],dtype=torch.float64).repeat(pairs))
    return result


class StaticTemplate(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.edge_phase = nn.Parameter(.3*torch.randn(n))
        self.frequencies = nn.Parameter(torch.logspace(0,-3,d//2))


class Model(r.RoutePredictor):
    def __init__(self, arm, *, task='cycle', seed=0, layers=1, gain=3.):
        base = {'zero':'mixing', 'static':'mixing', 'static-template':'mixing',
                'unrestricted':'aw', 'projected-flat':'aw'}.get(arm,arm)
        super().__init__(base, readout='attention',d=16,L=2,z=.8,layers=layers,seed=seed)
        self.field_arm, self.task = arm, task
        if arm == 'static-template':
            n = 12 if task=='cycle' else 4
            # Equal total parameter budget with active pointwise compensation.
            with torch.random.fork_rng():
                torch.manual_seed(918273+seed)
                for layer in range(layers):
                    budget = sum(p.numel() for p in self.adapters[layer].parameters())
                    self.pos[layer] = StaticTemplate(n,16)
                    count = sum(p.numel() for p in self.pos[layer].parameters())
                    self.adapters[layer] = r.BudgetedNodeAdapter(16,budget-count)
        if arm == 'projected-flat':
            assert task=='cycle'
        with torch.no_grad():
            for name,p in self.named_parameters():
                if p.ndim >= 2: p.mul_(gain)
        self.double()
        self.audit = dict(total_parameters=sum(p.numel() for p in self.parameters()),
                          initial_sha256=r.state_hash(self.state_dict()),gain=gain,
                          shared_initial_sha256=r.state_hash({k:v for k,v in self.state_dict().items() if not k.startswith(('pos.','adapters.'))}))

    def pool(self,h,x):
        return h.mean(1)

    def forward(self, data, *, intervention=None):
        x,t = data['x'],data['t'];b,n,_ = x.shape
        edge = r.flatten_edges(data['edges'],n)
        h = self.encoder(x).flatten(0,1)
        handles=[]
        if self.field_arm=='projected-flat' and intervention is None:
            # On a simple cycle, remove the sole harmonic circulation component.
            # This field is input-dependent and nonlocal, stronger than local gradient.
            def project(_module,_inputs,a):
                a=a.reshape(b,2*n)
                circulation=a[:,::2].sum(-1,keepdim=True)
                return (a-(circulation/n)*data['template_sign']).flatten()
            handles=[m.edge_field.register_forward_hook(project) for m in self.pos]
        try:
            for layer in range(self.layers):
                h=self.adapters[layer](h)
                q,k,v=self.qkv[layer](h).chunk(3,-1)
                if self.field_arm=='static-template':
                    pos=self.pos[layer]
                    a=pos.edge_phase[data['template_id']]*data['template_sign']
                    if intervention=='zero-phase':a=torch.zeros_like(a)
                    q,k=[r.AW.truncated_walk_resolvent(value,edge,a.flatten(),pos.frequencies,z=value.new_tensor(.8),num_steps=2) for value in (q,k)]
                else:
                    q,k=self.transform(q,k,h,edge,t.flatten(0,1),layer,intervention)
                q,k,v=[a.reshape(b,n,self.d) for a in (q,k,v)]
                attn=torch.softmax(q@k.transpose(-1,-2)/math.sqrt(self.d),-1)
                h=self.norm[layer](h+self.output[layer]((attn@v).flatten(0,1)))
                h=h+self.ff[layer](h)
            pooled=self.pool(h.reshape(b,n,self.d),x)
            return self.head(self.final_norm(pooled)).squeeze(-1)
        finally:
            for handle in handles:handle.remove()


def batch(d,idx):return {k:v[idx] for k,v in d.items()}


@torch.no_grad()
def evaluate(model,data,intervention=None):
    logits=torch.cat([model(batch(data,slice(i,i+128)),intervention=intervention) for i in range(0,len(data['y']),128)])
    labels=data['y']
    gap=logits[1::2]-logits[::2]
    return dict(accuracy=float(((logits>=0)==labels.bool()).double().mean()),
                bce=float(F.binary_cross_entropy_with_logits(logits,labels)),
                max_pair_logit_gap=float(gap.abs().max()),mean_signed_gap=float(gap.mean())),logits


def fit(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=False)
    config={**vars(args),'out':str(args.out)}
    r.atomic_json(args.out/'config.json',config)
    train,val=[generate(n,args.data_seed+i,args.task) for i,n in enumerate((128,64))]
    model=Model(args.arm,task=args.task,seed=args.seed,layers=args.layers,gain=args.gain)
    r.atomic_json(args.out/'audit.json',dict(**model.audit,data_sha256={k:r.state_hash(d) for k,d in [('train',train),('val',val)]},source_sha256={str(p):r.sha(p) for p in [Path(__file__),Path(r.__file__),r.LAYER]}))
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    g=torch.Generator().manual_seed(args.seed+751920)
    best=None;start=time.monotonic()
    for step in range(args.steps+1):
        if step%100==0 or step==args.steps:
            metrics,_=evaluate(model,val)
            row=dict(step=step,validation=metrics,seconds=time.monotonic()-start)
            with (args.out/'trajectory.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if best is None or metrics['bce']<best['validation']['bce']:
                best=row;torch.save(model.state_dict(),args.out/'best.pt')
            r.atomic_json(args.out/'progress.json',dict(current=row,best=best,complete=False))
        if step==args.steps:break
        pairs=torch.randint(128,(16,),generator=g);idx=(2*pairs[:,None]+torch.arange(2)).flatten()
        opt.zero_grad();logits=model(batch(train,idx));loss=F.binary_cross_entropy_with_logits(logits,train['y'][idx])
        if not torch.isfinite(loss):raise FloatingPointError('training loss')
        loss.backward();opt.step()
    torch.save(model.state_dict(),args.out/'latest.pt')
    model.load_state_dict(torch.load(args.out/'best.pt',weights_only=True))
    result=dict(config=config,best=best,validation=evaluate(model,val)[0],audit=model.audit,checkpoint_sha256=r.sha(args.out/'best.pt'))
    if args.formal:
        test=generate(256,args.data_seed+2,args.task)
        result['test'],logits=evaluate(model,test)
        result['test_zero_phase']=evaluate(model,test,intervention='zero-phase')[0]
        result['test_tensor_sha256']=r.state_hash(test)
        np.savez_compressed(args.out/'predictions.npz',logits=logits.numpy(),labels=test['y'].numpy())
    r.atomic_json(args.out/'result.json',result)
    r.atomic_json(args.out/'progress.json',dict(current=row,best=best,complete=True))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--task',choices=['diamond','cycle'],default='cycle')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--data-seed',type=int,default=91417100)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--layers',type=int,default=1);p.add_argument('--gain',type=float,default=3.)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--formal',action='store_true')
    fit(p.parse_args())


if __name__=='__main__':main()
