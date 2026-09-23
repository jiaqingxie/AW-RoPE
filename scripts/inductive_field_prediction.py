"""Anonymous unseen cycles: deterministic static-field obstruction.

No canonical node/edge IDs are supplied. A stronger randomized static field
and a nonlocal dynamic Flat control are kept outside the deterministic-field
theorem. All models contain a complete global attention block and classifier.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F
import full_network_field_prediction as full
from route_phase_prediction import AW, BudgetedNodeAdapter, atomic_json, state_hash, sha

ARMS=('unrestricted','gradient','zero','learned-static','random-static','projected-flat','wire','wire-mixing')
TRAIN_SIZES=(12,16,20,24,28,32)
VAL_SIZES=(14,18,22,26,30)
TEST_SIZES=(15,17,19,21,23,25,27,29,31)


def patterns(n):
    if n<12:raise ValueError('n >= 12 required')
    return tuple((0,)*(n-12)+p for p in full.PATTERNS)


def generate(pairs,seed,n):
    g=torch.Generator().manual_seed(seed)
    colors=torch.tensor(patterns(n));v=torch.arange(n)
    edge=torch.stack([v.repeat_interleave(2),torch.stack([(v+1)%n,(v-1)%n],-1).flatten()])
    theta=2*math.pi*v/n
    base_t=math.sqrt(2/n)*torch.stack([theta.cos(),theta.sin(),(2*theta).cos(),(2*theta).sin()],-1)
    xs,es,ts,random_fields=[],[],[],[]
    for _ in range(pairs):
        nuisance=.1*torch.randn(3,2,generator=g)
        x=torch.cat([torch.ones(2,n,1),F.one_hot(colors,3).float(),nuisance[colors]],-1)
        permutation=torch.randperm(n,generator=g);inverse=torch.argsort(permutation)
        t=base_t.clone()
        for block in (0,2):
            phi=2*math.pi*float(torch.rand((),generator=g));reflection=2*int(torch.randint(2,(),generator=g))-1
            t[:,block:block+2]=t[:,block:block+2]@torch.tensor([[math.cos(phi),-reflection*math.sin(phi)],[math.sin(phi),reflection*math.cos(phi)]])
        # Extra randomized control, independent of context and label. A draw
        # belongs to one paired graph instance; it provides no persistent IDs.
        a=torch.randn(n,generator=g)*.8
        a=torch.stack([a,-a.roll(1)],-1).flatten()
        xs.append(x[:,permutation]);es.append(inverse[edge].expand(2,-1,-1));ts.append(t[permutation].expand(2,-1,-1))
        random_fields.append(a.expand(2,-1))
    return dict(x=torch.cat(xs).double(),edges=torch.cat(es),t=torch.cat(ts).double(),
                template_sign=torch.tensor([1.,-1.],dtype=torch.float64).repeat(n).expand(2*pairs,-1),
                random_a=torch.cat(random_fields).double(),y=torch.tensor([0.,1.],dtype=torch.float64).repeat(pairs))


class StaticPolynomial(nn.Module):
    """All three hop matrices for Q and K: stronger than a fixed edge phase."""
    def __init__(self,d):
        super().__init__()
        self.matrices=nn.Parameter(torch.stack([torch.stack([torch.eye(d)*(.8**k) for k in range(3)]) for _ in range(2)]))


class RandomStatic(nn.Module):
    def __init__(self,d):
        super().__init__();self.frequencies=nn.Parameter(AW.standard_frequencies(d,10000.))


class Model(full.Model):
    def __init__(self,arm,seed=0,gain=3.,readout='mean'):
        if arm not in ARMS:raise ValueError(arm)
        base='zero' if arm in ('learned-static','random-static') else arm
        super().__init__(base,task='cycle',seed=seed,layers=1,gain=gain)
        self.field_arm=arm;self.readout_mode=readout
        if arm in ('learned-static','random-static'):
            budget=sum(p.numel() for p in self.adapters[0].parameters())
            with torch.random.fork_rng():
                torch.manual_seed(918273+seed)
                self.pos[0]=(StaticPolynomial(16) if arm=='learned-static' else RandomStatic(16)).double()
                count=sum(p.numel() for p in self.pos[0].parameters())
                self.adapters[0]=BudgetedNodeAdapter(16,budget-count).double()
                with torch.no_grad():
                    for p in self.adapters[0].parameters():
                        if p.ndim>=2:p.mul_(gain)
        self.audit=dict(total_parameters=sum(p.numel() for p in self.parameters()),
                        initial_sha256=state_hash(self.state_dict()),
                        shared_initial_sha256=state_hash({k:v for k,v in self.state_dict().items() if not k.startswith(('pos.','adapters.'))}))

    def pool(self,h,x):
        if self.readout_mode=='query':return (h*x[:,:,3:4]).sum(1)
        return h.mean(1)

    def forward(self,data,*,intervention=None):
        self._random_a=data['random_a']
        try:return super().forward(data,intervention=intervention)
        finally:del self._random_a

    def transform(self,q,k,h,edges,t,layer,intervention=None):
        if self.field_arm=='learned-static':
            source,target=edges;result=[]
            for i,value in enumerate((q,k)):
                state=value;out=state@self.pos[layer].matrices[i,0]
                for hop in (1,2):
                    state=torch.zeros_like(state).index_add_(0,source,state[target]/2)
                    out=out+state@self.pos[layer].matrices[i,hop]
                result.append(out)
            return tuple(result)
        if self.field_arm=='random-static':
            a=self._random_a
            if intervention=='zero-phase':a=torch.zeros_like(a)
            return tuple(AW.truncated_walk_resolvent(value,edges,a.flatten(),self.pos[layer].frequencies,
                         z=value.new_tensor(.8),num_steps=2) for value in (q,k))
        return super().transform(q,k,h,edges,t,layer,intervention)


@torch.no_grad()
def evaluate(model,data):
    by_size={}
    for n,d in data.items():
        metrics,_=full.evaluate(model,d);by_size[str(n)]=metrics
    return dict(accuracy=sum(x['accuracy'] for x in by_size.values())/len(by_size),
                bce=sum(x['bce'] for x in by_size.values())/len(by_size),by_size=by_size)


def fit(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=False)
    train={n:generate(64,args.data_seed+n,n) for n in TRAIN_SIZES}
    val={n:generate(32,args.data_seed+1000+n,n) for n in VAL_SIZES}
    model=Model(args.arm,args.seed,args.gain,args.readout)
    config={**vars(args),'out':str(args.out),'train_sizes':TRAIN_SIZES,'val_sizes':VAL_SIZES}
    atomic_json(args.out/'config.json',config)
    atomic_json(args.out/'audit.json',dict(**model.audit,source_sha256={str(p):sha(p) for p in (Path(__file__),Path(full.__file__),Path(full.r.__file__),full.r.LAYER)},train_hashes={str(n):state_hash(d) for n,d in train.items()},val_hashes={str(n):state_hash(d) for n,d in val.items()}))
    opt=torch.optim.Adam(model.parameters(),lr=args.lr);g=torch.Generator().manual_seed(args.seed+915141)
    best=None;start=time.monotonic()
    for step in range(args.steps+1):
        if step%100==0 or step==args.steps:
            metrics=evaluate(model,val);row=dict(step=step,validation=metrics,seconds=time.monotonic()-start)
            with (args.out/'trajectory.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
            if best is None or metrics['bce']<best['validation']['bce']:
                best=row;torch.save(model.state_dict(),args.out/'best.pt')
            atomic_json(args.out/'progress.json',dict(current=row,best=best,complete=False))
        if step==args.steps:break
        n=12 if step<args.warmup_steps else TRAIN_SIZES[int(torch.randint(len(TRAIN_SIZES),(),generator=g))]
        pairs=torch.randint(64,(16,),generator=g);idx=(2*pairs[:,None]+torch.arange(2)).flatten()
        opt.zero_grad();logits=model(full.batch(train[n],idx))
        loss=F.binary_cross_entropy_with_logits(logits,train[n]['y'][idx])
        if args.pair_scale:
            loss=loss+F.softplus(-args.pair_scale*(logits[1::2]-logits[::2])).mean()
        if not torch.isfinite(loss):raise FloatingPointError('training loss')
        loss.backward();opt.step()
    torch.save(model.state_dict(),args.out/'latest.pt')
    model.load_state_dict(torch.load(args.out/'best.pt',weights_only=True))
    result=dict(config=config,best=best,audit=model.audit,checkpoint_sha256=sha(args.out/'best.pt'))
    if args.formal:
        test={n:generate(128,args.data_seed+2000+n,n) for n in TEST_SIZES}
        result['test']=evaluate(model,test);result['test_hashes']={str(n):state_hash(d) for n,d in test.items()}
        result['extrapolation']=evaluate(model,{n:generate(64,args.data_seed+3000+n,n) for n in (48,64)})
    atomic_json(args.out/'result.json',result);atomic_json(args.out/'progress.json',dict(current=row,best=best,complete=True))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--data-seed',type=int,default=914172000);p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--gain',type=float,default=3.)
    p.add_argument('--readout',choices=['mean','query'],default='mean');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--warmup-steps',type=int,default=0,help='Shared curriculum: first train on the smallest training graph size.')
    p.add_argument('--pair-scale',type=float,default=0,help='Optional shared supervised pair-ranking loss, in addition to BCE.')
    p.add_argument('--formal',action='store_true');fit(p.parse_args())


if __name__=='__main__':main()
