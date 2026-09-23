"""Input-conditioned transport on a fixed diamond; not a full-network lower bound.

Only the normalized receiver power is a prediction. Context features enter
the field estimator, never the probe or readout. All four contexts share the
same topology, identities, nuisance features and probe within each quartet.
Static denotes an arbitrary trainable template fixed across inputs, which is
stronger than a prescribed topology-only field on this particular graph.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from route_phase_prediction import AW, atomic_json, state_hash, sha

ARMS = ('zero', 'static', 'gradient', 'unrestricted')
EDGE = torch.tensor([[0, 1, 1, 3, 0, 2, 2, 3], [1, 0, 3, 1, 2, 0, 3, 2]])
CONTEXT = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=torch.float64)


def generate(quartets, seed):
    g = torch.Generator().manual_seed(seed)
    roles = torch.eye(4, dtype=torch.float64).expand(quartets, 4, 4, 4)
    c = torch.zeros(quartets, 4, 4, 1, dtype=torch.float64)
    c[:, :, 1, 0], c[:, :, 2, 0] = CONTEXT[:, 0], CONTEXT[:, 1]
    noise = .1*torch.randn(quartets, 1, 4, 2, generator=g, dtype=torch.float64)
    x = torch.cat([roles, c, noise.expand(-1, 4, -1, -1)], -1).flatten(0, 1)
    angle = 2*math.pi*torch.rand(quartets, generator=g, dtype=torch.float64)
    amplitude = .5+torch.rand(quartets, generator=g, dtype=torch.float64)
    probe = (amplitude[:, None]*torch.stack([angle.cos(), angle.sin()], -1)).repeat_interleave(4, 0)
    return dict(x=x, probe=probe, y=torch.tensor([1., 0., 0., 1.], dtype=torch.float64).repeat(quartets))


def edges(n):
    return (EDGE[None]+4*torch.arange(n)[:, None, None]).permute(1, 0, 2).reshape(2, -1)


def power(displacement, probe):
    n = len(probe)
    signal = probe.new_zeros(n, 4, 2)
    signal[:, 3] = probe
    response = AW.truncated_walk_resolvent(
        signal.flatten(0, 1), edges(n), displacement.flatten(), probe.new_ones(1),
        z=probe.new_tensor(.8), num_steps=2).reshape(n, 4, 2)[:, 0]
    return response.square().sum(-1)/(probe.square().sum(-1)*(.8**2/2)**2)


class Predictor(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        self.arm = arm
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            if arm in ('gradient', 'unrestricted'):
                cls = AW.MatchedPotentialEdgeField if arm == 'gradient' else AW.AntisymmetricEdgeField
                self.field = cls(7, 32, math.pi).double()
            elif arm == 'static':
                self.template = nn.Parameter(.1*torch.randn(4, dtype=torch.float64))
            elif arm != 'zero':
                raise ValueError(arm)

    def displacement(self, x):
        if self.arm == 'zero':
            return x.new_zeros(len(x), 8)
        if self.arm == 'static':
            a = math.pi*torch.tanh(self.template/math.pi)
            return torch.stack([a, -a], -1).flatten().expand(len(x), -1)
        return self.field(x.flatten(0, 1), edges(len(x))).reshape(len(x), 8)

    def forward(self, x, probe, zero=False):
        a = self.displacement(x)
        return power(torch.zeros_like(a) if zero else a, probe)


@torch.no_grad()
def evaluate(model, data, zero=False):
    p = model(data['x'], data['probe'], zero=zero)
    return dict(accuracy=float(((p > .5) == data['y'].bool()).double().mean()),
                brier=float((p-data['y']).square().mean()),
                min_power=float(p.min()), max_power=float(p.max()))


def fit(arm, seed, root, *, data_seed, steps, lr, development=False):
    root.mkdir(parents=True, exist_ok=False)
    train, val = [generate(n, data_seed+i) for i, n in enumerate((128, 64))]
    model = Predictor(arm, seed)
    initial_hash = state_hash(model.state_dict())
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr) if params else None
    sampler = torch.Generator().manual_seed(seed+913581)
    best, best_state, best_step = evaluate(model, val)['brier'], copy.deepcopy(model.state_dict()), 0
    trajectory = []
    start = time.time()
    for step in range(1, steps+1):
        if optimizer is not None:
            q = torch.randint(128, (16,), generator=sampler)
            idx = (4*q[:, None]+torch.arange(4)).flatten()
            pred = model(train['x'][idx], train['probe'][idx])
            loss = (pred-train['y'][idx]).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 25 == 0 or step == steps:
            metrics = evaluate(model, val)
            trajectory.append(dict(step=step, validation=metrics))
            if metrics['brier'] < best:
                best, best_step, best_state = metrics['brier'], step, copy.deepcopy(model.state_dict())
    torch.save(dict(state_dict=model.state_dict(), step=steps), root/'latest.pt')
    model.load_state_dict(best_state)
    torch.save(dict(state_dict=best_state, step=best_step), root/'validation-best.pt')
    result = dict(arm=arm, seed=seed, steps=steps, lr=lr, data_seed=data_seed,
                  development=development, validation_best_step=best_step,
                  initial_sha256=initial_hash, parameters=sum(p.numel() for p in params),
                  validation=evaluate(model, val), elapsed_seconds=time.time()-start,
                  checkpoint_sha256=sha(root/'validation-best.pt'),
                  data_sha256=dict(train=state_hash(train), val=state_hash(val)))
    if not development:
        test = generate(256, data_seed+2)
        result.update(test=evaluate(model, test), zeroed_test=evaluate(model, test, zero=True))
        result['data_sha256']['test'] = state_hash(test)
    atomic_json(root/'trajectory.json', trajectory)
    atomic_json(root/'result.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--development', action='store_true')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=.01)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.development:
        fit('unrestricted', 17, args.root/'development', data_seed=26091417,
            steps=args.steps, lr=args.lr, development=True)
    else:
        args.root.mkdir(parents=True, exist_ok=False)
        protocol = dict(arms=ARMS, seeds=list(range(5)), data_seed=26091400,
                        steps=args.steps, lr=args.lr, selection='minimum validation Brier; first tie',
                        scope='fixed probe-to-power transport interface, no context-to-readout bypass',
                        source_sha256={str(q):sha(q) for q in (Path(__file__), Path(AW.__file__))})
        atomic_json(args.root/'protocol.json', protocol)
        results = [fit(arm, seed, args.root/f'{arm}-{seed}', data_seed=protocol['data_seed'],
                       steps=args.steps, lr=args.lr) for seed in range(5) for arm in ARMS]
        atomic_json(args.root/'results.json', results)


if __name__ == '__main__':
    main()
