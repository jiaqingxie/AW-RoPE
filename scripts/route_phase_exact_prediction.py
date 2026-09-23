"""Exact-AW extension of the unchanged route-parity task and training loop.

Replace only the Sparse finite walk action by the production complete
resolvent. The frozen model also uses the complete action when its learned
phases are zeroed; this intervention must not silently switch to Sparse.
"""
from pathlib import Path
import types

import torch
import route_phase_prediction as base
from exact_aw_operator import exact_walk_resolvent

SparseRoutePredictor = base.RoutePredictor


def exact_transport(module, x, edge_index, displacement, edge_weight,
                    reverse_edge, frequencies=None):
    if module.method != 'aw' or module.normalize_resolvent:
        raise ValueError('Exact extension requires ordinary unnormalized AW')
    return exact_walk_resolvent(x, edge_index, displacement,
                module.frequencies if frequencies is None else frequencies,
                z=module.z, batch=module._route_exact_batch, edge_weight=edge_weight)


class ExactRoutePredictor(SparseRoutePredictor):
    def __init__(self, arm='aw', **kwargs):
        if arm != 'aw': raise ValueError('only AW is extended')
        super().__init__(arm, **kwargs)
        for pos in self.pos:
            pos._transport_single = types.MethodType(exact_transport, pos)
            pos._route_exact_batch = None

    def forward(self, data, *, intervention=None):
        b, n, _ = data['x'].shape
        graph_ids = torch.arange(b,device=data['x'].device).repeat_interleave(n)
        for pos in self.pos: pos._route_exact_batch = graph_ids
        try:
            return super().forward(data,intervention=intervention)
        finally:
            for pos in self.pos: pos._route_exact_batch = None

    def transform(self,q,k,h,edges,t,layer,intervention=None):
        if intervention == 'zero-phase':
            pos = self.pos[layer]
            return tuple(exact_walk_resolvent(x,edges,x.new_zeros(edges.shape[1]),
                         pos.frequencies,z=pos.z,batch=pos._route_exact_batch) for x in (q,k))
        return super().transform(q,k,h,edges,t,layer,intervention)


def main():
    original_fit = base.fit
    def fit(args):
        args.transport_backend = 'complete-resolvent'
        original_fit(args)
        result_path = args.out/'result.json'
        import json
        result = json.loads(result_path.read_text())
        result['exact_extension'] = dict(operator='(I-z*T)^(-1) applied by linear solve',
            zero_phase_intervention='same complete resolvent with zero edge phases',
            extension_sha256=base.sha(__file__),
            exact_operator_sha256=base.sha(Path(__file__).with_name('exact_aw_operator.py')))
        base.atomic_json(result_path,result)
    base.RoutePredictor = ExactRoutePredictor
    base.fit = fit
    base.main()


if __name__ == '__main__': main()
