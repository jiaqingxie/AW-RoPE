"""Check the small study's attention block against production GraphRoPE."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import route_phase_prediction as r


def production_class():
    layer_dir = ROOT/'external/Graph-RoPE/graphgps/layer'
    package = types.ModuleType('route_attention_reference')
    package.__path__ = [str(layer_dir)]
    sys.modules[package.__name__] = package
    name = package.__name__+'.graphrope'
    spec = importlib.util.spec_from_file_location(name, layer_dir/'graphrope.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphRoPE


@pytest.mark.parametrize('arm', r.ARMS+r.CONTROLS)
def test_production_attention_forward_and_vjp(arm):
    torch.set_num_threads(1)
    torch.manual_seed(74)
    model = r.RoutePredictor(arm, readout='attention', L=8, z=.8, seed=74).double()
    data = r.generate(3, 473, motifs=2, route_length=8)
    data = {k:v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k,v in data.items()}
    # Nonzero WIRE angles exercise the rotary action, not only initialization.
    if arm in ('wire', 'wire-mixing'):
        with torch.no_grad():
            model.pos[0].weight.normal_()
    data['x'].requires_grad_(True)
    actual = model(data)
    reference = production_class()(4, 16, 1, positional_method='none').double()
    reference.WQKV = model.qkv[0]
    reference.WO = model.output[0]
    reference.positional_method = 'aw' if arm in ('aw', 'gradient') else ('wire' if 'wire' in arm else 'none')
    if arm in ('aw', 'gradient'):
        reference.aw_rope = model.pos[0]
    if arm in ('wire', 'wire-mixing'):
        reference.OmegaQ = model.pos[0]
    b,n,_ = data['x'].shape
    edge = r.flatten_edges(data['edges'], n)
    h = model.adapters[0](model.encoder(data['x']).flatten(0,1))
    hook = None
    if arm in ('mixing','wire-mixing'):
        def mix_qk(module, inputs, projected):
            q,k,v = projected.chunk(3,-1)
            mq,mk = r.phase_free(torch.cat((q,k),-1),edge,8,.8).chunk(2,-1)
            return torch.cat((mq,mk,v),-1)
        hook = reference.WQKV.register_forward_hook(mix_qk)
    try:
        output = reference(types.SimpleNamespace(x=h, t=data['t'].flatten(0,1),
                           edge_index=edge, batch=torch.arange(b).repeat_interleave(n)))
    finally:
        if hook is not None:
            hook.remove()
    hidden = model.norm[0](h+output)
    hidden = hidden+model.ff[0](hidden)
    root = (hidden.reshape(b,n,16)*data['x'][:,:,:1]).sum(1)
    expected = model.head(model.final_norm(root)).squeeze(-1)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    variables = (data['x'],)+tuple(model.parameters())
    left = torch.autograd.grad(actual.sum(), variables)
    right = torch.autograd.grad(expected.sum(), variables)
    for a,b in zip(left,right):
        torch.testing.assert_close(a,b,atol=2e-11,rtol=2e-10)
